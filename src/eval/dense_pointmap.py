"""Dense per-pixel inference: turn OpenD4RT's query API into depth / flow maps.

OpenD4RT is query-based. Given ``(u, v, t_src, t_tgt, t_cam)`` it returns ``xyz_3d``,
``uv_2d``, ``visibility``, ``displacement``, ``normal`` and ``confidence`` for those
queries -- there is no dense head, so nothing here produces a depth map directly.

Querying *every* pixel of a frame closes that gap:

* ``t_tgt == t_src`` and ``t_cam == t_src`` gives the frame's own pointmap in its own
  camera frame, so the Z channel of ``xyz_3d`` **is** metric depth. (Camera-frame
  output is the convention this repo's WorldTrack evaluation already uses.)
* ``t_tgt == t_src + 1`` gives where each pixel lands in the next frame, so
  ``uv_2d - (u, v)`` **is** optical flow in pixels.

This lives here, next to the model it understands, rather than in a downstream
consumer: the query conventions, the head names and the clip-length limit are all
facts about OpenD4RT. Consumers get plain tensors and need no knowledge of them.

Returned tensors use ``(N, C, H, W)`` with N = frames, which is the ordinary video
layout; a consumer wanting a batch axis adds it.

    from src.eval.dense_pointmap import dense_depth_and_flow
    out = dense_depth_and_flow(model, video)      # video: (1, N, 3, H, W)
    out["depth"]                                   # (N, 1, H, W) metres
    out["flow"]                                    # (N, 2, H, W) pixels, last frame 0
"""

from __future__ import annotations

from typing import Any

import torch

from src.eval.tasks import (_encode_model_memory, _model_clip_frames,
                            _run_model_for_queries)


def max_clip_frames(model: Any) -> int:
    """Longest clip this checkpoint can take (its timestep embedding is a fixed table)."""
    return _model_clip_frames(model)


def min_clip_frames(model: Any) -> int:
    """Shortest clip the video encoder accepts = its TEMPORAL patch size.

    The encoder patchifies with ``patch_size_t_h_w`` (2 x 16 x 16 for the released
    checkpoints), so a single frame cannot be encoded at all: conv3d raises "Kernel size
    can't be greater than actual input size". Callers feeding a mixed corpus hit this on
    any single-image dataset, where the failure names a kernel rather than the clip.
    """
    cached = getattr(model, "module", model)
    for attr in ("encoder", "video_encoder"):
        enc = getattr(cached, attr, None)
        patch = getattr(enc, "patch_size_t_h_w", None) if enc is not None else None
        if patch:
            return max(1, int(patch[0]))
    return 2                                    # released checkpoints all use t=2


@torch.no_grad()
def dense_depth_and_flow(model: Any, video: torch.Tensor,
                         query_chunk: int = 4096) -> dict[str, torch.Tensor]:
    """Per-pixel depth and forward flow for one clip.

    ``video``: ``(1, N, 3, H, W)``. Returns ``depth`` ``(N, 1, H, W)`` in metres and
    ``flow`` ``(N, 2, H, W)`` in pixels, the final frame's flow being zero because it
    has no successor.
    """
    if video.dim() != 5 or video.shape[0] != 1:
        raise ValueError(f"expected one clip shaped (1, N, 3, H, W), got {tuple(video.shape)}")
    n, _, h, w = video.shape[1:]
    limit = max_clip_frames(model)
    if n > limit:
        # The query embedder's timestep embedding is a learned table of fixed length, so
        # t_src beyond it indexes out of range. Fail here, naming the clip, rather than
        # deep inside the model where the error names an embedding.
        raise ValueError(f"clip has {n} frames; this checkpoint takes at most {limit}")

    device = next(model.parameters()).device
    video = video.to(device)
    # Pad a too-short clip by repeating its last frame, then trim the results back. A
    # single-frame input is common in mixed corpora (image-only datasets) and would
    # otherwise abort the whole evaluation on an unrelated-looking conv3d error.
    n_real, min_t = n, min_clip_frames(model)
    if n < min_t:
        video = torch.cat([video, video[:, -1:].expand(-1, min_t - n, -1, -1, -1)], dim=1)
        n = min_t
    # Queries are NORMALISED to [0, 1] -- see model/query_embedding.py ("uv: [B, M, 2],
    # normalized to [0, 1]"), and the repo's own caller names the argument
    # `query_uv_norm`. Passing raw pixel coordinates silently sends every query far
    # outside the image: the model still returns values, so nothing errors, but the
    # predictions are meaningless (measured EPE ~330 px before this was fixed).
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    u = (xs.reshape(-1).float() / max(w - 1, 1)).to(device)
    v = (ys.reshape(-1).float() / max(h - 1, 1)).to(device)
    uv_norm = torch.stack([u, v], dim=-1).cpu()
    # ...and uv_2d comes back in the same normalised space, so the flow it implies is a
    # fraction of the frame; scale to pixels for a metric expressed in pixels.
    scale = torch.tensor([max(w - 1, 1), max(h - 1, 1)], dtype=torch.float32)
    memory = _encode_model_memory(model, video, None)

    depth, flow = [], []
    for t in range(n):
        ts = torch.full_like(u, float(t))
        same = _run_model_for_queries(
            model, video, None,
            {"u": u, "v": v, "t_src": ts, "t_tgt": ts, "t_cam": ts}, query_chunk, memory)
        depth.append(same["xyz_3d"][:, 2].reshape(1, h, w).clamp_min(0))
        if t + 1 < n:
            nxt = _run_model_for_queries(
                model, video, None,
                {"u": u, "v": v, "t_src": ts, "t_tgt": ts + 1, "t_cam": ts},
                query_chunk, memory)
            d = (nxt["uv_2d"] - uv_norm) * scale            # [0,1] -> pixels
            flow.append(d.reshape(1, h, w, 2).permute(0, 3, 1, 2)[0])
        else:
            flow.append(torch.zeros(2, h, w))

    # Stack along a NEW leading frame axis. Stacking into an existing axis yields
    # (1, N, H, W) -- the same element count, so nothing raises, but it presents N
    # frames as N channels of one frame and every downstream metric reads garbage.
    return {"depth": torch.stack(depth, 0)[:n_real],
            "flow": torch.stack(flow, 0)[:n_real]}


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Build the model from the ``model.yaml`` beside ``ckpt_path`` and load weights."""
    from pathlib import Path

    import yaml

    from src.model.builder import build_model

    ck = Path(ckpt_path)
    cfg = yaml.safe_load((ck.parent / "model.yaml").read_text())
    model = build_model(cfg["model"])
    # weights_only=True is sufficient here (top-level keys are model/optimizer/
    # scheduler/scaler/global_step/best_val, all tensors and scalars) and must stay:
    # these checkpoints are ~14 GB downloads from the Hub, and weights_only=False
    # would unpickle arbitrary code out of them.
    payload = torch.load(ck, map_location="cpu", weights_only=True, mmap=True)
    missing, unexpected = model.load_state_dict(payload.get("model", payload), strict=False)
    if missing or unexpected:
        print(f"[opend4rt] loaded with {len(missing)} missing / {len(unexpected)} unexpected keys")
    return model.eval().to(device)
