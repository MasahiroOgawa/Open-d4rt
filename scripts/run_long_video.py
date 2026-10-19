#!/usr/bin/env python
"""Run OpenD4RT over an arbitrarily long video in consecutive clips, streaming to disk.

Same clip layout and npz format as unified4DReconstructor/scripts/eval/run_long_video.py,
so the methods' per-frame outputs line up frame for frame. Each clip goes to
``<out>/clip_XXXXX.npz`` when it finishes; a clip already on disk is skipped (resume).

    uv run python scripts/run_long_video.py \
        --ckpt checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
        --frames <dir of jpg> --window 8 --out <dir>

Each npz: frames (n,) int32; depth (n,H,W) f16 metres; flow (n,2,H,W) f16 pixels (a
clip's last frame is 0 -- it has no successor inside the clip).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dense_pointmap import dense_depth_and_flow, load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--size", type=int, default=256, help="square input, as model.yaml image_size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.frames).glob("*.jpg"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [list(range(i, min(i + args.window, len(paths)))) for i in range(0, len(paths), args.window)]
    (out / "source.json").write_text(json.dumps(
        {"method": "OpenD4RT", "ckpt": str(Path(args.ckpt).resolve()), "frames": args.frames,
         "n_frames": len(paths), "window": args.window, "input_hw": [args.size, args.size]}, indent=2))
    print(f"[long] OpenD4RT {Path(args.ckpt).parent.name}: {len(paths)} frames, {len(clips)} clips "
          f"of {args.window} @ {args.size}x{args.size} -> {out}", flush=True)

    model = load_checkpoint(args.ckpt, "cuda")
    t0, done = time.time(), 0
    for ci, idx in enumerate(clips):
        dest = out / f"clip_{ci:05d}.npz"
        if dest.exists():
            continue
        video = torch.stack([torch.from_numpy(np.asarray(
            Image.open(paths[i]).convert("RGB").resize((args.size, args.size)))).permute(2, 0, 1)
            for i in idx]).float().div(255.0)[None]
        with torch.no_grad():
            d = dense_depth_and_flow(model, video)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, frames=np.asarray(idx, np.int32),
                            depth=d["depth"][:, 0].half().cpu().numpy(),
                            flow=d["flow"].half().cpu().numpy())
        tmp.rename(dest)
        done += 1
        if done % 25 == 0 or ci == len(clips) - 1:
            el = time.time() - t0
            left = sum(not (out / f"clip_{k:05d}.npz").exists() for k in range(ci + 1, len(clips)))
            print(f"[long] clip {ci + 1}/{len(clips)} elapsed {el / 60:.1f} min, "
                  f"ETA {el / done * left / 60:.1f} min", flush=True)
    print(f"[long] DONE {len(clips)} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
