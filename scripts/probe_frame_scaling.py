#!/usr/bin/env python
"""One dense inference pass over N consecutive frames; record time and peak GPU memory.

Same interface as the U4DR probe (unified4DReconstructor/scripts/eval/probe_frame_scaling.py)
so the two can be compared run for run. Two failure kinds are results, not crashes:
"oom", and "limit" -- the checkpoint's fixed timestep table caps the clip length, which
is a property of the model, not of the GPU.

    uv run python scripts/probe_frame_scaling.py \
        --ckpt checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
        --frames <dir of jpg> --n 32 --out frame_scaling.json --label OpenD4RT-32
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
from src.eval.dense_pointmap import (
    dense_depth_and_flow,
    load_checkpoint,
    max_clip_frames,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--size", type=int, default=256, help="square input, as model.yaml image_size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.frames).glob("*.jpg"))[:args.n]
    rec = {"label": args.label, "n": args.n, "input_hw": [args.size, args.size], "ckpt": args.ckpt}
    try:
        model = load_checkpoint(args.ckpt, "cuda")
        torch.cuda.synchronize()
        rec["weights_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        rec["clip_limit"] = max_clip_frames(model)
        frames = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB")
                                              .resize((args.size, args.size)))).permute(2, 0, 1)
                  for p in paths]
        video = (torch.stack(frames).float() / 255.0)[None]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            dense_depth_and_flow(model, video)
        torch.cuda.synchronize()
        rec.update(status="ok", seconds=round(time.time() - t0, 3))
    except torch.OutOfMemoryError as exc:
        rec.update(status="oom", message=str(exc).splitlines()[0])
    except ValueError as exc:
        if "takes at most" not in str(exc):
            raise
        rec.update(status="limit", message=str(exc))
    rec.update(torch_peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20),
               torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20))
    out = Path(args.out)
    runs = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps([*runs, rec], indent=2))
    print(f"[probe] {json.dumps(rec)}", flush=True)


if __name__ == "__main__":
    main()
