#!/usr/bin/env python3
"""Compare official packed embeddings vs llama.cpp clip dump."""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="ref .ptemb.npy [T,C]")
    ap.add_argument("--clip", required=True, help="dump .clip")
    args = ap.parse_args()

    pt = np.load(args.pt)
    with open(args.clip, "rb") as f:
        n_tok, n_embd = np.fromfile(f, dtype=np.int32, count=2)
        clip = np.fromfile(f, dtype=np.float32)
    clip = clip.reshape(int(n_tok), int(n_embd))
    print(f"pt   {pt.shape} mean={pt.mean():.6f} std={pt.std():.6f}")
    print(f"clip {clip.shape} mean={clip.mean():.6f} std={clip.std():.6f}")
    if pt.shape != clip.shape:
        print("SHAPE MISMATCH", file=sys.stderr)
        t = min(pt.shape[0], clip.shape[0])
        c = min(pt.shape[1], clip.shape[1])
        pt = pt[:t, :c]
        clip = clip[:t, :c]
        print(f"comparing truncated {pt.shape}")

    diff = clip - pt
    absd = np.abs(diff)
    # cosine per token
    pt_n = pt / (np.linalg.norm(pt, axis=1, keepdims=True) + 1e-12)
    cl_n = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-12)
    cos = (pt_n * cl_n).sum(axis=1)
    print(f"max|diff|={absd.max():.6g} mean|diff|={absd.mean():.6g}")
    print(f"cosine min={cos.min():.6f} mean={cos.mean():.6f} median={np.median(cos):.6f}")
    worst = int(np.argmin(cos))
    print(f"worst token {worst} cos={cos[worst]:.6f}  pt[:8]={pt[worst, :8]} clip[:8]={clip[worst, :8]}")
    print(f"token0 cos={cos[0]:.6f} token1 cos={cos[1]:.6f} last cos={cos[-1]:.6f}")
    ok = bool(cos.mean() > 0.99 and cos.min() > 0.95)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
