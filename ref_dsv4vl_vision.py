#!/usr/bin/env python3
"""CPU reference: official ViT+aligner+N-layout vs a dump from llama.cpp clip.

Writes:
  <out>.ptemb  float32 [T, 4096]
  <out>.hwc    int32 nx, ny + float32 HWC already normalized (x-0.5)/0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open

HF = Path("/models/dsv4-vision-exp-hf")
sys.path.insert(0, str(HF / "inference"))

from image_processor import (  # noqa: E402
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    build_image_block,
    load_image,
)
from vision import Aligner, ViT  # noqa: E402


def cfg_to_args(cfg: dict) -> types.SimpleNamespace:
    ns = types.SimpleNamespace(**cfg)
    ns.dim = int(cfg["hidden_size"])
    return ns


def load_modules(args, shard: Path) -> tuple[ViT, Aligner, dict[str, torch.Tensor]]:
    vit = ViT(args)
    aligner = Aligner(args)
    sent: dict[str, torch.Tensor] = {}
    sd_v: dict[str, torch.Tensor] = {}
    sd_a: dict[str, torch.Tensor] = {}
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k).float()
            if k.startswith("vision."):
                sd_v[k[len("vision.") :]] = t
            elif k.startswith("aligner."):
                sd_a[k[len("aligner.") :]] = t
            elif k in ("image_start", "image_end", "image_newline", "image_pad"):
                sent[k] = t
    vit.load_state_dict(sd_v)
    aligner.load_state_dict(sd_a)
    vit.eval()
    aligner.eval()
    return vit, aligner, sent


def pack(aligner_out: torch.Tensor, types: torch.Tensor, perm: torch.Tensor, sent: dict[str, torch.Tensor]) -> torch.Tensor:
    # aligner_out: [n_llm, 4096] in raster order
    img = aligner_out[perm]
    img_i = 0
    rows = []
    for ty in types.tolist():
        if ty == IMAGE:
            rows.append(img[img_i])
            img_i += 1
        elif ty == IMAGE_START:
            rows.append(sent["image_start"])
        elif ty == IMAGE_END:
            rows.append(sent["image_end"])
        elif ty == IMAGE_NEW_LINE:
            rows.append(sent["image_newline"])
        elif ty == IMAGE_PAD:
            rows.append(sent["image_pad"])
        else:
            raise ValueError(ty)
    return torch.stack(rows, dim=0)


def padded_hwc_from_load(record, args) -> tuple[np.ndarray, int, int, int, int, torch.Tensor]:
    """Re-run load_image but also return the normalized HWC f32 of the padded RGB."""
    from image_processor import safe_resize
    import math
    from PIL import Image, ImageOps
    import io
    from image_processor import load_image_bytes

    p = args.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    width, height = image.size
    orig = (width, height)
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    arr = np.asarray(image, dtype=np.float32)  # HWC 0..255
    hwc = (arr / 255.0 - 0.5) / 0.5
    x = torch.from_numpy(hwc).permute(2, 0, 1)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    print(
        f"orig={orig} padded={best_width}x{best_height} n_vit={n_vit_w}x{n_vit_h} n_llm={n_llm_w}x{n_llm_h}",
        flush=True,
    )
    return hwc, n_vit_h, n_vit_w, n_llm_h, n_llm_w, patches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HF))
    ap.add_argument("--image", default=str(HF / "carrots.jpeg"))
    ap.add_argument("--out", default="/tmp/dsv4vl-cpu/ref")
    ap.add_argument("--start-pos", type=int, default=3, help="prompt pos; 3 => compress_pad=0")
    args_cli = ap.parse_args()

    src = Path(args_cli.src)
    cfg = json.loads((src / "config.json").read_text())
    args = cfg_to_args(cfg)
    shard = src / "model-00001-of-00048.safetensors"
    vit, aligner, sent = load_modules(args, shard)

    hwc, n_vit_h, n_vit_w, n_llm_h, n_llm_w, patches = padded_hwc_from_load(
        {"url": args_cli.image}, args
    )
    types, perm = build_image_block(n_llm_h, n_llm_w, args_cli.start_pos)
    print(f"types={len(types)} image_slots={int((types == IMAGE).sum())} start_pos={args_cli.start_pos}", flush=True)

    with torch.no_grad():
        hid = vit(patches.float(), n_vit_h, n_vit_w)
        aln = aligner(hid, n_vit_h, n_vit_w)
        packed = pack(aln, types, perm, sent)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb = packed.cpu().numpy().astype(np.float32)
    np.save(str(out) + ".ptemb.npy", emb)
    # raw dump for the C++ tool: nx, ny, HWC f32
    ny, nx = hwc.shape[0], hwc.shape[1]
    with open(str(out) + ".hwc", "wb") as f:
        np.array([nx, ny], dtype=np.int32).tofile(f)
        np.ascontiguousarray(hwc, dtype=np.float32).tofile(f)
    print(f"wrote {out}.ptemb.npy {emb.shape}  {out}.hwc {nx}x{ny}", flush=True)
    print(f"token0[:8]={emb[0, :8]}", flush=True)
    print(f"token1[:8]={emb[1, :8]}", flush=True)
    print(f"stats mean={emb.mean():.6f} std={emb.std():.6f} min={emb.min():.6f} max={emb.max():.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
