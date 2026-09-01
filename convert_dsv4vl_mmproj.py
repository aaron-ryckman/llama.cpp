#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash-Vision-Exp vision tensors to a llama.cpp mmproj GGUF.

Reads only vision/aligner/image_* tensors (they all live in shard 00001).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open

# Allow `python convert_dsv4vl_mmproj.py` from a llama.cpp tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFWriter  # noqa: E402


def add_f16(writer: GGUFWriter, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    writer.add_tensor(name, arr)


def add_f32(writer: GGUFWriter, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    writer.add_tensor(name, arr)


def split_swiglu(w1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # PyTorch Linear(dim, 2*inter): weight [2*inter, dim], chunk gate then up on dim 0
    assert w1.ndim == 2 and w1.shape[0] % 2 == 0, w1.shape
    half = w1.shape[0] // 2
    return w1[:half], w1[half:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF snapshot dir containing shard 00001 + config.json")
    ap.add_argument("--out", required=True, help="output mmproj GGUF path")
    ap.add_argument("--f32", action="store_true", help="store F32 tensors (CPU debug); default F16")
    args = ap.parse_args()
    add_w = add_f32 if args.f32 else add_f16

    src = Path(args.src)
    cfg = json.loads((src / "config.json").read_text())
    n_layer = int(cfg["vision_n_layers"])
    n_embd = int(cfg["vision_dim"])
    n_head = int(cfg["vision_n_heads"])
    n_ff = int(cfg["vision_inter_dim"])
    patch = int(cfg["vision_patch_size"])
    rope_theta = float(cfg["vision_rope_theta"])
    merge = int(cfg["vision_downsample_ratio"])
    proj_dim = int(cfg["hidden_size"])
    max_tok = int(cfg["vision_max_n_token"])
    min_pixels = int(cfg["vision_min_pixels"])

    shard = src / "model-00001-of-00048.safetensors"
    if not shard.exists():
        raise SystemExit(f"missing {shard}")

    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(("vision.", "aligner.", "image_")):
                t = f.get_tensor(k)
                tensors[k] = t.float().cpu().numpy()

    need = [
        "vision.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias",
        "vision.norm.weight",
        "aligner.w1.weight",
        "aligner.w1.bias",
        "aligner.w2.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
        "image_pad",
    ]
    missing = [k for k in need if k not in tensors]
    if missing:
        raise SystemExit(f"missing tensors: {missing}")

    writer = GGUFWriter(args.out, "clip")
    writer.add_name("DeepSeek-V4-Flash-Vision-Exp")
    writer.add_description("mmproj for DeepSeek-V4-Flash-Vision-Exp")
    writer.add_file_type(0 if args.f32 else 1)  # 0=F32, 1=F16
    writer.add_clip_has_vision_encoder(True)
    writer.add_clip_projector_type("dsv4vl")
    writer.add_vision_embedding_length(n_embd)
    writer.add_vision_head_count(n_head)
    writer.add_vision_feed_forward_length(n_ff)
    writer.add_vision_block_count(n_layer)
    writer.add_vision_patch_size(patch)
    writer.add_vision_image_size(756)
    writer.add_vision_projection_dim(proj_dim)
    writer.add_vision_spatial_merge_size(merge)
    writer.add_vision_attention_layernorm_eps(1e-6)
    writer.add_rope_freq_base(rope_theta)
    writer.add_vision_use_silu(True)
    writer.add_vision_image_mean([0.5, 0.5, 0.5])
    writer.add_vision_image_std([0.5, 0.5, 0.5])
    writer.add_vision_min_pixels(min_pixels)
    writer.add_uint32("clip.vision.image_max_tokens", max_tok)

    # Linear(3*p*p, n_embd) -> conv2d [out, in_c, kH, kW]
    w = tensors["vision.patch_embed.proj.weight"]
    assert w.shape == (n_embd, 3 * patch * patch), w.shape
    w = w.reshape(n_embd, 3, patch, patch)
    add_w(writer, "v.patch_embd.weight", w)
    add_w(writer, "v.patch_embd.bias", tensors["vision.patch_embed.proj.bias"].reshape(-1))
    add_w(writer, "v.post_ln.weight", tensors["vision.norm.weight"].reshape(-1))

    for i in range(n_layer):
        pfx = f"vision.blocks.{i}"
        add_w(writer, f"v.blk.{i}.ln1.weight", tensors[f"{pfx}.norm1.weight"].reshape(-1))
        add_w(writer, f"v.blk.{i}.ln2.weight", tensors[f"{pfx}.norm2.weight"].reshape(-1))
        add_w(writer, f"v.blk.{i}.attn_qkv.weight", tensors[f"{pfx}.attn.wqkv.weight"])
        add_w(writer, f"v.blk.{i}.attn_qkv.bias", tensors[f"{pfx}.attn.wqkv.bias"].reshape(-1))
        add_w(writer, f"v.blk.{i}.attn_out.weight", tensors[f"{pfx}.attn.wo.weight"])
        add_w(writer, f"v.blk.{i}.attn_out.bias", tensors[f"{pfx}.attn.wo.bias"].reshape(-1))
        gate, up = split_swiglu(tensors[f"{pfx}.mlp.w1.weight"])
        add_w(writer, f"v.blk.{i}.ffn_gate.weight", gate)
        add_w(writer, f"v.blk.{i}.ffn_up.weight", up)
        add_w(writer, f"v.blk.{i}.ffn_down.weight", tensors[f"{pfx}.mlp.w2.weight"])

    add_w(writer, "mm.1.weight", tensors["aligner.w1.weight"])
    add_w(writer, "mm.1.bias", tensors["aligner.w1.bias"].reshape(-1))
    add_w(writer, "mm.2.weight", tensors["aligner.w2.weight"])
    add_w(writer, "mm.2.bias", tensors["aligner.w2.bias"].reshape(-1))
    add_w(writer, "v.image_start", tensors["image_start"].reshape(-1))
    add_w(writer, "v.image_end", tensors["image_end"].reshape(-1))
    add_w(writer, "v.image_newline", tensors["image_newline"].reshape(-1))
    add_w(writer, "v.image_pad", tensors["image_pad"].reshape(-1))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}  n_layer={n_layer} n_embd={n_embd} proj={proj_dim} tensors={len(tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
