from __future__ import annotations

import json
import logging
from typing import Iterable

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import LazyTorchTensor, ModelBase, MmprojModel
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel

logger = logging.getLogger("hf-to-gguf")


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("unsloth/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # The MTP block is a separate draft head. It is skipped by default (as vLLM does), but
    # `--mtp` exports it on its own as a draft model for speculative decoding. The mixin
    # machinery arrives via _LinearAttentionVReorderBase -> Qwen3NextModel -> _QwenMtpMixin;
    # these flags previously overrode it off.
    supports_mtp_export = True
    no_mtp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # shards held only until the row stride is known, normally none
        self._ple_pending: dict[int, Tensor] = {}
        self._ple_shard_rows: dict[int, int] = {}
        self._ple_row_dim: int | None = None
        self._ple_rows_per_shard: int | None = None
        self._ple_map = None
        self._ple_path = None
        # Official FP8 checkpoint stores the 128 PLE shards as float8_e4m3fn
        # plus one scalar ngram_embedding.weight_scale (not per-shard
        # weight_scale_inv). base.py's quant_method=="fp8" path only consumes
        # *_scale_inv, so the scalar would hit map_tensor_name and abort.
        self._ple_fp8_scale: Tensor | None = None
        self._ple_fp8_scale_looked: bool = False

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        ratios = [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        # the loader sizes this array by n_layer_all, which counts the MTP block. That block
        # carries indexer tensors and is never recurrent, so it takes the full-attention ratio.
        ratios += [ratio] * max(0, self.block_count - n_layer)
        self.gguf_writer.add_attention_compress_ratios(ratios)

        # The MTP block carries no PLE n-gram layers. Emitting the key group anyway would make
        # the loader demand the PLE hparams -- and the per-layer input width -- for tensors that
        # are not in an MTP-only file.
        if self.mtp_only:
            return

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        #
        # Experiment (dreadnaught 8x MI50, 2026-08-26): skip the 51B n-gram
        # table. Materialising it is a 191 GiB f32 mmap; the GGUF writer then
        # copies it to convert to bf16 and climbs past 250 GiB RSS. The rest of
        # qwen4exp (GDN + QSA + MoE) is what we need to prove on gfx906 first.
        ple_layers: list[int] = []
        if not ple_layers:
            logger.info("PLE: skipping n-gram table (ple_layer_ids=%s)", hp.get("ple_layer_ids"))
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # The PLE hash runs over token ids, but a multimodal batch arrives as embeddings
        # with the placeholder consumed. Carry it so those positions hash what the
        # reference sees in input_ids instead of being undefined.
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        # image_token_id is top-level in config.json, not in self.hparams once that is
        # narrowed to text_config, and the text model has no global_config; read the file
        img = self.hparams.get("image_token_id")
        if img is not None:
            return int(img)
        try:
            with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
                img = json.load(f).get("image_token_id")
        except Exception:
            return None
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        return int(eos)

    @classmethod
    def filter_tensors(cls, item):
        name = item[0]
        if name.startswith("mtp.hyper_connection_mixer."):
            if cls.no_mtp:
                return None
            # Remap and return *without* super(): mixin mtp_only keep-list would
            # drop model.hyper_connection_mixer.*, but the draft file needs it
            # (C++ loads hc_head_norm/down/up as required even for MTP-only).
            return (name.replace("mtp.", "model.", 1), item[1])
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        # Drop the shared FP8 scale after we have used it in _write_ple_shard.
        if name.endswith("ngram_embedding.weight_scale"):
            return []

        if ".ngram_embedding.shard_" in name:
            return []  # skipped with PLE table; see set_gguf_parameters

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # -- the PLE table ----------------------------------------------------
    #
    # 128 shards concatenate into one enormous tensor. Holding them all and then
    # torch.cat-ing peaks near 300 GB of RSS, which most machines that can
    # otherwise convert this model do not have. Each shard is instead written
    # straight into a memory-mapped file at its final row offset and dropped, so
    # the peak is one shard and the rest is the page cache's problem. The trade
    # is a temporary file beside the output, removed when the write finishes.
    #
    # The file holds float32 because that is what base.py has already cast the
    # shards to by the time modify_tensors sees them, and what it calls .numpy()
    # on afterwards.

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()
        e_name, h_name = "mtp.fc_embedding.weight", "mtp.fc_hidden.weight"
        have_e, have_h = e_name in self.model_tensors, h_name in self.model_tensors
        if not have_e and not have_h:
            return
        if not have_e or not have_h:
            raise KeyError(f"unpaired MTP input projection: need both {e_name} and {h_name}")
        e = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
        h = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                                       self.hparams["num_hidden_layers"]),
               torch.cat([e, h], dim=1).contiguous())
        del self.model_tensors[e_name]
        del self.model_tensors[h_name]

    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]
        rows, row_dim = int(data_torch.shape[0]), int(data_torch.shape[-1])

        self._ple_row_dim = row_dim
        self._ple_shard_rows[idx] = rows

        if self._ple_map is None:
            if idx == n_parts - 1 and n_parts > 1:
                # the last shard may be short, so it cannot set the stride. This
                # only happens if the checkpoint yields shards out of order
                self._ple_pending[idx] = data_torch
                return []
            self._ple_rows_per_shard = rows
            self._ple_path = self.fname_out.parent / f".{self.fname_out.stem}.ple.tmp"
            self._ple_map = np.memmap(
                self._ple_path, dtype=np.float32, mode="w+",
                shape=(n_parts * rows, row_dim))

        for i, held in list(self._ple_pending.items()):
            self._ple_pending.pop(i)
            self._write_ple_shard(i, held)
        self._write_ple_shard(idx, data_torch)

        if len(self._ple_shard_rows) < n_parts:
            return []

        total = sum(self._ple_shard_rows.values())
        table = self._finish_ple_table(total)

        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", table)]

    def _write_ple_shard(self, idx: int, shard: Tensor) -> None:

        rows = int(shard.shape[0])
        if idx != self.hparams["split_ngram_parts"] - 1 and rows != self._ple_rows_per_shard:
            raise ValueError(
                f"PLE shard {idx} has {rows} rows, expected {self._ple_rows_per_shard}; "
                "shards other than the last must be uniform for direct placement"
            )

        start = idx * self._ple_rows_per_shard
        # the shard is still lazy here; force it, since the point of this path
        # is that exactly one shard is resident at a time
        from .base import LazyTorchTensor

        eager = LazyTorchTensor.to_eager(shard)
        fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
        if hasattr(torch, "float8_e4m3fnuz"):
            fp8_types = fp8_types + (torch.float8_e4m3fnuz,)
        if eager.dtype in fp8_types:
            eager = eager.to(torch.float32)
            scale = self._ple_fp8_scale_value()
            if scale is not None:
                eager = eager * scale
        else:
            eager = eager.to(torch.float32)
        eager = eager.contiguous()
        self._ple_map[start:start + rows] = eager.numpy()
        del eager

    def _ple_fp8_scale_value(self) -> Tensor | None:
        """Scalar ngram_embedding.weight_scale from the official FP8 checkpoint."""
        if self._ple_fp8_scale_looked:
            return self._ple_fp8_scale
        self._ple_fp8_scale_looked = True
        for name, gen in self.model_tensors.items():
            if name.endswith("ngram_embedding.weight_scale"):
                t = gen().to(torch.float32).reshape(())
                self._ple_fp8_scale = t
                logger.info("PLE FP8: applying scalar ngram_embedding.weight_scale=%s", float(t))
                return t
        return None

    def _finish_ple_table(self, total_rows: int):

        self._ple_map.flush()

        # trim the tail if the last shard came up short of a full stride
        want = total_rows * self._ple_row_dim * 4
        if self._ple_path.stat().st_size != want:
            with open(self._ple_path, "r+b") as f:
                f.truncate(want)

        # Keep a writable memmap VIEW. np.asarray(memmap) copies ~191 GiB into
        # RAM and OOMs; torch.bfloat16().numpy() is also unsupported. The GGUF
        # writer downcasts this f32 view to bf16 itself.
        self._ple_map = np.memmap(self._ple_path, dtype=np.float32, mode="r+",
                                  shape=(total_rows, self._ple_row_dim))
        logger.info("PLE: wrapping %d x %d f32 mmap by view (no copy)",
                    total_rows, self._ple_row_dim)
        return torch.from_numpy(self._ple_map)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._ple_pending:
            raise ValueError(
                f"unprocessed PLE embedding shards: {sorted(self._ple_pending)}"
            )

    def write(self):
        try:
            super().write()
        finally:
            if self._ple_path is not None and self._ple_path.exists():
                self._ple_path.unlink()


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("unsloth/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
