#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    if (hparams.n_layer_nextn > 0) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, head_hc_flags);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, head_hc_flags);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, head_hc_flags);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        // for wo_a, the shape in the file is (n_head * n_embd_head / o_groups, o_lora_rank*o_groups)
        // so we reshape here, to avoid reshaping the tensor in the graph
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}


// mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * dsv4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_deepseek4::graph::build_head_fold(const llama_model & model, ggml_tensor * x) const {
    if (model.hc_head_fn == nullptr) {
        // V4.1: fold with the mix the last layer computed from its pre-FFN input.
        // Recomputing one here from the post-FFN residual would not be the same value.
        GGML_ASSERT(last_ffn_pre != nullptr);
        return build_hc_pre(x, last_ffn_pre, -1);
    }

    return build_hc_head(x, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il,
        ggml_tensor ** pre_out,
        ggml_tensor  * mix_in,
        bool           collapse) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    if (pre_out) {
        *pre_out = pre;
    }

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    if (!collapse) {
        // the caller collapses on its own (V4.1 layer 0: one-hot on copy 0); do not build a collapse that would
        // never join the graph, see the declaration
        return nullptr;
    }

    // V4 collapses with the mix this sublayer just computed.
    // V4.1 hands its coefficients to the NEXT sublayer instead - attention uses the previous block's ffn mix, the FFN uses this block's attention mix - so the caller passes the carried one.
    ggml_tensor * result = build_hc_pre(x, mix_in ? mix_in : pre, il);
    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_pre(x, pre, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il,
        ggml_tensor ** pre_rope) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == ratio*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, ratio, n_blocks);
    cb(kv, name, il);

    // at ratio 1 the softmax below runs over a single element and is the identity, which is why a ratio 1 source layer needs no gate at all and the file carries no attn_comp_wgate for it
    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, ratio, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    // V4.1 builds its index keys from this form, before RoPE (model.py Compressor: "Pre-RoPE is deliberate: the indexer needs the unrotated form")
    if (pre_rope) {
        *pre_rope = comp;
    }

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    indexer_q = ggml_rope_set_offset(indexer_q, n_embd_indexer_head_nope);
    cb(indexer_q, "lid_q_rope", il);

    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    cb(kq_mask, "csa_lid_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il,
        int il_kv,
        bool idx_tier,
        ggml_tensor * top_k) const {
    // il_kv is the layer that owns the compressed rows.
    // For V4 that is this layer; V4.1 has the layers after a source read the source's cache, which is the only place they differ.
    const auto & inp_hca = idx_tier ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);

        // V4.1 selection: the indexer chain just before this may be pinned to the key owner's device, and the scheduler
        // hands unpinned ops the device of the node before them; anchor the layer's own work back on its device
        if (top_k) {
            cb(q->src[0],  "dsv41_pin", il);
            cb(kv->src[0], "dsv41_pin", il);
        }
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = (idx_tier ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca())->get_k(ctx0, il_kv);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    // V4.1 sparse compute: gather the picked rows and attend over those alone (see build_dsv41_sparse_attention)
    if (top_k && dsv41_sparse) {
        GGML_ASSERT(dsv41_sel_k && dsv41_sel_mask && "sparse selection without the gathered picks");

        ggml_tensor * out = build_dsv41_sparse_attention(q, raw_k, inp_attn->get_kq_mask(),
                dsv41_sel_k, dsv41_sel_mask, sinks, kq_scale, il);
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_hca_sparse", il);

        return out;
    }

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    // V4.1 sparse selection keeps only the rows the indexer picked; the 128-token window in raw_mask stays whole, as the reference concatenates window and compressed picks (model.py:774-778)
    ggml_tensor * hca_mask = top_k
        ? build_top_k_mask(inp_hca.kq_mask, top_k, "hca_top_k_mask", il)
        : inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

// DeepSeek-V4.1 sparse selection (CSA2), opt-in with LLAMA_DSV41_TOPK=1.
//
// The reference (DeepSeek-V4.1-Flash inference/model.py) spreads the selection over three kinds of layer:
//  - a KV source (2, 8, 14 at ratio 2; 20 at ratio 1) pools its compressed latent and, before rotating it, derives one
//    index key per compressed row: k = rms_norm(wk(latent)) with RoPE on the last n_rot dims at the row's first token
//    (Indexer.forward, model.py:537-548). Every layer up to the next source reads the same rows and the same keys.
//  - an index source (2, 8, 14, 20, 24, 28, 32, 36) scores its own queries against those keys,
//    score[t] = sum_h relu(q_h . k_t) * w_h with w = weights_proj(x) * index_head_dim**-0.5 * n_heads**-0.5
//    (model.py:550-557), masks the rows the query has not passed yet (model.py:561-566) and keeps the best
//    index_topk = 512 (model.py:578-580). V4 scores with the same formula, so the fused lightning indexer applies.
//  - every other compressed layer reuses the rows its nearest index source picked (model.py:722-737).
//  - the candidate source (20) also ranks blocks of candidate_block_size rows by their best row, pins the block
//    holding the query's newest row, keeps candidate_topk_blocks of them (select_candidate_blocks, model.py:583-610),
//    and the index sources after it (24..36) only score inside those blocks (model.py:569-576). Below
//    candidate_topk_blocks * candidate_block_size = 16384 rows every block is a candidate and this changes nothing.
// The picks are always added to the 128-token sliding window (model.py:774-778), never instead of it.
//
// Here the index keys live in a K-only cache that mirrors the tier's compressed cache row for row (kv_lid for
// ratio 2, kv_lid_plain for ratio 1), so the tier's plan (write ids, positions, visibility mask) serves both.
// What V4.1 does not do: no Hadamard on the indexer in the model itself; the rotation applied below is the
// cache's own quantization aid, orthonormal, and applied to both sides, so q.k is unchanged.
// Not reproduced: the reference's fp4/fp8 activation quantization of q, k and the compressed rows.

static bool dsv41_candidates_enabled(const llama_hparams & hparams) {
    return hparams.dsv41_candidate_topk > 0 && hparams.dsv41_candidate_block > 0 &&
           hparams.dsv41_is_index_source(hparams.dsv41_candidate_src_layer);
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_index_keys(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & tier,
        const llama_kv_cache_dsv4_comp_context * lid_ctx,
        ggml_tensor * latent_pre,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_index      = hparams.indexer_head_size;
    const int64_t n_embd_index_rope = hparams.n_rot();
    const int64_t n_embd_index_nope = n_embd_index - n_embd_index_rope;

    GGML_ASSERT(lid_ctx);
    GGML_ASSERT(latent_pre);
    GGML_ASSERT(layer.indexer_comp_wkv && layer.indexer_comp_norm && "V4.1 KV source without indexer key tensors");
    GGML_ASSERT(tier.state_write_idxs && tier.state_write_pos);
    GGML_ASSERT(n_embd_index >= n_embd_index_rope);

    // latent_pre is [n_embd_head, 1, n_blocks]: pooled and normed, before RoPE (Compressor.forward, model.py:458-486)
    ggml_tensor * k = build_lora_mm(layer.indexer_comp_wkv, latent_pre);
    k = build_norm(k, layer.indexer_comp_norm, nullptr, LLM_NORM_RMS, il);
    cb(k, "dsv41_idx_k_new", il);

    // a row stands for the first token of its group and takes that position, with the compressed tier's rope (model.py:538-546)
    k = ggml_rope_ext(ctx0, k, tier.state_write_pos, nullptr, n_embd_index_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    k = ggml_rope_set_offset(k, n_embd_index_nope);
    cb(k, "dsv41_idx_k_new_rope", il);

    // both index-key caches share hparams_lid, so one rotation matrix fits both
    if (inp_dsv4->get_lid().k_rot) {
        k = llama_mul_mat_hadamard(ctx0, k, inp_dsv4->get_lid().k_rot);
        cb(k, "dsv41_idx_k_new_rot", il);
    }

    // same row ids as the compressed KV of this tier, incomplete-group scratch row included
    return lid_ctx->cpy_k(ctx0, k, tier.state_write_idxs, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_select_mask(
        ggml_tensor * idxs,
        int64_t n_rows,
        ggml_type type,
        const char * name,
        int il) const {
    const int64_t n_sel    = idxs->ne[0];
    const int64_t nt       = idxs->ne[1];
    const int64_t n_stream = idxs->ne[3];

    GGML_ASSERT(idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(idxs->ne[2] == 1);
    GGML_ASSERT(n_sel > 0 && n_sel <= n_rows);
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16);

    // same layout trick as build_top_k_mask(): rows of width 1 so set_rows() can scatter per (token, stream)
    ggml_tensor * all = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, type, 1, n_rows, nt, n_stream), -INFINITY);
    ggml_tensor * sel = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, type, 1, n_sel,  nt, n_stream), 0.0f);

    ggml_tensor * idxs_3d = ggml_view_4d(ctx0, idxs, n_sel, nt, n_stream, 1,
            idxs->nb[1], idxs->nb[2], n_stream*idxs->nb[3], 0);

    ggml_tensor * mask = ggml_set_rows(ctx0, all, sel, idxs_3d);
    mask = ggml_view_4d(ctx0, mask, n_rows, nt, 1, n_stream, mask->nb[2], mask->nb[3], mask->nb[3], 0);
    cb(mask, name, il);

    return mask;
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_candidate_mask(
        ggml_tensor * score,
        ggml_tensor * kq_mask,
        int il) const {
    const int64_t n_kv     = score->ne[0];
    const int64_t nt       = score->ne[1];
    const int64_t n_stream = score->ne[3];
    const int64_t bsz      = hparams.dsv41_candidate_block;

    GGML_ASSERT(score->type == GGML_TYPE_F32 && kq_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(score) && ggml_is_contiguous(kq_mask));
    GGML_ASSERT(ggml_are_same_shape(score, kq_mask));
    GGML_ASSERT(score->ne[2] == 1);
    // the plan pads n_kv to a multiple of 256, so the reference's -inf tail padding to a whole block is already there
    GGML_ASSERT(bsz > 0 && n_kv % bsz == 0);

    const int64_t n_blocks = n_kv/bsz;
    GGML_ASSERT(n_blocks > 1);

    // a block scores as its best row (model.py:598-600). Unreachable rows are -inf in score; a block of only those pools
    // to -FLT_MAX, which ranks the same, and its rows stay -inf in the score whatever the block decision is.
    ggml_tensor * bscore = ggml_pool_1d(ctx0, score, GGML_OP_POOL_MAX, (int) bsz, (int) bsz, 0);
    cb(bscore, "dsv41_cand_block_score", il);

    // pin the block holding the query's newest reachable row (model.py:602-605): a block with any reachable row pools the
    // 0/-inf mask to 0 and exp() makes that 1, an unreachable block gives 0; the newest is the reachable block whose
    // successor is not.
    ggml_tensor * bvis = ggml_exp(ctx0, ggml_pool_1d(ctx0, kq_mask, GGML_OP_POOL_MAX, (int) bsz, (int) bsz, 0));
    ggml_tensor * bvis_next = ggml_concat(ctx0,
            ggml_cont(ctx0, ggml_view_4d(ctx0, bvis, n_blocks - 1, nt, 1, n_stream,
                    bvis->nb[1], bvis->nb[2], bvis->nb[3], ggml_element_size(bvis))),
            ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, nt, 1, n_stream), 0.0f),
            0);
    ggml_tensor * is_last = ggml_mul(ctx0, bvis, ggml_scale_bias(ctx0, bvis_next, -1.0f, 1.0f));
    cb(is_last, "dsv41_cand_last_block", il);

    // a finite huge pin rather than +inf, so a 0 * inf can never produce a NaN
    bscore = ggml_add(ctx0, bscore, ggml_scale(ctx0, is_last, 1e30f));
    cb(bscore, "dsv41_cand_block_score_pinned", il);

    const int64_t n_top_blocks = std::min<int64_t>(n_blocks, hparams.dsv41_candidate_topk);
    ggml_tensor * top_blocks = ggml_cont(ctx0, ggml_top_k(ctx0, bscore, (int) n_top_blocks));
    cb(top_blocks, "dsv41_cand_top_blocks", il);

    // 0 on the kept blocks, -inf elsewhere, spread over the bsz rows of each block (model.py:607-610)
    ggml_tensor * bmask = build_dsv41_select_mask(top_blocks, n_blocks, GGML_TYPE_F32, "dsv41_cand_block_mask", il);
    ggml_tensor * cand = ggml_reshape_4d(ctx0, bmask, 1, n_blocks, nt, n_stream);
    cand = ggml_repeat_4d(ctx0, cand, bsz, n_blocks, nt, n_stream);
    cand = ggml_reshape_4d(ctx0, cand, n_kv, nt, 1, n_stream);
    cb(cand, "dsv41_cand_mask", il);

    return cand;
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & tier,
        const llama_kv_cache_dsv4_comp_context * lid_ctx,
        int il_keys,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_index      = hparams.indexer_head_size;
    const int64_t n_embd_index_rope = hparams.n_rot();
    const int64_t n_embd_index_nope = n_embd_index - n_embd_index_rope;
    const int64_t n_index_head      = hparams.indexer_n_head;
    const int64_t nt                = cur->ne[1];

    GGML_ASSERT(tier.kq_mask);
    GGML_ASSERT(lid_ctx);
    GGML_ASSERT(layer.indexer_attn_q_b && layer.indexer_proj && "V4.1 index source without indexer query tensors");
    GGML_ASSERT(n_embd_index >= n_embd_index_rope);

    // q = wq_b(qr) per index head, qr being the normalized low-rank query; RoPE on the tail at the query position with
    // the compressed tier's rope (model.py:550-551)
    ggml_tensor * q = build_lora_mm(layer.indexer_attn_q_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_index, n_index_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_index_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    q = ggml_rope_set_offset(q, n_embd_index_nope);
    cb(q, "dsv41_idx_q", il);

    if (inp_dsv4->get_lid().k_rot) {
        q = llama_mul_mat_hadamard(ctx0, q, inp_dsv4->get_lid().k_rot);
        cb(q, "dsv41_idx_q_rot", il);
    }

    // per-head weights with the reference's softmax_scale * n_heads**-0.5 folded in (model.py:555)
    ggml_tensor * w = build_lora_mm(layer.indexer_proj, cur);
    w = ggml_scale(ctx0, w, 1.0f/sqrtf(float(n_embd_index*n_index_head)));
    cb(w, "dsv41_idx_w", il);

    // the keys the KV source published, as many rows as the tier exposes this ubatch
    ggml_tensor * k = lid_ctx->get_k(ctx0, il_keys);
    const int64_t n_kv = tier.kq_mask->ne[0];
    GGML_ASSERT(n_kv > 0);
    GGML_ASSERT(n_kv <= k->ne[2]);
    k = ggml_view_4d(ctx0, k, k->ne[0], k->ne[1], n_kv, k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
    cb(k, "dsv41_idx_k", il);

    const int64_t n_stream = k->ne[3];
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
            q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
    w = ggml_view_4d(ctx0, w, w->ne[0], w->ne[1]/n_stream, w->ne[2], n_stream,
            w->nb[1], w->nb[2]/n_stream, w->nb[3]/n_stream, 0);

    // the tier mask already holds -inf on every row the query has not passed (n_visible = (pos+1)/ratio), which is the
    // reference's compress_lens mask (model.py:561-566). It is F16 under flash attention and F32 otherwise; the fused
    // indexer wants F16, the unfused add wants the score's F32, and 0/-inf survive either cast exactly.
    ggml_tensor * mask = tier.kq_mask;
    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        ggml_tensor * mask_f16 = mask->type == GGML_TYPE_F16 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F16);
        score = ggml_lightning_indexer(ctx0, q, k, w, mask_f16);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        // score where the keys live: an index source after the key owner (24..36 read layer 20) would otherwise pull the
        // whole key view across devices every step; q and w are small, the score is O(n_kv) floats
        cb(score, "dsv41_pin", il_keys);
    } else {
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "dsv41_pin", il_keys);
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));

        score = ggml_relu(ctx0, kq);
        score = ggml_mul(ctx0, score, w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        ggml_tensor * mask_f32 = mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32);
        score = ggml_add(ctx0, score, mask_f32);
    }
    cb(score, "dsv41_idx_score", il);

    // two-level selection: the candidate source publishes the block mask and does not apply it to itself; the index
    // sources after it score only inside the candidate blocks (model.py:569-576)
    if (dsv41_candidates_enabled(hparams)) {
        if ((uint32_t) il == hparams.dsv41_candidate_src_layer) {
            ggml_tensor * mask_f32 = mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32);
            dsv41_cand_mask = build_dsv41_candidate_mask(score, mask_f32, il);
        } else if ((uint32_t) il > hparams.dsv41_candidate_src_layer) {
            GGML_ASSERT(dsv41_cand_mask && "LLAMA_DSV41_TOPK: index source after the candidate source found no candidate mask");
            GGML_ASSERT(ggml_are_same_shape(dsv41_cand_mask, score) && "LLAMA_DSV41_TOPK: candidate mask from a different compressed tier");
            score = ggml_add(ctx0, score, dsv41_cand_mask);
            cb(score, "dsv41_idx_score_cand", il);
        }
    }

    // top index_topk rows per query (model.py:578-580); when the tier exposes fewer rows than that, every row is kept and
    // the selection is the dense path exactly
    const int64_t n_top_k = std::min<int64_t>(n_kv, hparams.indexer_top_k);
    GGML_ASSERT(n_top_k > 0);
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, (int) n_top_k));
    cb(top_k, "dsv41_idx_top_k", il);

    // the gather path needs to know which picks are real rows: the same 0/-inf the mask path would add, read at the picks
    if (dsv41_sparse) {
        ggml_tensor * mask_f32 = mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32);
        dsv41_sel_mask = build_dsv41_sel_mask(mask_f32, top_k, il);
    }

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_sel_mask(
        ggml_tensor * mask_f32,
        ggml_tensor * top_k,
        int il) const {
    const int64_t n_kv     = mask_f32->ne[0];
    const int64_t nt       = mask_f32->ne[1];
    const int64_t n_stream = mask_f32->ne[3];
    const int64_t n_sel    = top_k->ne[0];

    GGML_ASSERT(mask_f32->type == GGML_TYPE_F32 && ggml_is_contiguous(mask_f32));
    GGML_ASSERT(mask_f32->ne[2] == 1);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && ggml_is_contiguous(top_k));
    GGML_ASSERT(top_k->ne[1] == nt && top_k->ne[2] == 1 && top_k->ne[3] == n_stream);

    // get_rows gathers rows, and the CUDA kernel wants rows of an even width, so each mask entry is doubled into a 2-wide
    // row first; per stream the gather is batched over the tokens (a->ne[2] == b->ne[1]) and column 0 is kept
    ggml_tensor * m2 = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, mask_f32, 1, n_kv, nt, n_stream), 2, n_kv, nt, n_stream);

    ggml_tensor * res = nullptr;
    for (int64_t s = 0; s < n_stream; ++s) {
        ggml_tensor * a = ggml_view_3d(ctx0, m2, 2, n_kv, nt, m2->nb[1], m2->nb[2], s*m2->nb[3]);
        ggml_tensor * b = ggml_view_2d(ctx0, top_k, n_sel, nt, top_k->nb[1], s*top_k->nb[3]);

        ggml_tensor * g = ggml_get_rows(ctx0, a, b); // F32 [2, n_sel, nt]
        g = ggml_cont(ctx0, ggml_view_3d(ctx0, g, 1, n_sel, nt, g->nb[1], g->nb[2], 0));

        res = res ? ggml_concat(ctx0, res, g, 3) : g;
    }

    res = ggml_reshape_4d(ctx0, res, n_sel, nt, 1, n_stream);
    cb(res, "dsv41_sel_mask", il);

    return res;
}

// Sparse compute for V4.1's selection. The mask path hands flash attention every compressed row and lets -inf do the
// selecting, so decode time grows with the context. Here the picked rows are gathered into a compact K (which is also V)
// of n_raw window rows plus n_top_k picks per query, and attention runs over those alone.
//
// Layout: each query gets its own K, so the tokens are folded into the batch dimension that build_attn_mha() already
// splits streams over: K is [n_embd_head, 1, n_kv_sel, n_tokens/n_stream * n_stream] with batch b = t + T*s, which is the
// order the queries have in q (a stream's tokens are contiguous). The window rows are shared by a stream's tokens and
// repeated; the picks are gathered per token with the stream's row ids, so no global ids are needed and multi-stream
// caches work unchanged. Rows come out of get_rows as F32 and build_attn_mha() casts them to F16 under flash attention.
//
// Equivalence: same picks, same validity (the tier mask read at the picks), same window and mask, same scale and sinks;
// masked and padding rows contribute nothing. Padding to FATTN_KQ_STRIDE keeps the D=512 flash-attention kernels eligible.
// Not bit-exact against the mask path on a quantized cache: the picks are dequantized once here, the mask path lets the
// kernel dequantize; with an F16 cache the rows are identical and only the kernel's tiling differs.
ggml_tensor * llama_model_deepseek4::graph::build_dsv41_gather_picks(
        ggml_tensor * comp_k,
        ggml_tensor * top_k,
        int il_kv) const {
    const int64_t n_embd_head = comp_k->ne[0];
    const int64_t n_stream    = comp_k->ne[3];
    const int64_t n_sel       = top_k->ne[0];
    const int64_t nt          = top_k->ne[1];       // tokens per stream

    GGML_ASSERT(comp_k->ne[1] == 1);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && ggml_is_contiguous(top_k) && top_k->ne[2] == 1 && top_k->ne[3] == n_stream);

    // The scheduler places an op by its weights or by its neighbours, never by a KV cache it reads, and a view is not a
    // node it looks at: left alone, this gather would land on the reader's device and the scheduler would copy the view it
    // reads, which is the stream's entire cache, across every step (measured: flat 2.7 t/s whatever the context). Pinning
    // each op here to the cache owner's device keeps the cache put; only the gathered rows travel.
    ggml_tensor * res = nullptr;
    for (int64_t s = 0; s < n_stream; ++s) {
        ggml_tensor * kc_s  = ggml_view_2d(ctx0, comp_k, n_embd_head, comp_k->ne[2], comp_k->nb[2], s*comp_k->nb[3]);
        ggml_tensor * idx_s = ggml_view_1d(ctx0, top_k, n_sel*nt, s*top_k->nb[3]);

        ggml_tensor * g_s = ggml_get_rows(ctx0, kc_s, idx_s); // F32 [n_embd_head, n_sel*nt]
        cb(g_s, "dsv41_pin", il_kv);
        g_s = ggml_reshape_3d(ctx0, g_s, n_embd_head, n_sel, nt);

        if (res == nullptr) {
            res = g_s;
        } else {
            res = ggml_concat(ctx0, res, g_s, 3);
            cb(res, "dsv41_pin", il_kv);
        }
    }

    // F16 before it travels: half the bytes, and flash attention wants F16 rows anyway
    if (cparams.flash_attn) {
        res = ggml_cast(ctx0, res, GGML_TYPE_F16);
        cb(res, "dsv41_pin", il_kv);
    }

    res = ggml_reshape_4d(ctx0, res, n_embd_head, n_sel, nt, n_stream);
    ggml_format_name(res, "dsv41_sel_k-%d", il_kv);

    return res;
}

ggml_tensor * llama_model_deepseek4::graph::build_dsv41_sparse_attention(
        ggml_tensor * q,
        ggml_tensor * raw_k,
        ggml_tensor * raw_mask,
        ggml_tensor * sel_k,
        ggml_tensor * sel_mask,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const int64_t n_embd_head = raw_k->ne[0];
    const int64_t n_raw       = raw_k->ne[2];
    const int64_t n_stream    = raw_k->ne[3];
    const int64_t n_sel       = sel_k->ne[1];
    const int64_t nt          = sel_k->ne[2];       // tokens per stream
    const int64_t n_batch     = nt*n_stream;

    GGML_ASSERT(raw_k->ne[1] == 1);
    GGML_ASSERT(sel_k->ne[0] == n_embd_head && sel_k->ne[3] == n_stream && ggml_is_contiguous(sel_k));
    GGML_ASSERT(q->ne[0] == n_embd_head && q->ne[2] == n_batch);
    GGML_ASSERT(raw_mask->ne[0] == n_raw && raw_mask->ne[1] == nt && raw_mask->ne[2] == 1 && raw_mask->ne[3] == n_stream);
    GGML_ASSERT(sel_mask->type == GGML_TYPE_F32 && ggml_is_contiguous(sel_mask));
    GGML_ASSERT(sel_mask->ne[0] == n_sel && sel_mask->ne[1] == nt && sel_mask->ne[2] == 1 && sel_mask->ne[3] == n_stream);

    // every window row of a stream, in order: identity ids, built once per graph (argsort of an arange stays I32 end to end)
    if (dsv41_raw_ids == nullptr || dsv41_raw_ids->ne[0] != n_raw) {
        dsv41_raw_ids = ggml_argsort(ctx0, ggml_arange(ctx0, 0.0f, (float) n_raw, 1.0f), GGML_SORT_ORDER_ASC);
        cb(dsv41_raw_ids, "dsv41_raw_ids", -1);
    }

    ggml_tensor * k_all = nullptr;
    ggml_tensor * m_all = nullptr;

    for (int64_t s = 0; s < n_stream; ++s) {
        // window rows of this layer's own cache (same device, no pin needed), dequantized, then one copy per token of the
        // stream: [n_embd_head, n_raw, nt]
        ggml_tensor * kr_s = ggml_view_2d(ctx0, raw_k, n_embd_head, n_raw, raw_k->nb[2], s*raw_k->nb[3]);
        ggml_tensor * r_s  = ggml_get_rows(ctx0, kr_s, dsv41_raw_ids);
        if (r_s->type != sel_k->type) {
            r_s = ggml_cast(ctx0, r_s, sel_k->type);
        }
        r_s = ggml_repeat_4d(ctx0, ggml_reshape_2d(ctx0, r_s, n_embd_head*n_raw, 1), n_embd_head*n_raw, nt, 1, 1);
        r_s = ggml_reshape_3d(ctx0, r_s, n_embd_head, n_raw, nt);

        // the picks, gathered once at the index source: [n_embd_head, n_sel, nt]
        ggml_tensor * g_s = ggml_view_3d(ctx0, sel_k, n_embd_head, n_sel, nt, sel_k->nb[1], sel_k->nb[2], s*sel_k->nb[3]);

        ggml_tensor * k_s = ggml_concat(ctx0, r_s, g_s, 1);

        // the matching mask columns: the window's per-token mask, then the picks' validity
        ggml_tensor * rm_s = ggml_cast(ctx0,
                ggml_view_2d(ctx0, raw_mask, n_raw, nt, raw_mask->nb[1], s*raw_mask->nb[3]), GGML_TYPE_F32);
        ggml_tensor * sm_s = ggml_cont(ctx0,
                ggml_view_2d(ctx0, sel_mask, n_sel, nt, sel_mask->nb[1], s*sel_mask->nb[3]));
        ggml_tensor * m_s = ggml_concat(ctx0, rm_s, sm_s, 0);

        k_all = k_all ? ggml_concat(ctx0, k_all, k_s, 3) : k_s;
        m_all = m_all ? ggml_concat(ctx0, m_all, m_s, 3) : m_s;
    }

    // [n_embd_head, n_kv_sel, nt, n_stream] -> one K per query, batch b = t + nt*s
    int64_t n_kv_sel = n_raw + n_sel;
    k_all = ggml_reshape_4d(ctx0, k_all, n_embd_head, 1, n_kv_sel, n_batch);
    m_all = ggml_reshape_4d(ctx0, m_all, n_kv_sel, 1, 1, n_batch);

    // the CUDA flash-attention kernels for D=512 want the row count a multiple of FATTN_KQ_STRIDE (256); pad with zero rows
    // that the mask removes
    const int64_t n_kv_pad = GGML_PAD(n_kv_sel, 256);
    if (n_kv_pad != n_kv_sel) {
        const int64_t n_pad = n_kv_pad - n_kv_sel;
        k_all = ggml_concat(ctx0, k_all,
                ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, k_all->type, n_embd_head, 1, n_pad, n_batch), 0.0f), 2);
        m_all = ggml_concat(ctx0, m_all,
                ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_pad, 1, 1, n_batch), -INFINITY), 0);
        n_kv_sel = n_kv_pad;
    }

    // K is already F16 under flash attention (the picks were cast where they were gathered, the window rows above); the
    // mask follows, contiguous as ggml_flash_attn_ext requires
    if (cparams.flash_attn) {
        GGML_ASSERT(k_all->type == GGML_TYPE_F16);
        m_all = ggml_cast(ctx0, m_all, GGML_TYPE_F16);
    }
    cb(k_all, "dsv41_sparse_k", il);
    cb(m_all, "dsv41_sparse_mask", il);

    // build_attn_mha() splits q over k_all->ne[3] = n_batch, so each query meets only its own rows; K doubles as V
    return build_attn_mha(q, k_all, k_all, nullptr, m_all, sinks, nullptr, kq_scale, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il,
        ggml_tensor * cur_comp) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il, cur_comp);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il,
        ggml_tensor * cur_comp) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    // the compressor's input; only the decoder skip hands in something other than cur (the whole ubatch, where cur is its tail)
    ggml_tensor * cur_comp_in = cur_comp ? cur_comp : cur;

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    // V4 renormalizes q per head after wq_b.
    // V4.1 does not: its only query norm is the one on wq_a's output, and this second one rescales every head's attention scores.
    // The V4.1 DSpark sidecar's stages (arch dflash, hparams.dsv41_dspark) are V4.1 attention too (reference DSparkAttention.forward: wq_b(qr) straight into RoPE).
    // LLAMA_DSV41_QNORM=1 puts it back for A/B.
    static const bool q_norm_off = [] {
        const char * e = getenv("LLAMA_DSV41_QNORM");
        return !(e != nullptr && atoi(e) != 0);
    }();
    const bool v41_attn = hparams.dsv41_n_kv_source > 0 || hparams.dsv41_dspark;
    if (!(v41_attn && q_norm_off)) {
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    }
    cb(q, "q_norm", il);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    // Which compressed tier this layer belongs to, and who owns its KV.
    // V4 compresses on every ratio layer, so each is its own source and both tiers keep their V4 shapes.
    // V4.1 pools non-overlapping groups on both tiers and writes only on a source layer, which the layers after it read.
    const bool v41        = hparams.dsv41_n_kv_source > 0;
    const bool tier_idx   = ratio != 0 && (uint32_t) ratio == hparams.dsv4_ratio_idx;
    const bool tier_plain = ratio != 0 && (uint32_t) ratio == hparams.dsv4_ratio_plain;
    const int  kv_src     = (v41 && ratio != 0) ? hparams.dsv41_kv_source_for(il) : il;
    const bool owns_kv    = ratio != 0 && kv_src == il;

    GGML_ASSERT(!(v41 && ratio != 0) || kv_src >= 0);

    // LLAMA_DSV41_NO_COMPRESS=1 drops the compressed tier and leaves every layer on its 128-token sliding window, to tell a broken compressed path from a broken stack.
    // RoPE is left alone so only one thing changes.
    static const bool compress_off = [] {
        const char * e = getenv("LLAMA_DSV41_NO_COMPRESS");
        const bool off = e != nullptr && atoi(e) != 0;
        if (off) {
            LLAMA_LOG_WARN("deepseek41: LLAMA_DSV41_NO_COMPRESS=1, compressed tier disabled\n");
        }
        return off;
    }();

    // LLAMA_DSV41_TOPK=1: V4.1's sparse selection over the compressed stream (see build_dsv41_index_keys). Off by default so
    // the dense path stays as it is; V4 never takes it.
    const bool topk_on = v41 && ratio != 0 && inp_dsv4 != nullptr && !compress_off && llama_dsv41_topk_enabled();
    const llama_kv_cache_dsv4_comp_context * lid_ctx = nullptr;

    // Sparse compute (gather) up to this many tokens per ubatch; larger ubatches (prefill) keep the mask path, because the
    // gather materializes n_top_k rows per token (512 x 512 x 4 B = 1 MiB per token per layer, plus the F16 copy).
    // LLAMA_DSV41_SPARSE_MAX_TOKENS overrides; 0 keeps the mask path everywhere.
    static const int64_t sparse_max_tokens = [] {
        const char * e = getenv("LLAMA_DSV41_SPARSE_MAX_TOKENS");
        return e != nullptr ? (int64_t) atoll(e) : (int64_t) 32;
    }();
    dsv41_sparse = topk_on && nt <= sparse_max_tokens;

    if (topk_on) {
        // each tier's index keys sit in a cache that mirrors that tier's compressed cache
        lid_ctx = tier_idx ? inp_dsv4->mctx->get_lid() : inp_dsv4->mctx->get_lid_plain();
        GGML_ASSERT(lid_ctx && "LLAMA_DSV41_TOPK: no index-key cache for this compressed tier");
    }

    // the pooled compressor serves V4's plain tier, and both of V4.1's
    const bool pooled   = (v41 ? owns_kv : tier_plain) && !(v41 && compress_off);
    const auto & tier   = inp_dsv4 ? (v41 && tier_idx ? inp_dsv4->get_csa() : inp_dsv4->get_hca())
                                   : inp_dsv4->get_hca();
    const int64_t tier_ratio = ratio;

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (pooled && tier.state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur_comp_in);
        cb(hca_state_kv, "comp_state_kv", il);

        // A ratio 1 tier pools one token, so its softmax is the identity and any score gives weight 1.
        // V4.1's ratio 1 source layer ships no gate at all, hence the zeros.
        hca_state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur_comp_in)
            : ggml_scale(ctx0, hca_state_kv, 0.0f);
        cb(hca_state_score, "comp_state_score", il);

        // V4 adds a learned position embedding to the gate. V4.1 ships none.
        if (layer.attn_comp_ape) {
            ggml_tensor * ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, tier.state_pos);
            hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
            cb(hca_state_score, "comp_state_score_ape", il);
        }
    }

    if (!v41 && tier_idx && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                hparams.dsv4_ratio_idx,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                hparams.dsv4_ratio_idx,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (pooled && tier.state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, tier, hca_state, il);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        // the index keys read the latent before it is rotated, so both forms come out of one pooling pass
        ggml_tensor * latent_pre = nullptr;
        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                tier.state_read_idxs,
                tier.state_write_pos,
                layer.attn_comp_norm,
                tier_ratio,
                n_embd_head,
                "comp_state_compress",
                il,
                topk_on ? &latent_pre : nullptr);

        // written before any indexer of this layer reads the cache: graph order is execution order here, as for the KV rows
        if (topk_on) {
            ggml_build_forward_expand(gf, build_dsv41_index_keys(model, inp_dsv4, tier, lid_ctx, latent_pre, il));
        }

        if (tier.k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, tier.k_rot);
            cb(kv_comp_hca, "comp_state_compress_rot", il);
        }

        const auto * tier_kv = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
        ggml_build_forward_expand(gf, tier_kv->cpy_k(ctx0,
                    kv_comp_hca, tier.state_write_idxs, il));
    }

    if (pooled && tier.state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, tier, hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, tier, hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, tier.state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, tier.state_persist_src_idxs);

        hca_state_kv = hca_state->cpy_kv(ctx0,
                hca_persist_kv, tier.state_persist_dst_idxs, il);
        hca_state_score = hca_state->cpy_score(ctx0,
                hca_persist_score, tier.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    // V4.1 sparse selection: an index source picks the rows and the layers up to the next index source reuse them.
    // The picks index the tier's compressed rows, so a tier change (ratio 2 -> 1 at layer 20) must coincide with an index
    // source, which the model's layer lists guarantee and the assert below checks.
    ggml_tensor * top_k = nullptr;
    if (topk_on && tier.kq_mask) {
        if (hparams.dsv41_is_index_source(il)) {
            dsv41_top_k       = build_dsv41_indexer_top_k(model, inp_dsv4, tier, lid_ctx, kv_src, qr, cur, inp_pos, il);
            dsv41_top_k_ratio = ratio;

            // gather path: pull the picks out of the source's cache once, on the source's device, for every layer up to the
            // next index source
            if (dsv41_sparse) {
                const auto * comp_ctx = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
                dsv41_sel_k = build_dsv41_gather_picks(comp_ctx->get_k(ctx0, kv_src), dsv41_top_k, kv_src);
            }
        }
        GGML_ASSERT(dsv41_top_k && "LLAMA_DSV41_TOPK: compressed layer before the first index source");
        GGML_ASSERT(dsv41_top_k_ratio == ratio && "LLAMA_DSV41_TOPK: reused top-k indexes a different compressed tier");
        GGML_ASSERT(dsv41_top_k->ne[1] == tier.kq_mask->ne[1] && dsv41_top_k->ne[3] == tier.kq_mask->ne[3]);
        top_k = dsv41_top_k;
    }

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, kv,
                nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (!v41 && tier_idx &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio != 0 && tier.kq_mask && !(v41 && compress_off)) {
        // the compressed rows live on the source layer, which for V4 is this layer itself
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il, kv_src, v41 && tier_idx, top_k);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

//
// V4.1 decoder skip (LLAMA_DSV41_DECODER_SKIP=1)
//
// V4.1 is a causal encoder-decoder: the encoder half compresses the prompt, and the decoder half reads one
// compressed KV that its first layer (the ratio 1 source) computes from the encoder output. DeepSeek's prefill
// runs only the encoder over the prompt; the decoder's own sliding-window cache for the prompt is filled by a
// "bounded replay" of just the last window of tokens, and the tech report calls that an approximation.
//
// This does the same inside one graph. A prompt ubatch (n_tokens > n_swa) runs the encoder over every token,
// lets the source layer write its compressed rows for every token, and then cuts everything per token - the
// hyper-connection stream, the carried mix coefficients, positions, raw-cache write idxs and mask rows - to the
// last n_swa tokens for the rest of the loop. Decoder K rows are written for those tokens only, and the mask
// for them drops this ubatch's earlier cells, which the decoder never writes. The next token attends to at most
// the n_swa - 1 tokens before it, so its window is covered.
//
// Ubatch boundaries: each prompt ubatch replays its own tail, so after the last one the decoder's window cache
// holds the tail of that ubatch plus the tail of the one before, which is what the first generated token needs.
// A tail token can only ever see cells of tokens that were in the tail of their ubatch (or in a ubatch small
// enough to run the full graph), so no unwritten row is ever read. Where the full graph would have let an early
// tail token see the encoder-only tokens before it, it now sees the compressed rows only - that is the
// approximation, and it mirrors the reference replay.
//
// Off by default; when off the graph is what it was. A decode ubatch (n_tokens <= n_swa) is never touched.
// Requests that want rows for tokens the decoder no longer sees - embeddings, all-token logits, layer-input
// taps past the source layer, unmasked next-n hidden states - take the full graph for that ubatch, as do
// ubatches holding more than one sequence (their tokens are not one run of positions). Every condition comes
// from the ubatch and the hparams, never from the cache: a one-sequence ubatch gets one raw mask block filled
// from its own stream's cells whatever the cache's stream count, and its write idxs are global rows
// (stream*size + cell), so a tail view of them is valid on a multi-stream cache too.
// The first decline of a prompt-sized ubatch is logged once with its reason, so a field run can be diagnosed.

static bool dsv41_decoder_skip_env() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_DSV41_DECODER_SKIP");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

struct dsv41_decoder_skip {
    bool    engaged = false;
    int     il_dec  = -1; // the decoder's KV source layer, where the narrowing happens
    int64_t n_tail  = 0;  // tokens the decoder half runs over

    const char * declined = nullptr; // when not engaged: the condition that failed
};

// The decoder's single KV source: a ratio 1 source layer with every layer after it at ratio 1 and reading its rows.
// Anything else is not the shape this optimisation is written for.
static int dsv41_decoder_source_layer(const llama_hparams & hparams) {
    if (hparams.dsv41_n_kv_source == 0 || hparams.dsv4_ratio_plain != 1) {
        return -1;
    }

    const uint32_t n_layer = hparams.n_layer();

    int il_dec = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.dsv4_compress_ratios[il] == 1 && hparams.dsv41_is_kv_source(il)) {
            il_dec = (int) il;
            break;
        }
    }
    if (il_dec < 0) {
        return -1;
    }

    for (uint32_t il = il_dec; il < n_layer; ++il) {
        if (hparams.dsv4_compress_ratios[il] != 1 || hparams.dsv41_kv_source_for(il) != il_dec) {
            return -1;
        }
    }

    return il_dec;
}

// Whether this ubatch takes the narrowed graph. Evaluated from the graph params so the reuse check can ask the same question.
// Every condition is read off the ubatch and the hparams; the cache's stream layout does not enter into it.
static dsv41_decoder_skip dsv41_plan_decoder_skip(const llm_graph_params & params) {
    dsv41_decoder_skip res;

    auto decline = [&](const char * why) {
        res.declined = why;
        return res;
    };

    if (!dsv41_decoder_skip_env()) {
        return decline("LLAMA_DSV41_DECODER_SKIP not set");
    }
    if (params.arch != LLM_ARCH_DEEPSEEK41) {
        return decline("arch is not deepseek41");
    }

    const auto & hparams = params.hparams;
    const auto & cparams = params.cparams;
    const auto & ubatch  = params.ubatch;

    const int64_t n_swa    = hparams.n_swa;
    const int64_t n_tokens = ubatch.n_tokens;

    if (n_swa <= 0) {
        return decline("n_swa == 0");
    }
    // a decode ubatch keeps the full graph
    if (n_tokens <= n_swa) {
        return decline("n_tokens <= n_swa (decode ubatch)");
    }

    const int il_dec = dsv41_decoder_source_layer(hparams);
    if (il_dec < 0) {
        return decline("hparams are not one ratio-1 KV source with every later layer reading it");
    }

    // everything that wants rows for tokens the decoder would no longer see
    if (!cparams.causal_attn) {
        return decline("causal_attn is off");
    }
    if (cparams.embeddings) {
        return decline("embeddings are on");
    }
    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked) {
        return decline("unmasked next-n hidden states are requested");
    }
    for (size_t il = (size_t) il_dec + 1; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            return decline("a layer-input tap past the source layer is requested");
        }
    }

    // one sequence, one run of positions: the tail is then the last n_swa positions of that sequence
    if (ubatch.n_pos != 1) {
        return decline("n_pos != 1");
    }
    if (ubatch.pos == nullptr || ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr) {
        return decline("ubatch carries no pos/seq_id");
    }
    if (ubatch.n_seqs_unq != 1) {
        return decline("n_seqs_unq != 1 (more than one sequence in the ubatch)");
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            return decline("n_seq_id != 1 (a token shared by sequences)");
        }
        if (ubatch.seq_id[i][0] != ubatch.seq_id[0][0]) {
            return decline("tokens of more than one seq_id");
        }
        if (ubatch.pos[i] != ubatch.pos[0] + i) {
            return decline("positions are not one contiguous run");
        }
    }

    if (params.mctx == nullptr) {
        return decline("no memory context");
    }

    // the outputs asked for have to be tail tokens; a prompt chunk asks for none, or for the last one
    const int64_t head = n_tokens - n_swa;
    if (params.n_outputs > 0) {
        if (ubatch.output == nullptr) {
            return decline("n_outputs > 0 without an output map");
        }
        int64_t n_tail_out = 0;
        for (int64_t i = 0; i < n_tokens; ++i) {
            if (!ubatch.output[i]) {
                continue;
            }
            if (i < head) {
                return decline("an output token is outside the tail");
            }
            ++n_tail_out;
        }
        if (n_tail_out != (int64_t) params.n_outputs) {
            return decline("n_outputs does not match the output map");
        }
    }

    res.engaged = true;
    res.il_dec  = il_dec;
    res.n_tail  = n_swa;

    return res;
}

// The last n_tail tokens of a per-token tensor. Tokens are the outermost dimension and the trailing rows of a
// contiguous tensor are contiguous themselves, so these stay views.
static ggml_tensor * dsv41_tail_2d(ggml_context * ctx, ggml_tensor * t, int64_t n_tail) {
    GGML_ASSERT(ggml_is_contiguous(t));
    GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1);
    GGML_ASSERT(t->ne[1] >= n_tail);

    return ggml_view_2d(ctx, t, t->ne[0], n_tail, t->nb[1], (t->ne[1] - n_tail)*t->nb[1]);
}

static ggml_tensor * dsv41_tail_3d(ggml_context * ctx, ggml_tensor * t, int64_t n_tail) {
    GGML_ASSERT(ggml_is_contiguous(t));
    GGML_ASSERT(t->ne[3] == 1);
    GGML_ASSERT(t->ne[2] >= n_tail);

    return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n_tail, t->nb[1], t->nb[2], (t->ne[2] - n_tail)*t->nb[2]);
}

// The last n_tail query rows of a one-stream kq mask [n_kv, n_tokens, 1, 1]. The strides of the unit dims are set so the view is contiguous, which flash attention requires.
static ggml_tensor * dsv41_tail_mask(ggml_context * ctx, ggml_tensor * m, int64_t n_tail) {
    GGML_ASSERT(m->ne[2] == 1 && m->ne[3] == 1);
    GGML_ASSERT(m->ne[1] >= n_tail);

    const size_t nb1 = m->nb[1];

    return ggml_view_4d(ctx, m, m->ne[0], n_tail, 1, 1, nb1, nb1*n_tail, nb1*n_tail, (m->ne[1] - n_tail)*nb1);
}

// The per-ubatch inputs the narrowed graph needs beyond views: the out ids shifted into the tail, and the decoder's raw mask rows.
class llm_graph_input_dsv41_tail : public llm_graph_input_i {
public:
    llm_graph_input_dsv41_tail(const dsv41_decoder_skip & skip, uint32_t n_outputs) : skip(skip), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_dsv41_tail() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    dsv41_decoder_skip skip;

    const uint32_t n_outputs;

    // the ubatch-wide raw inputs; registered before this one, so already set when this runs
    const llm_graph_input_dsv4_raw * inp_raw = nullptr;

    ggml_tensor * out_ids = nullptr; // I32 [n_outputs]: the ubatch out ids minus the head, indexing the tail
    ggml_tensor * kq_mask = nullptr; // F32/F16 [n_kv, n_tail, 1, 1]: the tail rows of the raw mask, with this ubatch's head cells masked out
};

void llm_graph_input_dsv41_tail::set_input(const llama_ubatch * ubatch) {
    if (!skip.engaged) {
        return;
    }

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_tail   = skip.n_tail;
    const int64_t head     = n_tokens - n_tail;

    GGML_ASSERT(head >= 0);

    if (out_ids) {
        GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
        GGML_ASSERT(ubatch->output);

        int32_t * data = (int32_t *) out_ids->data;

        int64_t n = 0;
        for (int64_t i = 0; i < n_tokens; ++i) {
            if (!ubatch->output[i]) {
                continue;
            }
            GGML_ASSERT(i >= head && "deepseek41 decoder skip: an output token is outside the tail");
            GGML_ASSERT(n < (int64_t) n_outputs);
            data[n++] = (int32_t) (i - head);
        }
        GGML_ASSERT(n == (int64_t) n_outputs);
    }

    if (kq_mask) {
        GGML_ASSERT(inp_raw && inp_raw->self_kq_mask && inp_raw->self_k_idxs && inp_raw->mctx);

        const ggml_tensor * src  = inp_raw->self_kq_mask;
        const ggml_tensor * idxs = inp_raw->self_k_idxs;

        GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(src->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(idxs->buffer));
        GGML_ASSERT(src->type == kq_mask->type);
        GGML_ASSERT(src->ne[0] == kq_mask->ne[0]);
        GGML_ASSERT(src->ne[1] == n_tokens && src->ne[2] == 1 && src->ne[3] == 1);
        GGML_ASSERT(kq_mask->ne[1] == n_tail);
        GGML_ASSERT(idxs->ne[0] == n_tokens);

        const int64_t n_kv = kq_mask->ne[0];
        const size_t  esz  = ggml_element_size(kq_mask);

        // the tail rows as the sliding window computed them
        memcpy(kq_mask->data, (const char *) src->data + head*src->nb[1], (size_t) n_tail*n_kv*esz);

        // then drop this ubatch's head cells: they sit inside the window of the first tail tokens, and no decoder layer writes them.
        // the idxs are global rows, stream*size + cell, and the mask columns are cells of the one stream this ubatch is on.
        const int64_t * kidx    = (const int64_t *) idxs->data;
        const int64_t   kv_size = inp_raw->mctx->get_size();

        GGML_ASSERT(kv_size > 0);

        const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);

        for (int64_t i = 0; i < head; ++i) {
            const int64_t col = kidx[i] % kv_size;
            if (col >= n_kv) {
                continue;
            }
            for (int64_t r = 0; r < n_tail; ++r) {
                if (kq_mask->type == GGML_TYPE_F16) {
                    ((ggml_fp16_t *) kq_mask->data)[r*n_kv + col] = f16_ninf;
                } else {
                    ((float *) kq_mask->data)[r*n_kv + col] = -INFINITY;
                }
            }
        }
    }
}

bool llm_graph_input_dsv41_tail::can_reuse(const llm_graph_params & params) {
    // a graph narrowed for one ubatch must not serve one that has an output outside the tail, and a full graph should not keep serving ubatches that could be narrowed
    const dsv41_decoder_skip next = dsv41_plan_decoder_skip(params);

    bool res = true;
    res &= next.engaged == skip.engaged;
    res &= next.il_dec  == skip.il_dec;
    res &= next.n_tail  == skip.n_tail;

    if (!skip.engaged) {
        return res;
    }

    res &= (out_ids ? out_ids->ne[0] : 0) == (int64_t) params.n_outputs;

    // the mask copy is sized to the raw mask, whose own reuse check covers the rest of its shape
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);
    res &= kq_mask != nullptr && mctx != nullptr && mctx->get_raw() != nullptr;
    if (res) {
        res &= kq_mask->ne[0] == (int64_t) mctx->get_raw()->get_n_kv();
    }

    return res;
}

// A dsv4 input object whose per-token tensors are tail views of the registered one's. It is never registered, so nothing sets it; the views read the buffers of the real inputs.
// The compressor-state tensors are shared as they are: after the source layer no layer runs the compressor.
static std::unique_ptr<llm_graph_input_dsv4> dsv41_build_tail_inputs(
        ggml_context * ctx,
        const llm_graph_input_dsv4 * inp,
        ggml_tensor * raw_kq_mask_tail,
        int64_t n_tail) {
    const llm_graph_input_dsv4_raw * raw = inp->get_raw();

    GGML_ASSERT(raw && raw->self_k_idxs && raw_kq_mask_tail);

    const int64_t n_tokens = raw->self_k_idxs->ne[0];
    const int64_t head     = n_tokens - n_tail;

    GGML_ASSERT(head >= 0);

    auto raw_tail = std::make_unique<llm_graph_input_dsv4_raw>(*raw);
    raw_tail->self_k_idxs      = ggml_view_1d(ctx, raw->self_k_idxs, n_tail, head*ggml_element_size(raw->self_k_idxs));
    raw_tail->self_kq_mask     = raw_kq_mask_tail;
    raw_tail->self_kq_mask_cnv = raw_kq_mask_tail;

    auto res = std::make_unique<llm_graph_input_dsv4>(inp->cparams, std::move(raw_tail), inp->mctx);
    res->inp_csa = inp->inp_csa;
    res->inp_hca = inp->inp_hca;
    res->inp_lid = inp->inp_lid;

    for (auto * ci : { &res->inp_csa, &res->inp_hca, &res->inp_lid }) {
        if (ci->kq_mask) {
            ci->kq_mask = dsv41_tail_mask(ctx, ci->kq_mask, n_tail);
        }
    }

    return res;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    // decided before any input is built, because it picks which out-ids input the graph gets
    const dsv41_decoder_skip skip = dsv41_plan_decoder_skip(params);

    std::unique_ptr<llm_graph_input_dsv41_tail> tail_inp;
    if (dsv41_decoder_skip_env() && arch == LLM_ARCH_DEEPSEEK41) {
        // registered even when not engaged, so the reuse check re-asks the question for the next ubatch
        tail_inp = std::make_unique<llm_graph_input_dsv41_tail>(skip, (uint32_t) n_outputs);
    }
    // WARN so the lines show at the server's default verbosity; each once per process
    if (tail_inp) {
        if (skip.engaged) {
            static bool logged_engaged = false;
            if (!logged_engaged) {
                LLAMA_LOG_WARN("deepseek41: decoder-skip engaged: layers %d-%d run over the last %d of %d tokens (n_outputs=%d)\n",
                        skip.il_dec, (int) n_layer - 1, (int) skip.n_tail, (int) n_tokens, (int) n_outputs);
                logged_engaged = true;
            }
        } else if ((int64_t) n_tokens > (int64_t) hparams.n_swa) {
            // a decode ubatch declines by design and is not worth a line
            static bool logged_declined = false;
            if (!logged_declined) {
                LLAMA_LOG_WARN("deepseek41: decoder-skip declined for a %d-token ubatch (n_outputs=%d): %s\n",
                        (int) n_tokens, (int) n_outputs, skip.declined ? skip.declined : "unknown");
                logged_declined = true;
            }
        }
    }

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = nullptr;
    if (skip.engaged && n_outputs > 0) {
        // the out ids index the narrowed stream; the ubatch-wide input is not built, since an input nothing reads has no buffer and aborts when set
        tail_inp->out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
        ggml_set_input(tail_inp->out_ids);
        ggml_set_name(tail_inp->out_ids, "dsv41_tail_out_ids");
        inp_out_ids = tail_inp->out_ids;
    } else {
        inp_out_ids = build_inp_out_ids();
    }

    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    // the tail input copies from the raw idxs and mask the dsv4 input sets, so it is registered after it
    llm_graph_input_dsv41_tail * inp_tail = nullptr;
    if (tail_inp) {
        tail_inp->inp_raw = inp_attn;
        if (skip.engaged) {
            // a one-sequence, non-coupled ubatch is written as itself and gets one mask block; the tail views rely on both
            GGML_ASSERT(inp_attn->self_k_idxs->ne[0] == n_tokens && "deepseek41 decoder skip: raw write idxs are not one per ubatch token");
            GGML_ASSERT(inp_attn->self_kq_mask->ne[3] == 1 && inp_attn->self_kq_mask->ne[1] == n_tokens && "deepseek41 decoder skip: raw mask is not one block of n_tokens rows");
            tail_inp->kq_mask = ggml_new_tensor_4d(ctx0, inp_attn->self_kq_mask->type, inp_attn->self_kq_mask->ne[0], skip.n_tail, 1, 1);
            ggml_set_input(tail_inp->kq_mask);
            ggml_set_name(tail_inp->kq_mask, "dsv41_tail_kq_mask");
            ggml_build_forward_expand(gf, tail_inp->kq_mask);
        }
        inp_tail = (llm_graph_input_dsv41_tail *) res->add_input(std::move(tail_inp));
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // V4.1 shifts the hyper-connection coefficients by one sublayer, so the mix a sublayer computes is carried to the next one.
    // The run starts from a one-hot mix on copy 0, and collapsing with a one-hot is just selecting that copy, so no constant tensor is needed.
    const bool hc_shift = model.hc_head_fn == nullptr;
    ggml_tensor * carry_pre = nullptr;

    // LLAMA_DSV41_NO_ENGRAM=1 skips the engram contribution, to tell a broken gate from a broken stack.
    // The flag is read here rather than inside the loop because an input that no consumer reads is still expanded, and an expanded input with no buffer aborts at load.
    static const bool engram_off = [] {
        const char * e = getenv("LLAMA_DSV41_NO_ENGRAM");
        const bool off = e != nullptr && atoi(e) != 0;
        if (off) {
            LLAMA_LOG_WARN("deepseek41: LLAMA_DSV41_NO_ENGRAM=1, engram disabled\n");
        }
        return off;
    }();

    // one row index per hash column per table, hashed host side from the token history
    ggml_tensor * inp_engram = nullptr;
    if (hparams.engram_n_layers > 0 && !engram_off) {
        inp_engram = build_inp_engram(model);
        ggml_build_forward_expand(gf, inp_engram);
    }

    // the per-token inputs the loop reads; the decoder skip swaps these for tail views at its source layer
    ggml_tensor * inp_pos_cur    = inp_pos;
    ggml_tensor * inp_tokens_cur = res->t_inp_tokens;
    ggml_tensor * inp_engram_cur = inp_engram;
    llm_graph_input_dsv4 * inp_dsv4_cur = inp_dsv4;
    std::unique_ptr<llm_graph_input_dsv4> inp_dsv4_tail;

    for (int il = 0; il < n_layer; ++il) {
        // the engram writes into the residual before the block body, so the whole layer sees it.
        if (!engram_off && model.layers[il].engram_embed) {
            inpL = build_engram(model, inpL, inp_engram_cur, il);
            cb(inpL, "engram_out", il);
        }

        // DSpark tap: the mean over the hc copies of this layer's attention input. The V4.1 reference takes it after the
        // layer's engram (Transformer.forward: engram, then main_hiddens.append(h.mean(dim=2)), then the block), so the
        // tap sits after it here too; V4 has no engram, and its sidecar's target_layers already carry the +1 that turns
        // V4's "output of layer i" into "input of layer i+1".
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            // expanded by set_outputs at the END of the build - expanding here walks
            // the dependency chain in a different DFS order than the natural build
            // and interleaves every device boundary, adding 2 sched splits per
            // boundary under -sm layer (measured 65 vs 51 splits)
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        // V4.1 layer 0 has no carried mix yet: the initial one-hot selects copy 0, so no collapse is built for it.
        // (Building one and discarding it left a fused node outside the graph, which disabled the fused HC ops for all layers.)
        const bool hc_onehot = hc_shift && carry_pre == nullptr;
        ggml_tensor * attn_pre = nullptr;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il, &attn_pre, hc_shift ? carry_pre : nullptr, /*collapse=*/ !hc_onehot);
        if (hc_onehot) {
            // the initial one-hot mix selects copy 0
            cur = ggml_cont_2d(ctx0, ggml_view_2d(ctx0, inpL, n_embd, inpL->ne[2], inpL->nb[2], 0),
                    n_embd, inpL->ne[2]);
        }
        GGML_ASSERT(cur != nullptr);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * cur_comp = nullptr;
        if (skip.engaged && il == skip.il_dec) {
            // The decoder's KV source. Its compressed rows for every token come from cur as it stands here - the
            // encoder output, normalized - so the compressor keeps the whole ubatch (cur_comp). Everything else per
            // token is cut to the tail from here on: this layer's own attention and FFN, and every layer after it.
            const int64_t n_tail = skip.n_tail;
            const int64_t head   = n_tokens - n_tail;

            cur_comp = cur;
            cur      = dsv41_tail_2d(ctx0, cur,      n_tail);
            residual = dsv41_tail_3d(ctx0, residual, n_tail);
            post     = dsv41_tail_2d(ctx0, post,     n_tail);
            comb     = dsv41_tail_3d(ctx0, comb,     n_tail);
            attn_pre = dsv41_tail_2d(ctx0, attn_pre, n_tail);
            inpL     = residual;
            cb(cur, "dsv41_tail_attn_norm", il);

            GGML_ASSERT(inp_pos->ne[0] == n_tokens);
            inp_pos_cur = ggml_view_1d(ctx0, inp_pos, n_tail, head*ggml_element_size(inp_pos));
            if (inp_tokens_cur) {
                GGML_ASSERT(inp_tokens_cur->ne[0] == n_tokens);
                inp_tokens_cur = ggml_view_1d(ctx0, inp_tokens_cur, n_tail, head*ggml_element_size(inp_tokens_cur));
            }
            if (inp_engram_cur) {
                // [n_cols*n_tokens, n_eng]: the tail of every table's run of rows
                const int64_t n_cols = hparams.engram_n_hash_cols();
                inp_engram_cur = ggml_view_2d(ctx0, inp_engram, n_cols*n_tail, inp_engram->ne[1], inp_engram->nb[1],
                        head*n_cols*ggml_element_size(inp_engram));
            }

            GGML_ASSERT(inp_tail && inp_tail->kq_mask);
            inp_dsv4_tail = dsv41_build_tail_inputs(ctx0, inp_dsv4, inp_tail->kq_mask, n_tail);
            inp_dsv4_cur  = inp_dsv4_tail.get();
        }

        cur = build_attention(model, inp_dsv4_cur, cur, inp_pos_cur, il, cur_comp);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il, &last_ffn_pre, hc_shift ? attn_pre : nullptr);
        if (hc_shift) {
            carry_pre = last_ffn_pre;
        }
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens_cur);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        // expanded by set_outputs, see the per-layer taps above
    }

    // the stream is n_tokens wide, or n_tail wide after the decoder skip; the out ids index whichever it is
    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, inpL->ne[2]);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);

        // V4.1 folds with the mix the last layer computed, and that mix has one row per token. build_hc_pre() takes its row count from x, so once x is narrowed to the output rows an un-narrowed mix silently supplies row 0 - the first token's coefficients applied to the last token.
        // Narrow it the same way.
        if (last_ffn_pre) {
            last_ffn_pre = ggml_get_rows(ctx0, last_ffn_pre, inp_out_ids);
            cb(last_ffn_pre, "hc_ffn_pre_out", -1);
        }
    }

    cur = build_head_fold(model, inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK4 MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    GGML_ASSERT(ubatch.token && "DEEPSEEK4 MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "DEEPSEEK4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT((uint32_t) il >= hparams.dsv4_hash_layer_count && "DEEPSEEK4 MTP does not support hash-routed MTP blocks");
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    inpL = build_cvec(inpL, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);

    cur = build_head_fold(model, inpL);
    cb(cur, "mtp_hc_head", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK4 MTP missing shared head norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DEEPSEEK4 MTP missing LM head");
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
