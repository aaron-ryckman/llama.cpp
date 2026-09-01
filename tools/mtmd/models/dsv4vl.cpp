#include "models.h"

#include <vector>

enum {
    DSV4_START   = 0,
    DSV4_PAD     = 1,
    DSV4_IMAGE   = 2,
    DSV4_NEWLINE = 3,
    DSV4_END     = 4,
};

static void dsv4_layout(int n_llm_h, int n_llm_w,
                        std::vector<int> & types, std::vector<int> & perm) {
    const int pad_h = n_llm_h % 2;
    const int rows = n_llm_h + pad_h;
    const int row_len = n_llm_w + 1;
    const int pad_last = ((rows / 2) * row_len % 2) * 2;

    std::vector<int> types_grid;
    types_grid.reserve(rows * row_len);
    for (int h = 0; h < n_llm_h; ++h) {
        for (int w = 0; w < n_llm_w; ++w) {
            types_grid.push_back(DSV4_IMAGE);
        }
        types_grid.push_back(DSV4_NEWLINE);
    }
    for (int i = 0; i < row_len * pad_h; ++i) {
        types_grid.push_back(DSV4_PAD);
    }

    std::vector<int> order;
    order.reserve(rows * row_len);
    for (int p = 0; p < rows / 2; ++p) {
        for (int c = 0; c < row_len; ++c) {
            order.push_back(p * 2 * row_len + c);
            order.push_back((p * 2 + 1) * row_len + c);
        }
    }

    std::vector<int> image_idx(rows * row_len, -1);
    int img = 0;
    for (int h = 0; h < n_llm_h; ++h) {
        for (int w = 0; w < n_llm_w; ++w) {
            image_idx[h * row_len + w] = img++;
        }
    }

    types.clear();
    perm.clear();
    types.push_back(DSV4_START);
    for (int o : order) {
        types.push_back(types_grid[(size_t) o]);
        if (image_idx[(size_t) o] >= 0) {
            perm.push_back(image_idx[(size_t) o]);
        }
    }
    for (int i = 0; i < pad_last; ++i) {
        types.push_back(DSV4_PAD);
    }
    types.push_back(DSV4_END);
}

static ggml_tensor * tok1(ggml_context * ctx, ggml_tensor * vec, int n_embd, ggml_type type) {
    ggml_tensor * t = ggml_reshape_2d(ctx, vec, n_embd, 1);
    if (t->type != type) {
        t = ggml_cast(ctx, t, type);
    }
    return t;
}

ggml_cgraph * clip_graph_dsv4vl::build() {
    const int merge = hparams.n_merge > 0 ? hparams.n_merge : 3;
    const int rd = d_head / 2;

    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rd, n_patches);
    ggml_set_name(rope_cos, "rope_cos");
    ggml_set_input(rope_cos);

    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rd, n_patches);
    ggml_set_name(rope_sin, "rope_sin");
    ggml_set_input(rope_sin);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        }
        // cur is [d_head, n_head, n_pos]; rotate-half uses first/last rd of each head
        ggml_tensor * x1 = ggml_view_4d(ctx0, cur, rd, n_head, n_patches, 1,
            cur->nb[1], cur->nb[2], cur->nb[3], 0);
        ggml_tensor * x2 = ggml_view_4d(ctx0, cur, rd, n_head, n_patches, 1,
            cur->nb[1], cur->nb[2], cur->nb[3],
            ggml_row_size(cur->type, rd));
        x1 = ggml_cont(ctx0, x1);
        x2 = ggml_cont(ctx0, x2);
        ggml_tensor * c = ggml_reshape_4d(ctx0, rope_cos, rd, 1, n_patches, 1);
        ggml_tensor * s = ggml_reshape_4d(ctx0, rope_sin, rd, 1, n_patches, 1);
        c = ggml_repeat(ctx0, c, x1);
        s = ggml_repeat(ctx0, s, x1);
        ggml_tensor * y1 = ggml_sub(ctx0, ggml_mul(ctx0, x1, c), ggml_mul(ctx0, x2, s));
        ggml_tensor * y2 = ggml_add(ctx0, ggml_mul(ctx0, x2, c), ggml_mul(ctx0, x1, s));
        return ggml_concat(ctx0, y1, y2, 0);
    };

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        ggml_tensor * b = model.patch_bias;
        if (b->type != inp->type) {
            b = ggml_cast(ctx0, b, inp->type);
        }
        inp = ggml_add(ctx0, inp, b);
    }
    cb(inp, "patch", -1);
    ggml_tensor * cur = build_vit(
        inp, n_patches,
        NORM_TYPE_RMS,
        FFN_SILU,
        nullptr,
        add_pos);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_patches);
    cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);

    const int pad_x = (merge - (n_patches_x % merge)) % merge;
    const int pad_y = (merge - (n_patches_y % merge)) % merge;
    if (pad_x > 0 || pad_y > 0) {
        cur = ggml_pad(ctx0, cur, 0, pad_x, pad_y, 0);
    }
    const int px = n_patches_x + pad_x;
    const int py = n_patches_y + pad_y;
    const int n_llm_w = px / merge;
    const int n_llm_h = py / merge;

    // ggml_im2col 2D wants [W, H, C]; permute(2,0,1,3) sends [C,W,H] → [W,H,C]
    cb(cur, "vit", -1);
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx0, cur);
    ggml_tensor * kernel = ggml_view_3d(ctx0, cur, merge, merge, cur->ne[2], 0, 0, 0);
    cur = ggml_im2col(ctx0, kernel, cur, merge, merge, 0, 0, 1, 1, true, GGML_TYPE_F32);
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "unfold", -1);

    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF, -1);
    cb(cur, "aligner", -1);

    // cur: [proj_dim, n_llm_h * n_llm_w]
    std::vector<int> types, perm;
    dsv4_layout(n_llm_h, n_llm_w, types, perm);
    GGML_ASSERT((int) perm.size() == n_llm_h * n_llm_w);

    const int n_proj = (int) cur->ne[0];
    ggml_tensor * packed = nullptr;
    int img_i = 0;
    for (int ty : types) {
        ggml_tensor * tok = nullptr;
        if (ty == DSV4_IMAGE) {
            const int row = perm[(size_t) img_i++];
            tok = ggml_view_2d(ctx0, cur, n_proj, 1,
                ggml_row_size(cur->type, n_proj),
                (size_t) row * ggml_row_size(cur->type, n_proj));
            tok = ggml_cont(ctx0, tok);
        } else if (ty == DSV4_START) {
            tok = tok1(ctx0, model.mm_img_begin, n_proj, cur->type);
        } else if (ty == DSV4_END) {
            tok = tok1(ctx0, model.mm_img_end, n_proj, cur->type);
        } else if (ty == DSV4_NEWLINE) {
            tok = tok1(ctx0, model.image_newline, n_proj, cur->type);
        } else {
            tok = tok1(ctx0, model.image_pad, n_proj, cur->type);
        }
        packed = packed ? ggml_concat(ctx0, packed, tok, 1) : tok;
    }

    ggml_build_forward_expand(gf, packed);
    return gf;
}
