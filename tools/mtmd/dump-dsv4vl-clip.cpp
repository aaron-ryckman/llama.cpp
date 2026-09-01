// Dump DSV4VL clip embeddings from a pre-normalized HWC f32 image.
// Input .hwc: int32 nx, ny; then nx*ny*3 float32 HWC already (x/255-0.5)/0.5
// Output .clip: int32 n_tokens, n_embd; then n_tokens*n_embd float32 (token-major)

#include "clip.h"
#include "clip-impl.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

struct dump_ud {
    std::unordered_map<std::string, std::vector<float>> * out;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const char * name = ggml_get_name(t);
    if (name == nullptr || name[0] == 0) {
        return false;
    }
    if (std::strcmp(name, "patch") != 0 && std::strcmp(name, "vit") != 0 &&
        std::strcmp(name, "unfold") != 0 && std::strcmp(name, "aligner") != 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    auto * ud = static_cast<dump_ud *>(user_data);
    std::vector<float> buf((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    fprintf(stderr, "dump %s ne=[%lld,%lld,%lld,%lld] n=%zu\n",
            name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], buf.size());
    (*ud->out)[name] = std::move(buf);
    return true;
}

static void ggml_log_stderr(enum ggml_log_level /*level*/, const char * text, void * /*ud*/) {
    fputs(text, stderr);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <mmproj.gguf> <in.hwc> <out.clip>\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const char * in_hwc = argv[2];
    const char * out_path = argv[3];

    ggml_backend_load_all();
    ggml_log_set(ggml_log_stderr, nullptr);

    std::ifstream in(in_hwc, std::ios::binary);
    if (!in) {
        fprintf(stderr, "failed to open %s\n", in_hwc);
        return 1;
    }
    int32_t nx = 0, ny = 0;
    in.read(reinterpret_cast<char *>(&nx), 4);
    in.read(reinterpret_cast<char *>(&ny), 4);
    if (nx <= 0 || ny <= 0) {
        fprintf(stderr, "bad size %d x %d\n", nx, ny);
        return 1;
    }
    std::vector<float> hwc((size_t) nx * (size_t) ny * 3);
    in.read(reinterpret_cast<char *>(hwc.data()), hwc.size() * sizeof(float));
    if (!in) {
        fprintf(stderr, "short read of %s\n", in_hwc);
        return 1;
    }
    in.close();
    fprintf(stderr, "hwc %d x %d  n=%zu\n", nx, ny, hwc.size());

    std::unordered_map<std::string, std::vector<float>> dumps;
    dump_ud ud{&dumps};

    clip_context_params cparams{};
    cparams.use_gpu = false;
    cparams.device = nullptr;
    cparams.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cparams.image_min_tokens = 0;
    cparams.image_max_tokens = 0;
    cparams.warmup = false;
    cparams.cb_eval = dump_cb;
    cparams.cb_eval_user_data = &ud;
    cparams.no_alloc = false;
    cparams.progress_callback = nullptr;
    cparams.progress_callback_user_data = nullptr;

    clip_init_result init = clip_init(mmproj, cparams);
    if (!init.ctx_v) {
        fprintf(stderr, "clip_init failed for %s\n", mmproj);
        return 1;
    }
    clip_ctx * ctx = init.ctx_v;

    clip_image_f32 img;
    img.set_size({nx, ny}, false, false);
    img.cpy_buf(hwc);

    const int n_tok = clip_n_output_tokens(ctx, &img);
    const int n_embd = clip_n_mmproj_embd(ctx);
    fprintf(stderr, "n_output_tokens=%d n_mmproj_embd=%d\n", n_tok, n_embd);

    std::vector<float> out((size_t) n_tok * (size_t) n_embd);
    if (!clip_image_encode(ctx, 24, &img, out)) {
        fprintf(stderr, "clip_image_encode failed\n");
        clip_free(ctx);
        return 1;
    }
    fprintf(stderr, "encoded floats=%zu expected=%d\n", out.size(), n_tok * n_embd);
    if ((int) out.size() != n_tok * n_embd) {
        fprintf(stderr, "size mismatch\n");
        clip_free(ctx);
        return 1;
    }
    fprintf(stderr, "token0[:8]=");
    for (int i = 0; i < 8 && i < n_embd; i++) {
        fprintf(stderr, " %.6f", out[i]);
    }
    fprintf(stderr, "\n");

    std::ofstream outf(out_path, std::ios::binary);
    int32_t n_tok32 = n_tok, n_embd32 = n_embd;
    outf.write(reinterpret_cast<char *>(&n_tok32), 4);
    outf.write(reinterpret_cast<char *>(&n_embd32), 4);
    outf.write(reinterpret_cast<char *>(out.data()), out.size() * sizeof(float));
    outf.close();
    fprintf(stderr, "wrote %s\n", out_path);

    for (const auto & kv : dumps) {
        const std::string path = std::string(out_path) + "." + kv.first;
        std::ofstream f(path, std::ios::binary);
        int32_t n = (int32_t) kv.second.size();
        f.write(reinterpret_cast<char *>(&n), 4);
        f.write(reinterpret_cast<const char *>(kv.second.data()), kv.second.size() * sizeof(float));
        fprintf(stderr, "wrote %s n=%d\n", path.c_str(), n);
    }

    clip_free(ctx);
    if (init.ctx_a) {
        clip_free(init.ctx_a);
    }
    if (init.ctx_gen_a) {
        clip_free(init.ctx_gen_a);
    }
    return 0;
}
