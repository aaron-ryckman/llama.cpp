// llama-moe-route-dump: record what the MoE router did, per token and per MoE layer, for every token of every document
// in a prompt file. Used for expert-routing maps (residency) and expert-pruning saliency (frequency, REAP).
//
// Documents are separated by a line containing exactly "<<<DOC>>>". Each document is tokenized on its own (BOS as the
// vocab says, special tokens parsed, as the server does), truncated to n_ctx, and decoded from an empty KV cache in
// n_batch chunks. The tool reads graph tensors through the eval callback, so it needs no llama.cpp source change; the
// scheduler cuts the graph after every observed tensor (no fusion across those cuts, e.g. the fused top-k MoE kernel).
//
// Environment:
//   ROUTE_DUMP_OUT          path of the ids stream (required)
//   ROUTE_DUMP_STREAMS      sidecar streams to write next to it, comma separated: weights,probs,norms,bias
//                           (default: all four; "none" writes the ids stream only, as before 2026-10-12)
//   ROUTE_DUMP_WEIGHTS      graph name (without "-<layer>") of the mixing weights the layer applies
//                           (default "ffn_moe_weights_scaled": build_moe_ffn's weights after normalization and scaling;
//                           a model without a weight scale ends at "ffn_moe_weights_norm" or "ffn_moe_weights")
//   ROUTE_DUMP_NORM_THREADS host threads for the expert-output norm reduction (default 8)
//
// ids stream (ROUTE_DUMP_OUT): format unchanged since 2026-10-11, no header, little endian.
//   record:     int32 layer, int32 n_tokens, int32 k, then n_tokens*k int32 expert ids (token-major; slot order is the
//               router's top-k order)
//   doc marker: int32 -1, int32 doc_tokens, int32 0 (no payload), after each document
//
// sidecar streams: <ROUTE_DUMP_OUT minus a trailing ".bin">.<stream>.bin, e.g. v41.routes.bin -> v41.routes.norms.bin
//   header, 64 bytes: char[4] "MRDS", int32 version (1), int32 stream (1 weights, 2 probs, 3 norms, 4 bias),
//                     int32 dtype (0 f32, 1 f16), int32 element bytes, int32[3] reserved (0),
//                     char[32] source graph name (NUL padded)
//   weights, probs, norms: the same record / doc-marker sequence as the ids stream. Record i of each of them describes
//   the same (layer, ubatch) as record i of the ids stream (the tool checks this at every document end):
//     record:     int32 layer, int32 n_tokens, int32 width, then n_tokens*width elements (token-major)
//     doc marker: int32 -1, int32 doc_tokens, int32 0
//     weights  width k, f32: the weight applied to each selected expert's output, in the ids' slot order
//     probs    width n_expert, f16: the router's gating output for every expert ("ffn_moe_probs"), before the
//              expert-selection bias (exp_probs_b) is added; selection uses probs + bias, the weights use probs
//     norms    width k, f32: L2 norm of each selected expert's output before weighting ("ffn_moe_down",
//              [n_embd, k, n_tokens]), in the ids' slot order; reduced on the host in double precision
//   bias: one record per layer that has a selection bias, written the first time the layer runs, no doc markers:
//     int32 layer, int32 1, int32 n_expert, then n_expert f32 (the exp_probs_b the router adds before top-k)
//
// Never writes token ids or text: token ids are the text.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

enum stream_kind : int32_t { STREAM_WEIGHTS = 1, STREAM_PROBS = 2, STREAM_NORMS = 3, STREAM_BIAS = 4 };
enum stream_dtype : int32_t { DTYPE_F32 = 0, DTYPE_F16 = 1 };

struct sidecar {
    const char * key = "";
    std::string  graph_name;   // without "-<layer>"
    int32_t      kind  = 0;
    int32_t      dtype = DTYPE_F32;
    FILE *       f     = nullptr;
    long long    records = 0;
    uint64_t     seq_hash = 1469598103934665603ULL; // FNV-1a over the (layer, n_tokens) record sequence
    double       seconds = 0.0;                     // time spent in the callback for this stream
};

static void seq_mix(uint64_t & h, int32_t a, int32_t b) {
    const int32_t v[2] = {a, b};
    const auto * p = reinterpret_cast<const unsigned char *>(v);
    for (size_t i = 0; i < sizeof(v); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
}

struct dump_state {
    FILE *    out = nullptr;
    long long records = 0;
    uint64_t  seq_hash = 1469598103934665603ULL;
    double    seconds = 0.0;
    bool      bad = false;

    sidecar weights, probs, norms, bias;
    std::set<int32_t> bias_done;
    int norm_threads = 8;

    std::vector<int32_t>     ids;
    std::vector<float>       fbuf;
    std::vector<ggml_fp16_t> hbuf;
    std::vector<float>       down;
    std::vector<float>       nbuf;
};

static const char * kIds   = "ffn_moe_topk";
static const char * kProbs = "ffn_moe_probs";
static const char * kDown  = "ffn_moe_down";

// "<base>-<digits>" -> layer; anything else (other names, "(view)"/"(reshaped)" suffixes) -> -1
static int32_t match_layer(const char * name, const std::string & base) {
    const size_t n = base.size();
    if (strncmp(name, base.c_str(), n) != 0 || name[n] != '-' || name[n + 1] == '\0') {
        return -1;
    }
    for (const char * p = name + n + 1; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
    }
    return atoi(name + n + 1);
}

static bool open_sidecar(sidecar & s, const std::string & path) {
    s.f = fopen(path.c_str(), "wb");
    if (!s.f) {
        LOG_ERR("cannot open %s\n", path.c_str());
        return false;
    }
    char head[64] = {0};
    memcpy(head, "MRDS", 4);
    const int32_t fields[7] = {1, s.kind, s.dtype, s.dtype == DTYPE_F16 ? 2 : 4, 0, 0, 0};
    memcpy(head + 4, fields, sizeof(fields));
    strncpy(head + 32, s.graph_name.c_str(), 31);
    fwrite(head, 1, sizeof(head), s.f);
    return true;
}

static void write_record(sidecar & s, int32_t layer, int32_t n_tokens, int32_t width, const void * data, size_t bytes) {
    const int32_t head[3] = {layer, n_tokens, width};
    fwrite(head, sizeof(int32_t), 3, s.f);
    fwrite(data, 1, bytes, s.f);
    s.records++;
    seq_mix(s.seq_hash, layer, n_tokens);
}

// rows of a 2-D view [width, n_rows] (row stride nb[1]) into dst
static void get_rows_f32(const ggml_tensor * t, int64_t width, int64_t n_rows, size_t row_stride, float * dst) {
    if (row_stride == (size_t) width * sizeof(float)) {
        ggml_backend_tensor_get(t, dst, 0, (size_t) width * n_rows * sizeof(float));
        return;
    }
    for (int64_t j = 0; j < n_rows; ++j) {
        ggml_backend_tensor_get(t, dst + j * width, (size_t) j * row_stride, (size_t) width * sizeof(float));
    }
}

static bool on_ids(dump_state * st, ggml_tensor * t, int32_t layer) {
    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s: unexpected type %d\n", t->name, (int) t->type);
        return false;
    }
    const int32_t k = (int32_t) t->ne[0];
    const int32_t n_tokens = (int32_t) t->ne[1];
    st->ids.resize((size_t) k * n_tokens);
    // The top-k tensor can be a strided view of the argsort output: copy row by row using nb[1].
    for (int32_t j = 0; j < n_tokens; ++j) {
        ggml_backend_tensor_get(t, st->ids.data() + (size_t) j * k, (size_t) j * t->nb[1], (size_t) k * sizeof(int32_t));
    }
    for (int32_t v : st->ids) {
        if (v < 0 || v >= 512) {
            LOG_ERR("%s: expert id %d out of range\n", t->name, v);
            return false;
        }
    }
    const int32_t head[3] = {layer, n_tokens, k};
    fwrite(head, sizeof(int32_t), 3, st->out);
    fwrite(st->ids.data(), sizeof(int32_t), st->ids.size(), st->out);
    st->records++;
    seq_mix(st->seq_hash, layer, n_tokens);

    // selection bias: topk = view(argsort(probs + exp_probs_b)); read it once per layer
    if (st->bias.f && !st->bias_done.count(layer)) {
        const ggml_tensor * sort = t->src[0];
        const ggml_tensor * add  = sort ? sort->src[0] : nullptr;
        const ggml_tensor * b    = (add && add->op == GGML_OP_ADD) ? add->src[1] : nullptr;
        if (b && b->type == GGML_TYPE_F32 && ggml_nelements(b) == add->ne[0] && ggml_is_contiguous(b)) {
            st->fbuf.resize((size_t) ggml_nelements(b));
            ggml_backend_tensor_get(b, st->fbuf.data(), 0, ggml_nbytes(b));
            write_record(st->bias, layer, 1, (int32_t) ggml_nelements(b), st->fbuf.data(), ggml_nbytes(b));
        }
        st->bias_done.insert(layer);
    }
    return true;
}

static bool on_probs(dump_state * st, ggml_tensor * t, int32_t layer) {
    if (t->type != GGML_TYPE_F32) {
        LOG_ERR("%s: unexpected type %d\n", t->name, (int) t->type);
        return false;
    }
    const int64_t n_expert = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    st->fbuf.resize((size_t) n_expert * n_tokens);
    st->hbuf.resize((size_t) n_expert * n_tokens);
    get_rows_f32(t, n_expert, n_tokens, t->nb[1], st->fbuf.data());
    ggml_fp32_to_fp16_row(st->fbuf.data(), st->hbuf.data(), (int64_t) st->fbuf.size());
    write_record(st->probs, layer, (int32_t) n_tokens, (int32_t) n_expert, st->hbuf.data(),
                 st->hbuf.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool on_weights(dump_state * st, ggml_tensor * t, int32_t layer) {
    if (t->type != GGML_TYPE_F32) {
        LOG_ERR("%s: unexpected type %d\n", t->name, (int) t->type);
        return false;
    }
    // [1, k, n_tokens] (after reshape/scale) or [k, n_tokens] (after the normalizing div)
    const bool    lead1    = t->ne[0] == 1;
    const int64_t k        = lead1 ? t->ne[1] : t->ne[0];
    const int64_t n_tokens = lead1 ? t->ne[2] : t->ne[1];
    const size_t  row_nb   = lead1 ? t->nb[2] : t->nb[1];
    const size_t  el_nb    = lead1 ? t->nb[1] : t->nb[0];
    st->fbuf.resize((size_t) k * n_tokens);
    if (el_nb == sizeof(float)) {
        get_rows_f32(t, k, n_tokens, row_nb, st->fbuf.data());
    } else {
        for (int64_t j = 0; j < n_tokens; ++j) {
            for (int64_t s = 0; s < k; ++s) {
                ggml_backend_tensor_get(t, st->fbuf.data() + j * k + s, j * row_nb + s * el_nb, sizeof(float));
            }
        }
    }
    write_record(st->weights, layer, (int32_t) n_tokens, (int32_t) k, st->fbuf.data(), st->fbuf.size() * sizeof(float));
    return true;
}

static bool on_down(dump_state * st, ggml_tensor * t, int32_t layer) {
    if (t->type != GGML_TYPE_F32) {
        LOG_ERR("%s: unexpected type %d\n", t->name, (int) t->type);
        return false;
    }
    const int64_t n_embd = t->ne[0], k = t->ne[1], n_tokens = t->ne[2];
    st->down.resize((size_t) n_embd * k * n_tokens);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, st->down.data(), 0, ggml_nbytes(t));
    } else {
        for (int64_t j = 0; j < n_tokens; ++j) {
            for (int64_t s = 0; s < k; ++s) {
                ggml_backend_tensor_get(t, st->down.data() + (j * k + s) * n_embd, j * t->nb[2] + s * t->nb[1],
                                        (size_t) n_embd * sizeof(float));
            }
        }
    }
    const int64_t rows = k * n_tokens;
    st->nbuf.resize((size_t) rows);
    const float * src = st->down.data();
    float *       dst = st->nbuf.data();
    auto reduce = [src, dst, n_embd](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const float * x = src + r * n_embd;
            double acc = 0.0;
            for (int64_t e = 0; e < n_embd; ++e) {
                acc += (double) x[e] * (double) x[e];
            }
            dst[r] = (float) std::sqrt(acc);
        }
    };
    const int n_thr = (int) std::max<int64_t>(1, std::min<int64_t>(st->norm_threads, rows / 64));
    if (n_thr <= 1) {
        reduce(0, rows);
    } else {
        std::vector<std::thread> pool;
        const int64_t per = (rows + n_thr - 1) / n_thr;
        for (int i = 0; i < n_thr; ++i) {
            const int64_t r0 = i * per, r1 = std::min(rows, r0 + per);
            if (r0 < r1) {
                pool.emplace_back(reduce, r0, r1);
            }
        }
        for (auto & th : pool) {
            th.join();
        }
    }
    for (float v : st->nbuf) {
        if (!std::isfinite(v)) {
            LOG_ERR("%s: non-finite expert output norm\n", t->name);
            return false;
        }
    }
    write_record(st->norms, layer, (int32_t) n_tokens, (int32_t) k, st->nbuf.data(), st->nbuf.size() * sizeof(float));
    return true;
}

static bool route_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<dump_state *>(user_data);
    int32_t layer = match_layer(t->name, kIds);
    int which = layer >= 0 ? 0 : -1;
    if (which < 0 && st->probs.f && (layer = match_layer(t->name, st->probs.graph_name)) >= 0) {
        which = 1;
    }
    if (which < 0 && st->weights.f && (layer = match_layer(t->name, st->weights.graph_name)) >= 0) {
        which = 2;
    }
    if (which < 0 && st->norms.f && (layer = match_layer(t->name, st->norms.graph_name)) >= 0) {
        which = 3;
    }
    if (ask) {
        return which >= 0;
    }
    if (which < 0) {
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = false;
    double * secs = nullptr;
    switch (which) {
        case 0: ok = on_ids(st, t, layer);     secs = &st->seconds;         break;
        case 1: ok = on_probs(st, t, layer);   secs = &st->probs.seconds;   break;
        case 2: ok = on_weights(st, t, layer); secs = &st->weights.seconds; break;
        case 3: ok = on_down(st, t, layer);    secs = &st->norms.seconds;   break;
    }
    *secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        st->bad = true;
    }
    return ok;
}

static std::vector<std::string> split_docs(const std::string & text) {
    std::vector<std::string> docs;
    std::istringstream in(text);
    std::string line, cur;
    while (std::getline(in, line)) {
        if (line == "<<<DOC>>>") {
            if (!cur.empty()) docs.push_back(cur);
            cur.clear();
        } else {
            cur += line;
            cur += '\n';
        }
    }
    if (!cur.empty()) docs.push_back(cur);
    return docs;
}

static std::string sidecar_path(const std::string & out, const char * key) {
    std::string base = out;
    if (base.size() > 4 && base.compare(base.size() - 4, 4, ".bin") == 0) {
        base.resize(base.size() - 4);
    }
    return base + "." + key + ".bin";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * out_path = getenv("ROUTE_DUMP_OUT");
    if (!out_path) {
        fprintf(stderr, "set ROUTE_DUMP_OUT\n");
        return 1;
    }
    const char * streams_env = getenv("ROUTE_DUMP_STREAMS");
    const std::string streams = streams_env ? streams_env : "weights,probs,norms,bias";
    const char * weights_env = getenv("ROUTE_DUMP_WEIGHTS");
    const char * threads_env = getenv("ROUTE_DUMP_NORM_THREADS");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    dump_state st;
    st.out = fopen(out_path, "wb");
    if (!st.out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    st.weights = {"weights", weights_env ? weights_env : "ffn_moe_weights_scaled", STREAM_WEIGHTS, DTYPE_F32};
    st.probs   = {"probs",   kProbs, STREAM_PROBS, DTYPE_F16};
    st.norms   = {"norms",   kDown,  STREAM_NORMS, DTYPE_F32};
    st.bias    = {"bias",    "ffn_moe_probs_biased", STREAM_BIAS, DTYPE_F32};
    st.norm_threads = threads_env ? std::max(1, atoi(threads_env)) : 8;
    std::vector<sidecar *> aligned;
    for (sidecar * s : {&st.weights, &st.probs, &st.norms, &st.bias}) {
        std::string list = "," + streams + ",";
        if (list.find("," + std::string(s->key) + ",") == std::string::npos) {
            continue;
        }
        if (!open_sidecar(*s, sidecar_path(out_path, s->key))) {
            return 1;
        }
        LOG_INF("route dump: stream %s <- %s -> %s\n", s->key, s->graph_name.c_str(), sidecar_path(out_path, s->key).c_str());
        if (s->kind != STREAM_BIAS) {
            aligned.push_back(s);
        }
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = route_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const auto docs = split_docs(params.prompt);
    const int n_batch = params.n_batch;
    for (size_t d = 0; d < docs.size(); ++d) {
        std::vector<llama_token> tokens = common_tokenize(ctx, docs[d], add_bos, true);
        if ((int) tokens.size() > (int) llama_n_ctx(ctx)) {
            tokens.resize(llama_n_ctx(ctx));
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i = 0; i < tokens.size(); i += n_batch) {
            const int n = (int) std::min<size_t>(n_batch, tokens.size() - i);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) || st.bad) {
                LOG_ERR("decode failed in doc %zu\n", d);
                return 1;
            }
        }
        const int32_t mark[3] = {-1, (int32_t) tokens.size(), 0};
        fwrite(mark, sizeof(int32_t), 3, st.out);
        fflush(st.out);
        for (sidecar * s : aligned) {
            fwrite(mark, sizeof(int32_t), 3, s->f);
            fflush(s->f);
            if (s->records != st.records || s->seq_hash != st.seq_hash) {
                LOG_ERR("stream %s (%s) is not aligned with the ids after doc %zu: %lld vs %lld records\n", s->key,
                        s->graph_name.c_str(), d + 1, s->records, st.records);
                return 1;
            }
        }
        if (st.bias.f) {
            fflush(st.bias.f);
        }
        LOG_INF("doc %zu/%zu: %zu tokens\n", d + 1, docs.size(), tokens.size());
    }
    fclose(st.out);
    LOG_INF("route dump: ids %lld records, %.1f s in the callback\n", st.records, st.seconds);
    for (sidecar * s : {&st.weights, &st.probs, &st.norms, &st.bias}) {
        if (s->f) {
            fclose(s->f);
            LOG_INF("route dump: %s %lld records, %.1f s in the callback\n", s->key, s->records, s->seconds);
        }
    }
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
