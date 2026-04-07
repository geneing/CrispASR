// canary.cpp — nvidia/canary-1b-v2 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer with biases), Transformer decoder
// (self-attn + cross-attn + FFN with KV cache), task-token prompt and
// greedy decode loop will land in subsequent commits.
//
// Architecture:
//   Mel:           128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann)
//   Encoder:       32× FastConformer block (use_bias=True), d_model=1024,
//                  8 heads, head_dim=128, ff_dim=4096, conv kernel=9,
//                  8× temporal subsampling via dw_striding
//   Decoder:       8× pre-LN Transformer block (SA + CA + FFN),
//                  d_model=1024, 8 heads, head_dim=128, ff_dim=4096
//   Embedding:     token (16384, 1024) + learned pos_enc (1024, 1024) + LN
//   Output head:   linear (1024 → 16384)
//
// Decoder prompt format (mirrors Cohere — same vocab):
//   <|startoftranscript|> <|src|> <|tgt|> <|pnc|> <|notimestamp|> <|nodiarize|> ...

#include "canary.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct canary_hparams {
    uint32_t sample_rate          = 16000;
    uint32_t n_mels               = 128;
    uint32_t n_fft                = 512;
    uint32_t win_length           = 400;
    uint32_t hop_length           = 160;
    uint32_t d_model              = 1024;
    uint32_t enc_n_layers         = 32;
    uint32_t dec_n_layers         = 8;
    uint32_t n_heads              = 8;
    uint32_t head_dim             = 128;
    uint32_t ff_dim               = 4096;
    uint32_t subsampling_factor   = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel          = 9;
    uint32_t vocab_size           = 16384;
    uint32_t max_dec_ctx          = 1024;
    uint32_t frame_dur_cs         = 8;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct canary_pre_encode {
    ggml_tensor * conv0_w = nullptr, * conv0_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    ggml_tensor * conv3_w = nullptr, * conv3_b = nullptr;
    ggml_tensor * conv5_w = nullptr, * conv5_b = nullptr;
    ggml_tensor * conv6_w = nullptr, * conv6_b = nullptr;
    ggml_tensor * out_w   = nullptr, * out_b   = nullptr;
};

struct canary_enc_layer {
    ggml_tensor * norm_ff1_w = nullptr, * norm_ff1_b = nullptr;
    ggml_tensor * ff1_l1_w   = nullptr, * ff1_l1_b   = nullptr;
    ggml_tensor * ff1_l2_w   = nullptr, * ff1_l2_b   = nullptr;

    ggml_tensor * norm_attn_w = nullptr, * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr, * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr, * attn_k_b    = nullptr;
    ggml_tensor * attn_v_w    = nullptr, * attn_v_b    = nullptr;
    ggml_tensor * attn_out_w  = nullptr, * attn_out_b  = nullptr;
    ggml_tensor * attn_pos_w  = nullptr;     // no bias on the rel-pos projection
    ggml_tensor * pos_bias_u  = nullptr;
    ggml_tensor * pos_bias_v  = nullptr;

    ggml_tensor * norm_conv_w = nullptr, * norm_conv_b = nullptr;
    ggml_tensor * conv_pw1_w  = nullptr, * conv_pw1_b  = nullptr;
    ggml_tensor * conv_dw_w   = nullptr, * conv_dw_b   = nullptr;
    ggml_tensor * conv_bn_w   = nullptr, * conv_bn_b   = nullptr;
    ggml_tensor * conv_bn_rm  = nullptr, * conv_bn_rv  = nullptr;
    ggml_tensor * conv_pw2_w  = nullptr, * conv_pw2_b  = nullptr;

    ggml_tensor * norm_ff2_w = nullptr, * norm_ff2_b = nullptr;
    ggml_tensor * ff2_l1_w   = nullptr, * ff2_l1_b   = nullptr;
    ggml_tensor * ff2_l2_w   = nullptr, * ff2_l2_b   = nullptr;

    ggml_tensor * norm_out_w = nullptr, * norm_out_b = nullptr;
};

struct canary_dec_layer {
    // Pre-LN block layout:
    //   x = x + sa_out @ SA(norm_sa(x))
    //   x = x + ca_out @ CA(norm_ca(x), enc_kv)
    //   x = x + ff_out @ activation(ff_in @ norm_ff(x))
    ggml_tensor * norm_sa_w = nullptr, * norm_sa_b = nullptr;
    ggml_tensor * sa_q_w    = nullptr, * sa_q_b    = nullptr;
    ggml_tensor * sa_k_w    = nullptr, * sa_k_b    = nullptr;
    ggml_tensor * sa_v_w    = nullptr, * sa_v_b    = nullptr;
    ggml_tensor * sa_out_w  = nullptr, * sa_out_b  = nullptr;

    ggml_tensor * norm_ca_w = nullptr, * norm_ca_b = nullptr;
    ggml_tensor * ca_q_w    = nullptr, * ca_q_b    = nullptr;
    ggml_tensor * ca_k_w    = nullptr, * ca_k_b    = nullptr;
    ggml_tensor * ca_v_w    = nullptr, * ca_v_b    = nullptr;
    ggml_tensor * ca_out_w  = nullptr, * ca_out_b  = nullptr;

    ggml_tensor * norm_ff_w = nullptr, * norm_ff_b = nullptr;
    ggml_tensor * ff_in_w   = nullptr, * ff_in_b   = nullptr;
    ggml_tensor * ff_out_w  = nullptr, * ff_out_b  = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct canary_model {
    canary_hparams hparams;

    ggml_tensor * mel_fb     = nullptr;
    ggml_tensor * mel_window = nullptr;

    canary_pre_encode               pre_encode;
    std::vector<canary_enc_layer>   enc;
    std::vector<canary_dec_layer>   dec;

    // Decoder embeddings + final norm + output head
    ggml_tensor * dec_embed_w     = nullptr;   // (vocab, d_model)
    ggml_tensor * dec_pos_enc     = nullptr;   // (max_ctx, d_model) — learned
    ggml_tensor * dec_embed_ln_w  = nullptr;
    ggml_tensor * dec_embed_ln_b  = nullptr;
    ggml_tensor * dec_final_ln_w  = nullptr;
    ggml_tensor * dec_final_ln_b  = nullptr;
    ggml_tensor * dec_head_w      = nullptr;   // (vocab, d_model)
    ggml_tensor * dec_head_b      = nullptr;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor *> tensors;
};

struct canary_vocab {
    std::vector<std::string>            id_to_token;
    std::unordered_map<std::string,int> token_to_id;
};

struct canary_context {
    canary_context_params params;

    canary_model model;
    canary_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

static ggml_tensor * try_get(canary_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static ggml_tensor * require(canary_model & m, const char * name) {
    auto t = try_get(m, name);
    if (!t) fprintf(stderr, "canary: required tensor '%s' not found\n", name);
    return t;
}

static uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t def = 0) {
    int ki = gguf_find_key(gctx, key);
    return ki >= 0 ? (uint32_t)gguf_get_val_u32(gctx, ki) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool canary_load_model(canary_model & model,
                              canary_vocab  & vocab,
                              const char    * path,
                              ggml_backend_t  backend) {
    // ---- pass 1: hparams + vocab ----
    {
        ggml_init_params meta_params = {
            /*mem_size=*/   4 * 1024 * 1024,
            /*mem_buffer=*/ nullptr,
            /*no_alloc=*/   true,
        };
        ggml_context * meta_ctx = ggml_init(meta_params);
        gguf_init_params load_params_meta = { /*no_alloc=*/true, /*ctx=*/&meta_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params_meta);
        if (!gctx) {
            fprintf(stderr, "canary: failed to open '%s'\n", path);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }

        auto & hp = model.hparams;
        hp.sample_rate          = kv_u32(gctx, "canary.sample_rate",          hp.sample_rate);
        hp.n_mels               = kv_u32(gctx, "canary.n_mels",               hp.n_mels);
        hp.n_fft                = kv_u32(gctx, "canary.n_fft",                hp.n_fft);
        hp.win_length           = kv_u32(gctx, "canary.win_length",           hp.win_length);
        hp.hop_length           = kv_u32(gctx, "canary.hop_length",           hp.hop_length);
        hp.d_model              = kv_u32(gctx, "canary.d_model",              hp.d_model);
        hp.enc_n_layers         = kv_u32(gctx, "canary.enc_n_layers",         hp.enc_n_layers);
        hp.dec_n_layers         = kv_u32(gctx, "canary.dec_n_layers",         hp.dec_n_layers);
        hp.n_heads              = kv_u32(gctx, "canary.n_heads",              hp.n_heads);
        hp.head_dim             = kv_u32(gctx, "canary.head_dim",             hp.head_dim);
        hp.ff_dim               = kv_u32(gctx, "canary.ff_dim",               hp.ff_dim);
        hp.subsampling_factor   = kv_u32(gctx, "canary.subsampling_factor",   hp.subsampling_factor);
        hp.subsampling_channels = kv_u32(gctx, "canary.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel          = kv_u32(gctx, "canary.conv_kernel",          hp.conv_kernel);
        hp.vocab_size           = kv_u32(gctx, "canary.vocab_size",           hp.vocab_size);
        hp.max_dec_ctx          = kv_u32(gctx, "canary.max_dec_ctx",          hp.max_dec_ctx);
        hp.frame_dur_cs         = kv_u32(gctx, "canary.frame_dur_cs",         hp.frame_dur_cs);

        int ki = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (ki >= 0) {
            int n = gguf_get_arr_n(gctx, ki);
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gctx, ki, i);
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        gguf_free(gctx);
        ggml_free(meta_ctx);
    }

    // ---- pass 2: tensor metadata + mmap into backend buffer ----
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params load_params = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params);
        if (!gctx || !weight_ctx) {
            fprintf(stderr, "canary: failed to load tensor metadata\n");
            return false;
        }

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "canary: open failed\n"); return false; }
        struct stat st; fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "canary: mmap failed\n");
            return false;
        }

        size_t data_offset = gguf_get_data_offset(gctx);

        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;

            int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            size_t off    = gguf_get_tensor_offset(gctx, tid);
            size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char *)mmap_base + data_offset + off, 0, nbytes);
        }

        munmap(mmap_base, file_size);
        model.ctx = weight_ctx;
        gguf_free(gctx);
    }

    // ---- bind named tensors ----

    // Mel preprocessor
    model.mel_fb     = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode
    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = require(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = require(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = require(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = require(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w   = require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b   = require(model, "encoder.pre.out.bias");

    // Encoder layers
    model.enc.resize(model.hparams.enc_n_layers);
    for (uint32_t i = 0; i < model.hparams.enc_n_layers; i++) {
        char buf[128];
        auto & e = model.enc[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight"); e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w   = get("ff1.linear1.weight"); e.ff1_l1_b = get("ff1.linear1.bias");
        e.ff1_l2_w   = get("ff1.linear2.weight"); e.ff1_l2_b = get("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight"); e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w    = get("attn.q.weight");    e.attn_q_b    = get("attn.q.bias");
        e.attn_k_w    = get("attn.k.weight");    e.attn_k_b    = get("attn.k.bias");
        e.attn_v_w    = get("attn.v.weight");    e.attn_v_b    = get("attn.v.bias");
        e.attn_out_w  = get("attn.out.weight");  e.attn_out_b  = get("attn.out.bias");
        e.attn_pos_w  = get("attn.pos.weight");
        e.pos_bias_u  = get("attn.pos_bias_u");
        e.pos_bias_v  = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight"); e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w  = get("conv.pw1.weight");  e.conv_pw1_b  = get("conv.pw1.bias");
        e.conv_dw_w   = get("conv.dw.weight");   e.conv_dw_b   = get("conv.dw.bias");
        e.conv_pw2_w  = get("conv.pw2.weight");  e.conv_pw2_b  = get("conv.pw2.bias");
        e.conv_bn_w   = get("conv.bn.weight");   e.conv_bn_b   = get("conv.bn.bias");
        e.conv_bn_rm  = get("conv.bn.running_mean");
        e.conv_bn_rv  = get("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight"); e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w   = get("ff2.linear1.weight"); e.ff2_l1_b = get("ff2.linear1.bias");
        e.ff2_l2_w   = get("ff2.linear2.weight"); e.ff2_l2_b = get("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight"); e.norm_out_b = get("norm_out.bias");
    }

    // Decoder
    model.dec.resize(model.hparams.dec_n_layers);
    for (uint32_t i = 0; i < model.hparams.dec_n_layers; i++) {
        char buf[128];
        auto & d = model.dec[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "decoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        d.norm_sa_w = get("norm_sa.weight"); d.norm_sa_b = get("norm_sa.bias");
        d.sa_q_w    = get("sa_q.weight");    d.sa_q_b    = get("sa_q.bias");
        d.sa_k_w    = get("sa_k.weight");    d.sa_k_b    = get("sa_k.bias");
        d.sa_v_w    = get("sa_v.weight");    d.sa_v_b    = get("sa_v.bias");
        d.sa_out_w  = get("sa_out.weight");  d.sa_out_b  = get("sa_out.bias");

        d.norm_ca_w = get("norm_ca.weight"); d.norm_ca_b = get("norm_ca.bias");
        d.ca_q_w    = get("ca_q.weight");    d.ca_q_b    = get("ca_q.bias");
        d.ca_k_w    = get("ca_k.weight");    d.ca_k_b    = get("ca_k.bias");
        d.ca_v_w    = get("ca_v.weight");    d.ca_v_b    = get("ca_v.bias");
        d.ca_out_w  = get("ca_out.weight");  d.ca_out_b  = get("ca_out.bias");

        d.norm_ff_w = get("norm_ff.weight"); d.norm_ff_b = get("norm_ff.bias");
        d.ff_in_w   = get("ff_in.weight");   d.ff_in_b   = get("ff_in.bias");
        d.ff_out_w  = get("ff_out.weight");  d.ff_out_b  = get("ff_out.bias");
    }

    // Decoder embeddings + output head
    model.dec_embed_w    = require(model, "decoder.embed.weight");
    model.dec_pos_enc    = require(model, "decoder.pos_enc");
    model.dec_embed_ln_w = require(model, "decoder.embed_ln.weight");
    model.dec_embed_ln_b = require(model, "decoder.embed_ln.bias");
    model.dec_final_ln_w = require(model, "decoder.final_norm.weight");
    model.dec_final_ln_b = require(model, "decoder.final_norm.bias");
    model.dec_head_w     = require(model, "decoder.head.weight");
    model.dec_head_b     = require(model, "decoder.head.bias");

    fprintf(stderr,
        "canary: vocab=%u  d_model=%u  enc_layers=%u  dec_layers=%u  heads=%u  ff=%u  max_ctx=%u\n",
        model.hparams.vocab_size, model.hparams.d_model,
        model.hparams.enc_n_layers, model.hparams.dec_n_layers,
        model.hparams.n_heads, model.hparams.ff_dim, model.hparams.max_dec_ctx);
    return true;
}

// ===========================================================================
// BatchNorm folding (load-time, once) — same trick as parakeet/cohere
// ===========================================================================

static void canary_fold_batchnorm(canary_model & model) {
    const int d   = (int)model.hparams.d_model;
    const int K   = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto & e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b ||
            !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) continue;

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d), dw_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w,  bn_w   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b,  bn_b   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_dw_b,  dw_b   .data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++) s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        std::vector<ggml_fp16_t> w_f16((size_t)K * d);
        ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        std::vector<float> w_f32((size_t)K * d);
        for (size_t i = 0; i < w_f16.size(); i++) w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
        for (int c = 0; c < d; c++)
            for (int ki = 0; ki < K; ki++)
                w_f32[ki + c * K] *= s[c];
        for (size_t i = 0; i < w_f16.size(); i++) w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
        ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));

        // Fold into existing dw_b: b'[c] = (dw_b[c] - mean[c]) * s[c] + bn_b[c]
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "canary: BN folded into conv_dw weights for %u layers\n",
            model.hparams.enc_n_layers);
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
#ifdef GGML_USE_METAL
    if (ggml_backend_t b = ggml_backend_metal_init()) return b;
#endif
#ifdef GGML_USE_CUDA
    if (ggml_backend_t b = ggml_backend_cuda_init(0))  return b;
#endif
    return ggml_backend_cpu_init();
}

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" struct canary_context_params canary_context_default_params(void) {
    canary_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    return p;
}

extern "C" struct canary_context * canary_init_from_file(
    const char * path_model, struct canary_context_params params)
{
    auto * ctx = new canary_context();
    ctx->params    = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend     = pick_backend();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend) ctx->backend = ctx->backend_cpu;

    if (!canary_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        canary_free(ctx);
        return nullptr;
    }
    canary_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void canary_free(struct canary_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)             ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)         ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)         ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)       ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void canary_result_free(struct canary_result * r) {
    if (!r) return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" int canary_n_vocab    (struct canary_context * ctx) { return (int)ctx->model.hparams.vocab_size; }
extern "C" int canary_n_mels     (struct canary_context * ctx) { return (int)ctx->model.hparams.n_mels; }
extern "C" int canary_sample_rate(struct canary_context * ctx) { return (int)ctx->model.hparams.sample_rate; }
extern "C" int canary_frame_dur_cs(struct canary_context * ctx){ return (int)ctx->model.hparams.frame_dur_cs; }

extern "C" const char * canary_token_to_str(struct canary_context * ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "";
    return ctx->vocab.id_to_token[id].c_str();
}

extern "C" int canary_str_to_token(struct canary_context * ctx, const char * str) {
    auto it = ctx->vocab.token_to_id.find(str);
    return it != ctx->vocab.token_to_id.end() ? it->second : -1;
}

extern "C" int canary_test_load(struct canary_context * ctx) {
    fprintf(stderr,
        "canary: load test OK\n"
        "  vocab_size  = %d\n"
        "  d_model     = %d\n"
        "  enc_layers  = %d\n"
        "  dec_layers  = %d\n"
        "  n_heads     = %d\n"
        "  head_dim    = %d\n"
        "  ff_dim      = %d\n"
        "  max_dec_ctx = %d\n"
        "  n_mels      = %d\n"
        "  sample_rate = %d\n"
        "  frame_dur_cs= %d\n",
        (int)ctx->model.hparams.vocab_size,
        (int)ctx->model.hparams.d_model,
        (int)ctx->model.hparams.enc_n_layers,
        (int)ctx->model.hparams.dec_n_layers,
        (int)ctx->model.hparams.n_heads,
        (int)ctx->model.hparams.head_dim,
        (int)ctx->model.hparams.ff_dim,
        (int)ctx->model.hparams.max_dec_ctx,
        (int)ctx->model.hparams.n_mels,
        (int)ctx->model.hparams.sample_rate,
        (int)ctx->model.hparams.frame_dur_cs);

    // Confirm a few special tokens resolve
    const char * specials[] = {
        "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|de|>",
        "<|fr|>", "<|es|>", "<|nopnc|>", "<|notimestamp|>", "<|nodiarize|>",
    };
    for (const char * s : specials) {
        int id = canary_str_to_token(ctx, s);
        fprintf(stderr, "  token %-22s = %d\n", s, id);
    }
    return 0;
}

// Stubs (full implementation in subsequent commits).
extern "C" char * canary_transcribe(struct canary_context * /*ctx*/,
                                    const float * /*samples*/, int /*n_samples*/,
                                    const char * /*src*/, const char * /*tgt*/, bool /*pnc*/) {
    fprintf(stderr, "canary: transcribe() not yet implemented\n");
    return nullptr;
}

extern "C" struct canary_result * canary_transcribe_ex(
    struct canary_context * /*ctx*/, const float * /*samples*/, int /*n_samples*/,
    const char * /*src*/, const char * /*tgt*/, bool /*pnc*/, int64_t /*t_offset_cs*/) {
    fprintf(stderr, "canary: transcribe_ex() not yet implemented\n");
    return nullptr;
}
