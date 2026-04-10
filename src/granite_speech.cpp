// granite_speech.cpp — ibm-granite/granite-4.0-1b-speech ggml runtime
//
// Three-module speech-LLM:
//   1. 16-layer Conformer encoder (Macaron FFN, depthwise conv, rel pos emb)
//   2. 2-layer BLIP-2 Q-Former projector (3 learned query tokens → 3 LLM tokens)
//   3. 40-layer Granite 1B LLM (GQA 16/4, μP multipliers, RoPE)

#include "granite_speech.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
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
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct granite_speech_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;

    // Encoder
    uint32_t enc_n_layers = 16;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_input_dim = 160;
    uint32_t enc_conv_kernel = 15;
    uint32_t enc_ff_dim = 4096;

    // Projector (Q-Former)
    uint32_t proj_n_layers = 2;
    uint32_t proj_d_model = 1024;
    uint32_t proj_n_heads = 16;
    uint32_t proj_ff_dim = 4096;

    // LLM
    uint32_t llm_n_layers = 40;
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 4;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 4096;
    float    llm_rope_theta = 10000.0f;
    float    llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 100353;

    // μP multipliers
    float embedding_multiplier = 12.0f;
    float attention_multiplier = 0.0078125f;  // 1/128
    float residual_multiplier = 0.22f;
    float logits_scaling = 8.0f;

    uint32_t downsample_rate = 5;
    uint32_t window_size = 15;
    uint32_t audio_token_index = 100352;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct granite_enc_block {
    // Attention
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_kv_w = nullptr;  // combined K+V: (2*head_dim*n_heads, d_model)
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_out_b = nullptr;
    ggml_tensor * attn_rel_pos_w = nullptr;  // (max_pos*2+1, head_dim)

    // Conv module
    ggml_tensor * conv_up_w = nullptr;
    ggml_tensor * conv_up_b = nullptr;
    ggml_tensor * conv_dw_w = nullptr;  // depthwise: (2*d_model, 1, kernel)
    ggml_tensor * conv_bn_w = nullptr;
    ggml_tensor * conv_bn_b = nullptr;
    ggml_tensor * conv_bn_mean = nullptr;
    ggml_tensor * conv_bn_var = nullptr;
    ggml_tensor * conv_down_w = nullptr;
    ggml_tensor * conv_down_b = nullptr;
    ggml_tensor * conv_norm_w = nullptr;
    ggml_tensor * conv_norm_b = nullptr;

    // FFN1 (Macaron pre)
    ggml_tensor * ff1_norm_w = nullptr;
    ggml_tensor * ff1_norm_b = nullptr;
    ggml_tensor * ff1_up_w = nullptr;
    ggml_tensor * ff1_up_b = nullptr;
    ggml_tensor * ff1_down_w = nullptr;
    ggml_tensor * ff1_down_b = nullptr;

    // FFN2 (Macaron post)
    ggml_tensor * ff2_norm_w = nullptr;
    ggml_tensor * ff2_norm_b = nullptr;
    ggml_tensor * ff2_up_w = nullptr;
    ggml_tensor * ff2_up_b = nullptr;
    ggml_tensor * ff2_down_w = nullptr;
    ggml_tensor * ff2_down_b = nullptr;

    // Post-norm
    ggml_tensor * post_norm_w = nullptr;
    ggml_tensor * post_norm_b = nullptr;
};

struct granite_proj_block {
    // Self-attention
    ggml_tensor * sa_q_w = nullptr, * sa_q_b = nullptr;
    ggml_tensor * sa_k_w = nullptr, * sa_k_b = nullptr;
    ggml_tensor * sa_v_w = nullptr, * sa_v_b = nullptr;
    ggml_tensor * sa_out_w = nullptr, * sa_out_b = nullptr;
    ggml_tensor * sa_norm_w = nullptr, * sa_norm_b = nullptr;

    // Cross-attention
    ggml_tensor * ca_q_w = nullptr, * ca_q_b = nullptr;
    ggml_tensor * ca_k_w = nullptr, * ca_k_b = nullptr;
    ggml_tensor * ca_v_w = nullptr, * ca_v_b = nullptr;
    ggml_tensor * ca_out_w = nullptr, * ca_out_b = nullptr;
    ggml_tensor * ca_norm_w = nullptr, * ca_norm_b = nullptr;

    // FFN
    ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
};

struct granite_llm_block {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
};

struct granite_speech_model {
    granite_speech_hparams hparams;

    struct {
        ggml_tensor * input_w = nullptr;
        ggml_tensor * input_b = nullptr;
        ggml_tensor * mel_filters = nullptr;
        ggml_tensor * mel_window = nullptr;
        std::vector<granite_enc_block> blocks;
    } encoder;

    struct {
        ggml_tensor * query = nullptr;       // (1, n_query, d_model)
        ggml_tensor * ln_w = nullptr;
        ggml_tensor * ln_b = nullptr;
        ggml_tensor * linear_w = nullptr;    // (llm_d, proj_d)
        ggml_tensor * linear_b = nullptr;
        std::vector<granite_proj_block> blocks;
    } projector;

    struct {
        ggml_tensor * token_embd_w = nullptr;
        ggml_tensor * output_norm_w = nullptr;
        ggml_tensor * output_w = nullptr;   // separate lm_head (not tied)
        std::vector<granite_llm_block> blocks;
    } llm;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

struct granite_speech_context {
    granite_speech_context_params params;
    granite_speech_model model;

    ggml_backend_t       backend = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context *        kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor *         kv_k = nullptr;
    ggml_tensor *         kv_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// GGUF loader helpers
// ===========================================================================

static uint32_t kv_u32(gguf_context * g, const char * key, uint32_t def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? (uint32_t)gguf_get_val_u32(g, i) : def;
}
static float kv_f32(gguf_context * g, const char * key, float def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? gguf_get_val_f32(g, i) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool granite_speech_load_model(granite_speech_model & model, const char * path,
                                      ggml_backend_t backend) {
    // Pass 1: metadata
    {
        gguf_init_params mp = { true, nullptr };
        gguf_context * g = gguf_init_from_file(path, mp);
        if (!g) return false;
        auto & hp = model.hparams;

        hp.enc_n_layers = kv_u32(g, "granite_speech.enc.n_layers", hp.enc_n_layers);
        hp.enc_d_model  = kv_u32(g, "granite_speech.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads  = kv_u32(g, "granite_speech.enc.n_heads", hp.enc_n_heads);
        hp.enc_head_dim = kv_u32(g, "granite_speech.enc.head_dim", hp.enc_head_dim);
        hp.enc_input_dim = kv_u32(g, "granite_speech.enc.input_dim", hp.enc_input_dim);
        hp.enc_conv_kernel = kv_u32(g, "granite_speech.enc.conv_kernel", hp.enc_conv_kernel);
        hp.enc_ff_dim   = kv_u32(g, "granite_speech.enc.ff_dim", hp.enc_ff_dim);

        hp.proj_n_layers = kv_u32(g, "granite_speech.proj.n_layers", hp.proj_n_layers);
        hp.proj_d_model  = kv_u32(g, "granite_speech.proj.d_model", hp.proj_d_model);
        hp.proj_n_heads  = kv_u32(g, "granite_speech.proj.n_heads", hp.proj_n_heads);
        hp.proj_ff_dim   = kv_u32(g, "granite_speech.proj.ff_dim", hp.proj_ff_dim);

        hp.llm_n_layers   = kv_u32(g, "granite_speech.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model    = kv_u32(g, "granite_speech.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads    = kv_u32(g, "granite_speech.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = kv_u32(g, "granite_speech.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim   = kv_u32(g, "granite_speech.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim     = kv_u32(g, "granite_speech.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = kv_f32(g, "granite_speech.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps    = kv_f32(g, "granite_speech.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = kv_u32(g, "granite_speech.llm.vocab_size", hp.llm_vocab_size);

        hp.embedding_multiplier = kv_f32(g, "granite_speech.llm.embedding_multiplier", hp.embedding_multiplier);
        hp.attention_multiplier = kv_f32(g, "granite_speech.llm.attention_multiplier", hp.attention_multiplier);
        hp.residual_multiplier  = kv_f32(g, "granite_speech.llm.residual_multiplier", hp.residual_multiplier);
        hp.logits_scaling       = kv_f32(g, "granite_speech.llm.logits_scaling", hp.logits_scaling);

        hp.downsample_rate    = kv_u32(g, "granite_speech.downsample_rate", hp.downsample_rate);
        hp.window_size        = kv_u32(g, "granite_speech.window_size", hp.window_size);
        hp.audio_token_index  = kv_u32(g, "granite_speech.audio_token_index", hp.audio_token_index);

        gguf_free(g);
    }

    // Pass 2: load tensors
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params lp = { true, &weight_ctx };
        gguf_context * g = gguf_init_from_file(path, lp);
        if (!g || !weight_ctx) return false;

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        void * mmap_base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);

        int n_tensors = gguf_get_n_tensors(g);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(weight_ctx, name);
            if (!t) continue;
            size_t offset = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
            ggml_backend_tensor_set(t, (const char *)mmap_base + offset, 0, ggml_nbytes(t));
            model.tensors[name] = t;
        }
        munmap(mmap_base, (size_t)st.st_size);
        gguf_free(g);
    }
    model.ctx = weight_ctx;

    // Bind tensors
    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = model.tensors.find(n);
        return it != model.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string & n) -> ggml_tensor * {
        auto * t = get(n);
        if (!t) fprintf(stderr, "granite_speech: missing '%s'\n", n.c_str());
        return t;
    };

    // Encoder
    auto & e = model.encoder;
    e.input_w = require("enc.input.weight");
    e.input_b = require("enc.input.bias");
    e.mel_filters = get("audio.mel_filters");
    e.mel_window = get("audio.mel_window");

    e.blocks.resize(model.hparams.enc_n_layers);
    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto p = "enc.blk." + std::to_string(il) + ".";
        auto & b = e.blocks[il];
        b.attn_norm_w = get(p + "attn_norm.weight");
        b.attn_norm_b = get(p + "attn_norm.bias");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_kv_w = require(p + "attn_kv.weight");
        b.attn_out_w = require(p + "attn_out.weight");
        b.attn_out_b = get(p + "attn_out.bias");
        b.attn_rel_pos_w = get(p + "attn_rel_pos.weight");

        b.conv_up_w = get(p + "conv_up.weight");
        b.conv_up_b = get(p + "conv_up.bias");
        b.conv_dw_w = get(p + "conv_dw.weight");
        b.conv_bn_w = get(p + "conv_bn.weight");
        b.conv_bn_b = get(p + "conv_bn.bias");
        b.conv_bn_mean = get(p + "conv_bn.running_mean");
        b.conv_bn_var = get(p + "conv_bn.running_var");
        b.conv_down_w = get(p + "conv_down.weight");
        b.conv_down_b = get(p + "conv_down.bias");
        b.conv_norm_w = get(p + "conv_norm.weight");
        b.conv_norm_b = get(p + "conv_norm.bias");

        b.ff1_norm_w = get(p + "ff1_norm.weight");
        b.ff1_norm_b = get(p + "ff1_norm.bias");
        b.ff1_up_w = require(p + "ff1_up.weight");
        b.ff1_up_b = get(p + "ff1_up.bias");
        b.ff1_down_w = require(p + "ff1_down.weight");
        b.ff1_down_b = get(p + "ff1_down.bias");

        b.ff2_norm_w = get(p + "ff2_norm.weight");
        b.ff2_norm_b = get(p + "ff2_norm.bias");
        b.ff2_up_w = require(p + "ff2_up.weight");
        b.ff2_up_b = get(p + "ff2_up.bias");
        b.ff2_down_w = require(p + "ff2_down.weight");
        b.ff2_down_b = get(p + "ff2_down.bias");

        b.post_norm_w = get(p + "post_norm.weight");
        b.post_norm_b = get(p + "post_norm.bias");
    }

    // Projector
    auto & pr = model.projector;
    pr.query = require("proj.query");
    pr.ln_w = get("proj.ln.weight");
    pr.ln_b = get("proj.ln.bias");
    pr.linear_w = require("proj.linear.weight");
    pr.linear_b = get("proj.linear.bias");

    pr.blocks.resize(model.hparams.proj_n_layers);
    for (uint32_t il = 0; il < model.hparams.proj_n_layers; il++) {
        auto p = "proj.blk." + std::to_string(il) + ".";
        auto & b = pr.blocks[il];
        b.sa_q_w = require(p + "sa_query.weight"); b.sa_q_b = get(p + "sa_query.bias");
        b.sa_k_w = require(p + "sa_key.weight"); b.sa_k_b = get(p + "sa_key.bias");
        b.sa_v_w = require(p + "sa_value.weight"); b.sa_v_b = get(p + "sa_value.bias");
        b.sa_out_w = require(p + "sa_out.weight"); b.sa_out_b = get(p + "sa_out.bias");
        b.sa_norm_w = get(p + "sa_norm.weight"); b.sa_norm_b = get(p + "sa_norm.bias");

        b.ca_q_w = require(p + "ca_query.weight"); b.ca_q_b = get(p + "ca_query.bias");
        b.ca_k_w = require(p + "ca_key.weight"); b.ca_k_b = get(p + "ca_key.bias");
        b.ca_v_w = require(p + "ca_value.weight"); b.ca_v_b = get(p + "ca_value.bias");
        b.ca_out_w = require(p + "ca_out.weight"); b.ca_out_b = get(p + "ca_out.bias");
        b.ca_norm_w = get(p + "ca_norm.weight"); b.ca_norm_b = get(p + "ca_norm.bias");

        b.ffn_up_w = require(p + "ffn_up.weight"); b.ffn_up_b = get(p + "ffn_up.bias");
        b.ffn_down_w = require(p + "ffn_down.weight"); b.ffn_down_b = get(p + "ffn_down.bias");
        b.ffn_norm_w = get(p + "ffn_norm.weight"); b.ffn_norm_b = get(p + "ffn_norm.bias");
    }

    // LLM
    auto & l = model.llm;
    l.token_embd_w = require("token_embd.weight");
    l.output_norm_w = require("output_norm.weight");
    l.output_w = require("output.weight");

    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto p = "blk." + std::to_string(il) + ".";
        auto & b = l.blocks[il];
        b.attn_norm_w = require(p + "attn_norm.weight");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_k_w = require(p + "attn_k.weight");
        b.attn_v_w = require(p + "attn_v.weight");
        b.attn_out_w = require(p + "attn_output.weight");
        b.ffn_norm_w = require(p + "ffn_norm.weight");
        b.ffn_gate_w = require(p + "ffn_gate.weight");
        b.ffn_up_w = require(p + "ffn_up.weight");
        b.ffn_down_w = require(p + "ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// Public API (stubs — to be implemented)
// ===========================================================================

extern "C" struct granite_speech_context_params granite_speech_context_default_params(void) {
    return { /*n_threads=*/4, /*verbosity=*/1 };
}

extern "C" struct granite_speech_context * granite_speech_init_from_file(
    const char * path, struct granite_speech_context_params params)
{
    auto * ctx = new granite_speech_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!granite_speech_load_model(ctx->model, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Create scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        const auto & hp = ctx->model.hparams;
        fprintf(stderr, "granite_speech: loaded %s (enc %u layers, proj %u layers, llm %u layers, vocab %u)\n",
                path, hp.enc_n_layers, hp.proj_n_layers, hp.llm_n_layers, hp.llm_vocab_size);
    }
    return ctx;
}

extern "C" void granite_speech_free(struct granite_speech_context * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// TODO: Implement compute_mel, run_encoder, run_projector, run_llm_kv, embed_tokens, transcribe
// These follow the same patterns as voxtral4b.cpp but with:
//   - Conformer encoder (Macaron FFN + depthwise conv + batch norm + rel pos emb)
//   - Q-Former projector (self-attn + cross-attn + FFN per layer)
//   - Granite LLM (standard Llama-style but with μP multiplier scalars)

extern "C" float * granite_speech_compute_mel(struct granite_speech_context *, const float *, int, int *, int *) { return nullptr; }
extern "C" float * granite_speech_run_encoder(struct granite_speech_context *, const float *, int, int, int *, int *) { return nullptr; }
extern "C" float * granite_speech_run_projector(struct granite_speech_context *, const float *, int, int, int *, int *) { return nullptr; }
extern "C" bool granite_speech_kv_init(struct granite_speech_context *, int) { return false; }
extern "C" void granite_speech_kv_reset(struct granite_speech_context *) {}
extern "C" float * granite_speech_run_llm_kv(struct granite_speech_context *, const float *, int, int, int *, int *) { return nullptr; }
extern "C" float * granite_speech_embed_tokens(struct granite_speech_context *, const int32_t *, int) { return nullptr; }
extern "C" char * granite_speech_transcribe(struct granite_speech_context *, const float *, int) { return nullptr; }
