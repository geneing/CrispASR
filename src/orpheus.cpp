// orpheus.cpp — Orpheus-3B (Llama-3.2-3B + SNAC 24 kHz) TTS backend.
//
// Slice (a) of PLAN #57 Phase 2 lands the foundation:
//   * GGUF arch="orpheus" loader (hparams + tokenizer + weight tensors)
//   * Fixed-speaker table (orpheus.spk_names) + by-name selection
//   * C ABI surface in orpheus.h
//
// The talker AR forward and the SNAC C++ decoder land in slices (b)
// and (c). For now orpheus_synthesize / orpheus_synthesize_codes
// return nullptr with a clear "not yet implemented" message; the
// foundation is enough to:
//   1. Verify the converter output loads cleanly.
//   2. Enumerate baked speakers from a converted GGUF.
//   3. Wire the registry + factory into the CLI without crashes.

#include "orpheus.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct orpheus_hp {
    // Llama-3.2-3B-Instruct talker
    uint32_t n_layers = 28;
    uint32_t d_model = 3072;
    uint32_t n_heads = 24;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    uint32_t vocab_size = 156938; // base 128256 + 7*4096 custom + handful of markers
    uint32_t max_pos = 131072;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;

    // Audio wrapper / codec slot tokens (literal IDs from
    // orpheus_tts_pypi/orpheus_tts/{engine_class.py,decoder.py}).
    uint32_t audio_start = 128259;
    uint32_t audio_pre_end = 128009;
    uint32_t audio_end_a = 128260;
    uint32_t audio_end_b = 128261;
    uint32_t audio_end = 128257;
    uint32_t custom_token_offset = 128266; // <custom_token_0> id
    uint32_t custom_token_count = 7 * 4096;

    // SNAC slot layout — 7 LM tokens / super-frame, codes per book = [1,2,4]
    uint32_t super_frame_slots = 7;
    uint32_t cb_count = 3;
    uint32_t cb_size = 4096;

    // Variant ("base" | "fixed_speaker"). Drives whether --voice is
    // required at synthesis time.
    std::string tts_model_type = "fixed_speaker";

    // Baked speakers — used as the literal `name: text` prompt prefix.
    std::vector<std::string> spk_names;
};

struct orpheus_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct orpheus_talker {
    ggml_tensor* token_embd_w = nullptr; // (d_model, vocab_size)
    std::vector<orpheus_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr; // lm_head (d_model, vocab_size)
};

struct orpheus_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

} // namespace

struct orpheus_context {
    orpheus_context_params params{};
    int n_threads = 4;

    orpheus_hp hp;
    orpheus_vocab vocab;
    orpheus_talker talker;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // SNAC codec: not yet loaded in C++ (slice b).
    std::string snac_codec_path;

    // Currently selected fixed speaker (literal `name:` prompt prefix).
    int active_speaker = -1;

    ~orpheus_context() {
        if (ctx_w) {
            ggml_free(ctx_w);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

static std::string lower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return r;
}

// Bind the talker tensors out of the freshly-loaded weight map.
static bool bind_talker(orpheus_context* c) {
    auto& t = c->talker;
    auto& tensors = c->tensors;

    t.token_embd_w = core_gguf::require(tensors, "talker.token_embd.weight", "orpheus");
    t.output_norm_w = core_gguf::require(tensors, "talker.output_norm.weight", "orpheus");
    // lm_head — Llama-3.2-3B does NOT tie input + output embeddings, so
    // we expect a separate talker.output.weight tensor.
    t.output_w = core_gguf::try_get(tensors, "talker.output.weight");
    if (!t.output_w) {
        // Some converters tie the embeddings instead — fall back to
        // token_embd.weight for the lm_head.
        t.output_w = t.token_embd_w;
    }

    if (!t.token_embd_w || !t.output_norm_w) {
        return false;
    }

    t.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = t.blocks[i];
        char key[64];
#define FMT(fld, sub) do { \
        std::snprintf(key, sizeof(key), "talker.blk.%u." sub ".weight", i); \
        b.fld = core_gguf::require(tensors, key, "orpheus"); \
    } while (0)
        FMT(attn_norm_w, "attn_norm");
        FMT(attn_q_w, "attn_q");
        FMT(attn_k_w, "attn_k");
        FMT(attn_v_w, "attn_v");
        FMT(attn_output_w, "attn_output");
        FMT(ffn_norm_w, "ffn_norm");
        FMT(ffn_gate_w, "ffn_gate");
        FMT(ffn_up_w, "ffn_up");
        FMT(ffn_down_w, "ffn_down");
#undef FMT
        if (!b.attn_norm_w || !b.attn_q_w || !b.attn_k_w || !b.attn_v_w || !b.attn_output_w || !b.ffn_norm_w ||
            !b.ffn_gate_w || !b.ffn_up_w || !b.ffn_down_w) {
            fprintf(stderr, "orpheus: missing tensor in layer %u\n", i);
            return false;
        }
    }
    return true;
}

static void load_metadata(orpheus_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers = core_gguf::kv_u32(g, "orpheus.talker.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(g, "orpheus.talker.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(g, "orpheus.talker.n_heads", hp.n_heads);
    hp.n_kv_heads = core_gguf::kv_u32(g, "orpheus.talker.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = core_gguf::kv_u32(g, "orpheus.talker.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(g, "orpheus.talker.ff_dim", hp.ff_dim);
    hp.vocab_size = core_gguf::kv_u32(g, "orpheus.talker.vocab_size", hp.vocab_size);
    hp.max_pos = core_gguf::kv_u32(g, "orpheus.talker.max_pos", hp.max_pos);
    hp.rope_theta = core_gguf::kv_f32(g, "orpheus.talker.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "orpheus.talker.rms_norm_eps", hp.rms_norm_eps);

    hp.audio_start = core_gguf::kv_u32(g, "orpheus.audio_start_token", hp.audio_start);
    hp.audio_pre_end = core_gguf::kv_u32(g, "orpheus.audio_pre_end_token", hp.audio_pre_end);
    hp.audio_end_a = core_gguf::kv_u32(g, "orpheus.audio_end_a_token", hp.audio_end_a);
    hp.audio_end_b = core_gguf::kv_u32(g, "orpheus.audio_end_b_token", hp.audio_end_b);
    hp.audio_end = core_gguf::kv_u32(g, "orpheus.audio_end_token", hp.audio_end);
    hp.custom_token_offset = core_gguf::kv_u32(g, "orpheus.custom_token_offset", hp.custom_token_offset);
    hp.custom_token_count = core_gguf::kv_u32(g, "orpheus.custom_token_count", hp.custom_token_count);

    hp.super_frame_slots = core_gguf::kv_u32(g, "orpheus.snac.super_frame_slots", hp.super_frame_slots);
    hp.cb_count = core_gguf::kv_u32(g, "orpheus.snac.codebook_count", hp.cb_count);
    hp.cb_size = core_gguf::kv_u32(g, "orpheus.snac.codebook_size", hp.cb_size);

    hp.tts_model_type = core_gguf::kv_str(g, "orpheus.tts_model_type", hp.tts_model_type.c_str());
    hp.spk_names = core_gguf::kv_str_array(g, "orpheus.spk_names");

    // Tokenizer (Llama-3.2 BPE in tokenizer.ggml.tokens / .merges).
    auto tok = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        c->vocab.id_to_token = std::move(tok);
        c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
        for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
            c->vocab.token_to_id[c->vocab.id_to_token[i]] = i;
        }
    }
    auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++) {
        c->vocab.merge_rank[merges[i]] = (int32_t)i;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------------

extern "C" struct orpheus_context_params orpheus_context_default_params(void) {
    orpheus_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f;
    p.max_audio_tokens = 0;
    return p;
}

extern "C" struct orpheus_context* orpheus_init_from_file(const char* path_model,
                                                          struct orpheus_context_params params) {
    auto* c = new orpheus_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: hparams + vocab.
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "orpheus: variant=%s  talker=%uL d=%u h=%u/%u hd=%u ff=%u vocab=%u\n"
                "orpheus: rope_theta=%.0f  custom_token_offset=%u count=%u  speakers=%zu\n",
                c->hp.tts_model_type.c_str(), c->hp.n_layers, c->hp.d_model, c->hp.n_heads, c->hp.n_kv_heads,
                c->hp.head_dim, c->hp.ff_dim, c->hp.vocab_size, (double)c->hp.rope_theta, c->hp.custom_token_offset,
                c->hp.custom_token_count, c->hp.spk_names.size());
    }

    // Backend selection.
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "orpheus: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "orpheus", wl)) {
        fprintf(stderr, "orpheus: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_talker(c)) {
        fprintf(stderr, "orpheus: tensor binding failed\n");
        delete c;
        return nullptr;
    }

    // Default to first baked speaker for fixed_speaker variants.
    if (c->hp.tts_model_type == "fixed_speaker" && !c->hp.spk_names.empty()) {
        c->active_speaker = 0;
    }
    return c;
}

extern "C" void orpheus_free(struct orpheus_context* ctx) {
    delete ctx;
}

extern "C" void orpheus_set_n_threads(struct orpheus_context* ctx, int n_threads) {
    if (!ctx) {
        return;
    }
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
}

extern "C" int orpheus_set_codec_path(struct orpheus_context* ctx, const char* path) {
    if (!ctx || !path) {
        return -1;
    }
    ctx->snac_codec_path = path;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: codec path set to '%s' (SNAC C++ decoder lands in slice b)\n", path);
    }
    return 0;
}

extern "C" int orpheus_n_speakers(struct orpheus_context* ctx) {
    if (!ctx) {
        return 0;
    }
    return (int)ctx->hp.spk_names.size();
}

extern "C" const char* orpheus_get_speaker_name(struct orpheus_context* ctx, int i) {
    if (!ctx || i < 0 || (size_t)i >= ctx->hp.spk_names.size()) {
        return nullptr;
    }
    return ctx->hp.spk_names[i].c_str();
}

extern "C" int orpheus_set_speaker_by_name(struct orpheus_context* ctx, const char* name) {
    if (!ctx || !name) {
        return -1;
    }
    const std::string target = lower(name);
    for (size_t i = 0; i < ctx->hp.spk_names.size(); i++) {
        if (lower(ctx->hp.spk_names[i]) == target) {
            ctx->active_speaker = (int)i;
            return 0;
        }
    }
    return -2;
}

extern "C" int orpheus_is_fixed_speaker(struct orpheus_context* ctx) {
    return (ctx && ctx->hp.tts_model_type == "fixed_speaker") ? 1 : 0;
}

extern "C" int32_t* orpheus_synthesize_codes(struct orpheus_context* ctx, const char* /*text*/, int* out_n) {
    if (!ctx) {
        return nullptr;
    }
    if (out_n) {
        *out_n = 0;
    }
    fprintf(stderr, "orpheus: synthesize_codes is not implemented yet — talker AR forward "
                    "lands in slice (b) of PLAN #57 Phase 2.\n");
    return nullptr;
}

extern "C" float* orpheus_synthesize(struct orpheus_context* ctx, const char* /*text*/, int* out_n_samples) {
    if (!ctx) {
        return nullptr;
    }
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    fprintf(stderr, "orpheus: synthesize is not implemented yet — talker forward + SNAC C++ decoder "
                    "land in slices (b)/(c) of PLAN #57 Phase 2.\n");
    return nullptr;
}

extern "C" void orpheus_codes_free(int32_t* codes) {
    std::free(codes);
}

extern "C" void orpheus_pcm_free(float* pcm) {
    std::free(pcm);
}
