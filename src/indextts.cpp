// indextts.cpp -- IndexTTS-1.5 TTS backend (GPT-2 AR + BigVGAN vocoder).
//
// Phase 1: GPT-2 autoregressive decoder with KV cache.
//   - Loads the GPT GGUF (Conformer + Perceiver + GPT-2 + tokenizer)
//   - Builds the GPT forward pass graph
//   - Generates mel codes via AR decode
//   - Conformer/Perceiver conditioning is scaffolded but uses dummy zeros for now
//
// Phase 2: BigVGAN vocoder + latent extraction.
//   - Second GPT forward pass extracts hidden states (return_latent)
//   - BigVGAN vocoder converts hidden states to 24kHz waveform
//   - Speaker conditioning uses zero embedding (ECAPA-TDNN deferred)
//
// Architecture (from actual checkpoint analysis):
//   GPT-2: 24L, d=1280, 20 heads, head_dim=64, FFN=5120, GELU(tanh)
//   Text vocab: 12001 (12000 BPE + stop_text=1)
//   Mel vocab: 8194 (8192 codes + start_mel + stop_mel)
//   Text pos: 602, Mel pos: 803
//   Conditioning: Conformer(6L, d=512) → Perceiver(2L, 32 latents, d=1280)

#include "indextts.h"
#include "indextts_voc.h"
#include "core/fastconformer.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ── Hyperparameters ──────────────────────────────────────────────

struct indextts_hp {
    // GPT-2
    uint32_t n_layers = 24;
    uint32_t d_model = 1280;
    uint32_t n_heads = 20;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 5120;

    // Vocab / embeddings
    uint32_t text_vocab_size = 12001; // 12000 BPE + stop_text
    uint32_t mel_vocab_size = 8194;   // 8192 codes + start_mel + stop_mel
    uint32_t text_pos_size = 602;     // max_text_tokens + 2
    uint32_t mel_pos_size = 803;      // max_mel_tokens + 3

    // Special tokens
    uint32_t start_mel_token = 8192;
    uint32_t stop_mel_token = 8193;

    // Conditioning
    uint32_t cond_d_model = 512; // Conformer hidden dim
    uint32_t cond_n_layers = 6;  // Conformer layers
    uint32_t cond_n_heads = 8;   // Conformer attention heads
    uint32_t perceiver_n_layers = 2;
    uint32_t perceiver_n_latents = 32;
    uint32_t perceiver_d_model = 1280;

    // Mel spectrogram (for reference audio)
    uint32_t n_mels = 100;
    uint32_t sample_rate = 24000;
    uint32_t hop_length = 256;
    uint32_t n_fft = 1024;
};

// ── Tensor structs ───────────────────────────────────────────────

struct indextts_gpt2_block {
    ggml_tensor* attn_norm_w = nullptr; // ln_1.weight
    ggml_tensor* attn_norm_b = nullptr; // ln_1.bias
    ggml_tensor* attn_qkv_w = nullptr;  // c_attn.weight (1280 → 3840)
    ggml_tensor* attn_qkv_b = nullptr;  // c_attn.bias
    ggml_tensor* attn_proj_w = nullptr; // c_proj.weight
    ggml_tensor* attn_proj_b = nullptr; // c_proj.bias
    ggml_tensor* ffn_norm_w = nullptr;  // ln_2.weight
    ggml_tensor* ffn_norm_b = nullptr;  // ln_2.bias
    ggml_tensor* ffn_fc_w = nullptr;    // mlp.c_fc.weight (1280 → 5120)
    ggml_tensor* ffn_fc_b = nullptr;    // mlp.c_fc.bias
    ggml_tensor* ffn_proj_w = nullptr;  // mlp.c_proj.weight (5120 → 1280)
    ggml_tensor* ffn_proj_b = nullptr;  // mlp.c_proj.bias
};

struct indextts_model {
    // Embeddings
    ggml_tensor* text_emb_w = nullptr;     // (text_vocab, d_model)
    ggml_tensor* mel_emb_w = nullptr;      // (mel_vocab, d_model)
    ggml_tensor* text_pos_emb_w = nullptr; // (text_pos_size, d_model)
    ggml_tensor* mel_pos_emb_w = nullptr;  // (mel_pos_size, d_model)

    // GPT-2 transformer blocks
    std::vector<indextts_gpt2_block> blocks;

    // Final norm + LM head
    ggml_tensor* final_norm_w = nullptr; // (d_model)
    ggml_tensor* final_norm_b = nullptr; // (d_model)
    ggml_tensor* mel_head_w = nullptr;   // (mel_vocab, d_model)
    ggml_tensor* mel_head_b = nullptr;   // (mel_vocab)

    // Conditioning encoder (Conformer) — tensors loaded but not wired in Phase 1
    // Perceiver — tensors loaded but not wired in Phase 1
};

// ── Tokenizer ────────────────────────────────────────────────────

struct indextts_tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float> scores;
    bool loaded = false;
};

// Simple character-level fallback tokenizer when BPE vocab is not available.
// Each UTF-8 byte becomes its own token ID (clamped to vocab size).
static std::vector<int32_t> tokenize_fallback(const indextts_tokenizer& /*tok*/, const std::string& text,
                                              uint32_t vocab_size) {
    std::vector<int32_t> result;
    for (unsigned char c : text) {
        int32_t id = (int32_t)c;
        if (id >= (int32_t)vocab_size) {
            id = 0;
        }
        result.push_back(id);
    }
    return result;
}

// SentencePiece BPE tokenizer — iterative pair merging using vocab scores.
static std::vector<int32_t> tokenize_bpe(const indextts_tokenizer& tok, const std::string& text) {
    if (!tok.loaded || tok.id_to_token.empty())
        return tokenize_fallback(tok, text, 12000);

    // SentencePiece: leading space → ▁ (U+2581)
    std::string proc;
    proc += "\xe2\x96\x81";
    for (char c : text)
        proc += (c == ' ') ? std::string("\xe2\x96\x81") : std::string(1, c);

    // Split into UTF-8 characters
    struct piece {
        size_t start, len;
    };
    std::vector<piece> pieces;
    for (size_t i = 0; i < proc.size();) {
        size_t cl = 1;
        unsigned char c = proc[i];
        if ((c & 0xE0) == 0xC0)
            cl = 2;
        else if ((c & 0xF0) == 0xE0)
            cl = 3;
        else if ((c & 0xF8) == 0xF0)
            cl = 4;
        if (i + cl > proc.size())
            cl = 1;
        pieces.push_back({i, cl});
        i += cl;
    }

    // Iterative merge: find pair with highest score, merge, repeat
    while (pieces.size() > 1) {
        float best_s = -1e30f;
        size_t best_j = SIZE_MAX;
        for (size_t j = 0; j + 1 < pieces.size(); j++) {
            std::string m = proc.substr(pieces[j].start, pieces[j].len + pieces[j + 1].len);
            auto it = tok.token_to_id.find(m);
            if (it != tok.token_to_id.end()) {
                float s = (it->second < (int32_t)tok.scores.size()) ? tok.scores[it->second] : -1e20f;
                if (s > best_s) {
                    best_s = s;
                    best_j = j;
                }
            }
        }
        if (best_j == SIZE_MAX)
            break;
        pieces[best_j].len += pieces[best_j + 1].len;
        pieces.erase(pieces.begin() + (long)best_j + 1);
    }

    // Convert pieces to IDs, merging consecutive unknowns into single <unk>=2
    std::vector<int32_t> result;
    bool prev_unk = false;
    for (const auto& p : pieces) {
        std::string s = proc.substr(p.start, p.len);
        auto it = tok.token_to_id.find(s);
        if (it != tok.token_to_id.end()) {
            result.push_back(it->second);
            prev_unk = false;
        } else {
            if (!prev_unk)
                result.push_back(2); // <unk>
            prev_unk = true;
        }
    }
    return result;
}

// ── Sampler ──────────────────────────────────────────────────────

static float sample_rng(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static int32_t sample_top_k_top_p(const float* logits, int n_vocab, float temperature, int top_k, float top_p,
                                  float rep_penalty, const std::vector<int32_t>& recent_tokens, uint64_t& rng_state) {
    // Apply repetition penalty
    std::vector<float> adjusted(logits, logits + n_vocab);
    for (int32_t tok : recent_tokens) {
        if (tok >= 0 && tok < n_vocab) {
            if (adjusted[tok] > 0) {
                adjusted[tok] /= rep_penalty;
            } else {
                adjusted[tok] *= rep_penalty;
            }
        }
    }

    // Temperature
    if (temperature > 0.0f) {
        for (int i = 0; i < n_vocab; i++) {
            adjusted[i] /= temperature;
        }
    }

    // Top-k selection
    std::vector<std::pair<float, int>> scored(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        scored[i] = {adjusted[i], i};
    }
    if (top_k > 0 && top_k < n_vocab) {
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    }

    // Softmax
    float max_val = scored[0].first;
    for (const auto& s : scored) {
        if (s.first > max_val) {
            max_val = s.first;
        }
    }
    float sum = 0.0f;
    std::vector<float> probs(scored.size());
    for (size_t i = 0; i < scored.size(); i++) {
        probs[i] = std::exp(scored[i].first - max_val);
        sum += probs[i];
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }

    // Top-p (nucleus) filtering
    if (top_p < 1.0f) {
        // Sort by probability descending
        std::vector<size_t> indices(probs.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        float cumulative = 0.0f;
        size_t cutoff = probs.size();
        for (size_t i = 0; i < indices.size(); i++) {
            cumulative += probs[indices[i]];
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out everything after cutoff
        for (size_t i = cutoff; i < indices.size(); i++) {
            probs[indices[i]] = 0.0f;
        }
        // Re-normalize
        sum = 0.0f;
        for (float p : probs) {
            sum += p;
        }
        if (sum > 0.0f) {
            for (float& p : probs) {
                p /= sum;
            }
        }
    }

    // Sample
    if (temperature <= 0.0f) {
        // Greedy
        int best = 0;
        for (size_t i = 1; i < probs.size(); i++) {
            if (probs[i] > probs[best]) {
                best = (int)i;
            }
        }
        return scored[best].second;
    }

    float r = sample_rng(rng_state);
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        cum += probs[i];
        if (r <= cum) {
            return scored[i].second;
        }
    }
    return scored[probs.size() - 1].second;
}

// ── Tensor read helper ───────────────────────────────────────────

static void tensor_get_f32(ggml_tensor* t, float* out, size_t n) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    } else {
        // Dequantize into float
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        const auto* tt = ggml_get_type_traits(t->type);
        GGML_ASSERT(tt && tt->to_float && "unsupported weight type in tensor_get_f32");
        tt->to_float(tmp.data(), out, (int64_t)n);
    }
}

} // namespace

// ── Context ──────────────────────────────────────────────────────

struct indextts_context {
    indextts_context_params params{};
    int n_threads = 4;

    indextts_hp hp;
    indextts_model model;
    indextts_tokenizer tokenizer;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // BigVGAN vocoder (Phase 2)
    std::string vocoder_path;
    indextts_voc_context* voc = nullptr;

    // Conditioning (ref audio → conditioning latents + speaker embedding)
    std::vector<float> cond_latents; // [perceiver_n_latents * d_model] from Conformer+Perceiver
    std::vector<float> spk_emb;      // [512] from ECAPA-TDNN

    // Compute metadata for conditioning graph (separate from GPT graph)
    std::vector<uint8_t> cond_compute_meta;

    // RNG
    uint64_t rng_state = 0xdeadbeefcafebabeULL;

    ~indextts_context() {
        if (voc) {
            indextts_voc_free(voc);
        }
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (kv_buf) {
            ggml_backend_buffer_free(kv_buf);
        }
        if (kv_ctx) {
            ggml_free(kv_ctx);
        }
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

// ── Metadata loading ─────────────────────────────────────────────

static void load_metadata(indextts_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers = core_gguf::kv_u32(g, "indextts.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(g, "indextts.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(g, "indextts.n_heads", hp.n_heads);
    hp.head_dim = core_gguf::kv_u32(g, "indextts.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(g, "indextts.ff_dim", hp.ff_dim);

    hp.text_vocab_size = core_gguf::kv_u32(g, "indextts.text_vocab_size", hp.text_vocab_size);
    hp.mel_vocab_size = core_gguf::kv_u32(g, "indextts.mel_vocab_size", hp.mel_vocab_size);
    hp.text_pos_size = core_gguf::kv_u32(g, "indextts.text_pos_size", hp.text_pos_size);
    hp.mel_pos_size = core_gguf::kv_u32(g, "indextts.mel_pos_size", hp.mel_pos_size);

    hp.start_mel_token = core_gguf::kv_u32(g, "indextts.start_mel_token", hp.start_mel_token);
    hp.stop_mel_token = core_gguf::kv_u32(g, "indextts.stop_mel_token", hp.stop_mel_token);

    // Conditioning encoder
    hp.cond_d_model = core_gguf::kv_u32(g, "indextts.cond_d_model", hp.cond_d_model);
    hp.cond_n_layers = core_gguf::kv_u32(g, "indextts.cond_n_layers", hp.cond_n_layers);
    hp.cond_n_heads = core_gguf::kv_u32(g, "indextts.cond_n_heads", hp.cond_n_heads);
    hp.perceiver_n_layers = core_gguf::kv_u32(g, "indextts.perceiver_n_layers", hp.perceiver_n_layers);
    hp.perceiver_n_latents = core_gguf::kv_u32(g, "indextts.perceiver_n_latents", hp.perceiver_n_latents);

    // Load tokenizer vocab + scores if present
    std::vector<std::string> tokens = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!tokens.empty()) {
        c->tokenizer.id_to_token = std::move(tokens);
        for (size_t i = 0; i < c->tokenizer.id_to_token.size(); i++) {
            c->tokenizer.token_to_id[c->tokenizer.id_to_token[i]] = (int32_t)i;
        }
        int scores_key = gguf_find_key(g, "tokenizer.ggml.scores");
        if (scores_key >= 0) {
            int n = (int)gguf_get_arr_n(g, scores_key);
            const float* sd = (const float*)gguf_get_arr_data(g, scores_key);
            c->tokenizer.scores.assign(sd, sd + n);
        }
        c->tokenizer.loaded = true;
    }
}

// ── Tensor binding ───────────────────────────────────────────────

static bool bind_model(indextts_context* c) {
    auto& m = c->model;
    auto& t = c->tensors;

    // Embeddings
    m.text_emb_w = core_gguf::require(t, "text_embedding.weight", "indextts");
    m.mel_emb_w = core_gguf::require(t, "mel_embedding.weight", "indextts");
    m.text_pos_emb_w = core_gguf::require(t, "text_pos.weight", "indextts");
    m.mel_pos_emb_w = core_gguf::require(t, "mel_pos.weight", "indextts");

    if (!m.text_emb_w || !m.mel_emb_w || !m.text_pos_emb_w || !m.mel_pos_emb_w) {
        return false;
    }

    // GPT-2 blocks
    m.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = m.blocks[i];
        auto fmt = [&](const char* suffix) { return core_gguf::format_layer_name("gpt.h.%d.", i) + suffix; };

        b.attn_norm_w = core_gguf::require(t, fmt("ln_1.weight").c_str(), "indextts");
        b.attn_norm_b = core_gguf::require(t, fmt("ln_1.bias").c_str(), "indextts");
        b.attn_qkv_w = core_gguf::require(t, fmt("attn.c_attn.weight").c_str(), "indextts");
        b.attn_qkv_b = core_gguf::require(t, fmt("attn.c_attn.bias").c_str(), "indextts");
        b.attn_proj_w = core_gguf::require(t, fmt("attn.c_proj.weight").c_str(), "indextts");
        b.attn_proj_b = core_gguf::require(t, fmt("attn.c_proj.bias").c_str(), "indextts");
        b.ffn_norm_w = core_gguf::require(t, fmt("ln_2.weight").c_str(), "indextts");
        b.ffn_norm_b = core_gguf::require(t, fmt("ln_2.bias").c_str(), "indextts");
        b.ffn_fc_w = core_gguf::require(t, fmt("mlp.c_fc.weight").c_str(), "indextts");
        b.ffn_fc_b = core_gguf::require(t, fmt("mlp.c_fc.bias").c_str(), "indextts");
        b.ffn_proj_w = core_gguf::require(t, fmt("mlp.c_proj.weight").c_str(), "indextts");
        b.ffn_proj_b = core_gguf::require(t, fmt("mlp.c_proj.bias").c_str(), "indextts");

        if (!b.attn_norm_w || !b.attn_qkv_w || !b.attn_proj_w || !b.ffn_norm_w || !b.ffn_fc_w || !b.ffn_proj_w) {
            fprintf(stderr, "indextts: missing tensors in GPT block %u\n", i);
            return false;
        }
    }

    // Final norm + head
    m.final_norm_w = core_gguf::require(t, "final_norm.weight", "indextts");
    m.final_norm_b = core_gguf::require(t, "final_norm.bias", "indextts");
    m.mel_head_w = core_gguf::require(t, "mel_head.weight", "indextts");
    m.mel_head_b = core_gguf::try_get(t, "mel_head.bias");

    if (!m.final_norm_w || !m.final_norm_b || !m.mel_head_w) {
        return false;
    }

    // Debug: print key tensor shapes
    auto& b0 = m.blocks[0];
    fprintf(stderr, "indextts: c_attn.w ne=(%lld,%lld) text_emb ne=(%lld,%lld) mel_head ne=(%lld,%lld)\n",
            (long long)b0.attn_qkv_w->ne[0], (long long)b0.attn_qkv_w->ne[1], (long long)m.text_emb_w->ne[0],
            (long long)m.text_emb_w->ne[1], (long long)m.mel_head_w->ne[0], (long long)m.mel_head_w->ne[1]);

    return true;
}

// ── Reference mel computation ───────────────────────────────────
//
// Compute 100-band log-mel spectrogram from reference audio for the
// conditioning encoder. The Python reference uses torchaudio's
// mel spectrogram with 24kHz sample rate, 100 bands, hop=256, n_fft=1024.
// Input audio is expected at 16kHz — upsampled to 24kHz internally.

// Simple linear interpolation resampler (16kHz → 24kHz = ratio 3:2)
static std::vector<float> resample_16k_to_24k(const float* pcm, int n_samples) {
    const double ratio = 24000.0 / 16000.0; // 1.5
    int n_out = (int)(n_samples * ratio);
    std::vector<float> out(n_out);
    for (int i = 0; i < n_out; i++) {
        double src_pos = i / ratio;
        int idx = (int)src_pos;
        double frac = src_pos - idx;
        if (idx + 1 < n_samples) {
            out[i] = (float)((1.0 - frac) * pcm[idx] + frac * pcm[idx + 1]);
        } else if (idx < n_samples) {
            out[i] = pcm[idx];
        }
    }
    return out;
}

static std::vector<float> compute_ref_mel(const float* pcm, int n_samples, int* T_out) {
    const int n_fft = 1024, hop = 256, n_mels = 100, sr = 24000;
    const float fmin = 0.0f, fmax = 12000.0f;

    // Resample 16kHz input to 24kHz
    auto pcm_24k = resample_16k_to_24k(pcm, n_samples);
    const float* audio_ptr = pcm_24k.data();
    n_samples = (int)pcm_24k.size();
    const int pad = (n_fft - hop) / 2; // 384

    // Reflect-pad audio (use resampled audio_ptr, not original pcm!)
    std::vector<float> audio_p(n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; i++) {
        audio_p[i] = audio_ptr[std::min(pad - i, n_samples - 1)];
    }
    for (int i = 0; i < n_samples; i++) {
        audio_p[pad + i] = audio_ptr[i];
    }
    for (int i = 0; i < pad; i++) {
        audio_p[pad + n_samples + i] = audio_ptr[std::max(n_samples - 2 - i, 0)];
    }

    // Periodic Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)n_fft));
    }

    const int n_freqs = n_fft / 2 + 1;
    auto mel_fb = core_mel::build_slaney_fb(sr, n_fft, n_mels, fmin, fmax);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-7f;
    p.spec_kind = core_mel::SpecKind::Magnitude;
    p.norm = core_mel::Normalization::None;
    // MelsTime layout: [n_mels, T] — needed for Conv2d subsampling
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.center_pad = false;

    int T = 0;
    auto mel = core_mel::compute(audio_p.data(), (int)audio_p.size(), hann.data(), n_fft, mel_fb.data(), n_freqs,
                                 core_fft::fft_radix2_wrapper, p, T);
    if (T_out) {
        *T_out = T;
    }
    return mel;
}

// ── Conformer conditioning encoder ─────────────────────────────
//
// Conv2d(1→512, 3x3, stride=2, padding=0) subsampling
// → Linear(49*512 → 512)
// → add sinusoidal pos_enc
// → 6× Conformer blocks (no macaron FFN, single FFN per block)
// → after_norm LayerNorm
//
// The Conformer block ordering follows IndexTTS Python (Espnet style):
//   1. Self-attention (with rel-pos + untied biases)
//   2. Conv module (pw1 → GLU → dw conv → BN → SiLU → pw2)
//   3. FFN (up → ReLU → down)
//   4. Final LN

static ggml_cgraph* build_cond_enc_graph(indextts_context* c, int T_mel) {
    const auto& hp = c->hp;
    const int d = (int)hp.cond_d_model;         // 512
    const int n_heads = (int)hp.cond_n_heads;   // 8
    const int head_dim = d / n_heads;           // 64
    const int n_layers = (int)hp.cond_n_layers; // 6
    const float ln_eps = 1e-5f;
    auto& ts = c->tensors;

    const size_t n_nodes = 16384;
    ggml_init_params ip = {c->cond_compute_meta.size(), c->cond_compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_nodes, false);

    // Input: mel [n_mels=100, T_mel] as a 4D tensor [T_mel, 100, 1, 1]
    // Conv2d expects [W, H, C_in, batch]
    ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_mel, 100, 1, 1);
    ggml_set_name(mel, "cond_mel");
    ggml_set_input(mel);

    // Conv2d(1→512, 3x3, stride=2, padding=0)
    ggml_tensor* conv_w = core_gguf::try_get(ts, "cond_enc.embed.conv.0.weight");
    ggml_tensor* conv_b = core_gguf::try_get(ts, "cond_enc.embed.conv.0.bias");
    ggml_tensor* cur = ggml_conv_2d(ctx0, conv_w, mel, 2, 2, 0, 0, 1, 1);
    if (conv_b) {
        ggml_tensor* bias_4d = ggml_cast(ctx0, ggml_reshape_4d(ctx0, conv_b, 1, 1, conv_b->ne[0], 1), GGML_TYPE_F32);
        cur = ggml_add(ctx0, cur, bias_4d);
    }

    // T_enc = floor((T_mel - 3) / 2) + 1, H_enc = floor((100 - 3) / 2) + 1 = 49
    int T_enc = ((T_mel - 3) / 2) + 1;
    int H_enc = 49; // (100 - 3) / 2 + 1

    ggml_set_name(cur, "dbg_conv2d_out");
    ggml_set_output(cur);

    // Reshape to [H_enc * 512, T_enc] then linear → [512, T_enc]
    // cur is [W=T_enc, H=H_enc, C=512, batch=1]
    // Permute to [W, C, H, batch] → reshape to [H*C, T_enc]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [T_enc, 512, H_enc, 1]
    cur = ggml_reshape_2d(ctx0, cur, H_enc * 512, T_enc);       // [25088, T_enc]

    ggml_tensor* lin_w = core_gguf::try_get(ts, "cond_enc.embed.out.0.weight");
    ggml_tensor* lin_b = core_gguf::try_get(ts, "cond_enc.embed.out.0.bias");
    cur = ggml_mul_mat(ctx0, lin_w, cur); // [512, T_enc]
    if (lin_b) {
        cur = ggml_add(ctx0, cur, lin_b);
    }

    // Add sinusoidal positional encoding (stored as [512, 5000, 1], F16)
    ggml_tensor* pe_full = core_gguf::try_get(ts, "cond_enc.embed.pos_enc.pe");
    if (pe_full && T_enc > 0) {
        // pe_full is [512, 5000, 1], we need [512, T_enc]
        ggml_tensor* pe = ggml_view_2d(ctx0, pe_full, d, T_enc, pe_full->nb[1], 0);
        pe = ggml_cast(ctx0, pe, GGML_TYPE_F32);
        cur = ggml_add(ctx0, cur, pe);
    }

    // Debug: output pre-transformer embeddings
    ggml_set_name(cur, "dbg_pre_transformer");
    ggml_set_output(cur);

    // Build relative position sinusoidal table for self-attention
    auto pos_data = core_conformer::make_pos_enc(d, T_enc);
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T_enc - 1);
    ggml_set_name(pos_enc, "cond_pos_enc");
    ggml_set_input(pos_enc);

    // 6 Conformer blocks
    for (int il = 0; il < n_layers; il++) {
        auto fmt = [&](const char* s) -> std::string {
            char buf[128];
            snprintf(buf, sizeof(buf), "cond_enc.enc.%d.%s", il, s);
            return buf;
        };

        auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
            ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
            return b ? ggml_add(ctx0, y, b) : y;
        };

        ggml_tensor* inpL = cur;

        // ---- Self-Attention with rel-pos (Shaw-style with untied biases) ----
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, core_gguf::try_get(ts, fmt("norm_mha.weight").c_str()));
        x = ggml_add(ctx0, x, core_gguf::try_get(ts, fmt("norm_mha.bias").c_str()));

        ggml_tensor* Q = mm_bias(core_gguf::try_get(ts, fmt("sa.linear_q.weight").c_str()), x,
                                 core_gguf::try_get(ts, fmt("sa.linear_q.bias").c_str()));
        ggml_tensor* K_ = mm_bias(core_gguf::try_get(ts, fmt("sa.linear_k.weight").c_str()), x,
                                  core_gguf::try_get(ts, fmt("sa.linear_k.bias").c_str()));
        ggml_tensor* V = mm_bias(core_gguf::try_get(ts, fmt("sa.linear_v.weight").c_str()), x,
                                 core_gguf::try_get(ts, fmt("sa.linear_v.bias").c_str()));
        ggml_tensor* R = ggml_mul_mat(ctx0, core_gguf::try_get(ts, fmt("sa.linear_pos.weight").c_str()), pos_enc);

        ggml_tensor* pos_u = core_gguf::try_get(ts, fmt("sa.pos_bias_u").c_str());
        ggml_tensor* pos_v = core_gguf::try_get(ts, fmt("sa.pos_bias_v").c_str());
        // Cast F16 pos biases to F32
        ggml_tensor* pos_u_f32 = ggml_cast(ctx0, ggml_reshape_1d(ctx0, pos_u, d), GGML_TYPE_F32);
        ggml_tensor* pos_v_f32 = ggml_cast(ctx0, ggml_reshape_1d(ctx0, pos_v, d), GGML_TYPE_F32);
        ggml_tensor* Q_u = ggml_add(ctx0, Q, pos_u_f32);
        ggml_tensor* Q_v = ggml_add(ctx0, Q, pos_v_f32);

        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T_enc), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T_enc), 0, 2, 1, 3);
        K_ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T_enc), 0, 2, 1, 3);
        R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T_enc - 1), 0, 2, 1, 3);

        ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        ggml_tensor* BD = core_conformer::rel_shift(ctx0, BD_raw);

        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* BD_c = ggml_cont(ctx0, BD);
        ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);
        ggml_tensor* BD_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);

        ggml_tensor* V_ =
            ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc), 0, 2, 1, 3));

        ggml_tensor* attn_out =
            ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q_u), ggml_cont(ctx0, K_), V_, BD_mask, scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx0, attn_out, d, T_enc);
        attn_out = mm_bias(core_gguf::try_get(ts, fmt("sa.linear_out.weight").c_str()), attn_out,
                           core_gguf::try_get(ts, fmt("sa.linear_out.bias").c_str()));
        cur = ggml_add(ctx0, inpL, attn_out);

        // ---- Conformer convolution module ----
        ggml_tensor* inpConv = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, core_gguf::try_get(ts, fmt("norm_conv.weight").c_str()));
        x = ggml_add(ctx0, x, core_gguf::try_get(ts, fmt("norm_conv.bias").c_str()));

        // pw1: (512 → 1024), then GLU
        ggml_tensor* pw1_w = core_gguf::try_get(ts, fmt("conv.pw1.weight").c_str());
        ggml_tensor* pw1_b = core_gguf::try_get(ts, fmt("conv.pw1.bias").c_str());
        ggml_tensor* pw1_w2d = ggml_reshape_2d(ctx0, pw1_w, d, 2 * d);
        ggml_tensor* cnv = mm_bias(pw1_w2d, x, pw1_b);
        ggml_tensor* cnv_gate = ggml_view_2d(ctx0, cnv, d, T_enc, cnv->nb[1], d * sizeof(float));
        cnv = ggml_mul(ctx0, ggml_view_2d(ctx0, cnv, d, T_enc, cnv->nb[1], 0), ggml_sigmoid(ctx0, cnv_gate));

        // dw conv (k=15, padding=7)
        const int K_conv = 15;
        ggml_tensor* dw_w = core_gguf::try_get(ts, fmt("conv.dw.weight").c_str());
        ggml_tensor* dw_b = core_gguf::try_get(ts, fmt("conv.dw.bias").c_str());
        ggml_tensor* dw_w_f32 = ggml_cast(ctx0, dw_w, GGML_TYPE_F32);
        ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K_conv, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T_enc) → (T_enc, d)
        cnv = ggml_reshape_4d(ctx0, cnv, T_enc, 1, d, 1);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K_conv - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        cnv = ggml_reshape_2d(ctx0, cnv, d, T_enc);
        if (dw_b) {
            cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, dw_b, d, 1));
        }

        // BatchNorm in conv module
        ggml_tensor* conv_norm_w = core_gguf::try_get(ts, fmt("conv.norm.weight").c_str());
        ggml_tensor* conv_norm_b = core_gguf::try_get(ts, fmt("conv.norm.bias").c_str());
        if (conv_norm_w && conv_norm_b) {
            // This is a LayerNorm (not BatchNorm) in the Conformer conv module
            cnv = ggml_norm(ctx0, cnv, ln_eps);
            cnv = ggml_mul(ctx0, cnv, conv_norm_w);
            cnv = ggml_add(ctx0, cnv, conv_norm_b);
        }
        cnv = ggml_silu(ctx0, cnv);

        // pw2: (512 → 512)
        ggml_tensor* pw2_w = core_gguf::try_get(ts, fmt("conv.pw2.weight").c_str());
        ggml_tensor* pw2_b = core_gguf::try_get(ts, fmt("conv.pw2.bias").c_str());
        ggml_tensor* pw2_w2d = ggml_reshape_2d(ctx0, pw2_w, d, d);
        cnv = mm_bias(pw2_w2d, cnv, pw2_b);
        cur = ggml_add(ctx0, inpConv, cnv);

        // ---- FFN ----
        ggml_tensor* inpFF = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, core_gguf::try_get(ts, fmt("norm_ff.weight").c_str()));
        x = ggml_add(ctx0, x, core_gguf::try_get(ts, fmt("norm_ff.bias").c_str()));

        x = mm_bias(core_gguf::try_get(ts, fmt("ff.w_1.weight").c_str()), x,
                    core_gguf::try_get(ts, fmt("ff.w_1.bias").c_str()));
        x = ggml_silu(ctx0, x);
        x = mm_bias(core_gguf::try_get(ts, fmt("ff.w_2.weight").c_str()), x,
                    core_gguf::try_get(ts, fmt("ff.w_2.bias").c_str()));
        cur = ggml_add(ctx0, inpFF, x);

        // ---- Final LN ----
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, core_gguf::try_get(ts, fmt("norm_final.weight").c_str()));
        cur = ggml_add(ctx0, cur, core_gguf::try_get(ts, fmt("norm_final.bias").c_str()));
    }

    // After-norm
    ggml_tensor* after_w = core_gguf::try_get(ts, "cond_enc.after_norm.weight");
    ggml_tensor* after_b = core_gguf::try_get(ts, "cond_enc.after_norm.bias");
    if (after_w && after_b) {
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, after_w);
        cur = ggml_add(ctx0, cur, after_b);
    }

    // cur is [512, T_enc] — the conformer output
    ggml_set_name(cur, "conformer_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Perceiver resampler ────────────────────────────────────────
//
// Takes conformer output (512, T_enc) and produces (d_model=1280, 32)
// conditioning latents via cross-attention.
//
// Architecture:
//   latents: [1280, 32] learned queries
//   proj_context: Linear(512→1280) projects conformer output
//   2 layers: cross-attn + GEGLU FFN
//   Final RMSNorm

static ggml_cgraph* build_perceiver_graph(indextts_context* c, int T_enc) {
    auto& ts = c->tensors;
    const int d_perc = (int)c->hp.perceiver_d_model;      // 1280
    const int n_latents = (int)c->hp.perceiver_n_latents; // 32
    const int n_layers_p = (int)c->hp.perceiver_n_layers; // 2

    // Cross-attention dimensions from weight shapes
    // to_q: [1280, 512] — d_q = 512
    // to_kv: [1280, 1024] — d_k = d_v = 512 each
    // to_out: [512, 1280] — back to 1280
    const int d_q = 512;
    // Number of heads: d_q / head_dim. The Python model uses 8 heads.
    const int n_heads_p = 8;
    const int head_dim_p = d_q / n_heads_p; // 64

    const size_t n_nodes = 4096;
    ggml_init_params ip = {c->cond_compute_meta.size(), c->cond_compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_nodes, false);

    // Input: conformer output [512, T_enc]
    ggml_tensor* context = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)c->hp.cond_d_model, T_enc);
    ggml_set_name(context, "perc_context");
    ggml_set_input(context);

    // Project context: 512 → 1280
    ggml_tensor* proj_w = core_gguf::try_get(ts, "perc.proj_context.weight");
    ggml_tensor* proj_b = core_gguf::try_get(ts, "perc.proj_context.bias");
    ggml_tensor* proj_ctx = ggml_mul_mat(ctx0, proj_w, context); // [1280, T_enc]
    if (proj_b) {
        proj_ctx = ggml_add(ctx0, proj_ctx, proj_b);
    }

    // Learned latent queries: [1280, 32]
    ggml_tensor* latents_w = core_gguf::try_get(ts, "perc.latents");
    // Copy latents to a compute tensor
    ggml_tensor* latents = ggml_cast(ctx0, latents_w, GGML_TYPE_F32); // [1280, 32]

    for (int il = 0; il < n_layers_p; il++) {
        char key[128];

        // ---- Cross-attention ----
        // Q from latents only
        snprintf(key, sizeof(key), "perc.layers.%d.0.to_q.weight", il);
        ggml_tensor* q_w = core_gguf::try_get(ts, key);
        ggml_tensor* Q = ggml_mul_mat(ctx0, q_w, latents); // [512, 32]

        // KV from [latents; projected_context] (cross_attn_include_queries=True)
        // Concatenate along sequence dimension: [1280, 32+T_enc]
        ggml_tensor* kv_input = ggml_concat(ctx0, latents, proj_ctx, 1); // [1280, 32+T_enc]
        int T_kv = n_latents + T_enc;

        snprintf(key, sizeof(key), "perc.layers.%d.0.to_kv.weight", il);
        ggml_tensor* kv_w = core_gguf::try_get(ts, key);
        ggml_tensor* KV = ggml_mul_mat(ctx0, kv_w, kv_input); // [1024, T_kv]

        // Split KV into K and V, each [512, T_kv]
        ggml_tensor* K_p = ggml_view_2d(ctx0, KV, d_q, T_kv, KV->nb[1], 0);
        ggml_tensor* V_p = ggml_view_2d(ctx0, KV, d_q, T_kv, KV->nb[1], d_q * sizeof(float));
        K_p = ggml_cont(ctx0, K_p);
        V_p = ggml_cont(ctx0, V_p);

        // Reshape for multi-head attention
        // Q: [512, 32] → [head_dim, n_heads, 32] → permute to [head_dim, 32, n_heads]
        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, head_dim_p, n_heads_p, n_latents), 0, 2, 1, 3);
        K_p = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_p, head_dim_p, n_heads_p, T_kv), 0, 2, 1, 3);
        V_p = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V_p, head_dim_p, n_heads_p, T_kv), 0, 2, 1, 3);

        // Flash attention (no causal mask for cross-attention)
        const float scale_p = 1.0f / sqrtf((float)head_dim_p);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q), ggml_cont(ctx0, K_p), ggml_cont(ctx0, V_p),
                                                nullptr, scale_p, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, d_q, n_latents); // [512, 32]

        // Output projection: [512, 32] → [1280, 32]
        snprintf(key, sizeof(key), "perc.layers.%d.0.to_out.weight", il);
        ggml_tensor* out_w = core_gguf::try_get(ts, key);
        attn = ggml_mul_mat(ctx0, out_w, attn); // [1280, 32]

        latents = ggml_add(ctx0, latents, attn);

        // ---- GEGLU FFN ----
        // gate+up: [1280, 32] → [3412, 32]
        snprintf(key, sizeof(key), "perc.layers.%d.1.0.weight", il);
        ggml_tensor* ffn_up_w = core_gguf::try_get(ts, key);
        snprintf(key, sizeof(key), "perc.layers.%d.1.0.bias", il);
        ggml_tensor* ffn_up_b = core_gguf::try_get(ts, key);
        ggml_tensor* ffn_h = ggml_mul_mat(ctx0, ffn_up_w, latents);
        if (ffn_up_b) {
            ffn_h = ggml_add(ctx0, ffn_h, ffn_up_b);
        }
        // Split at midpoint: 3412/2 = 1706
        const int inner = 1706;
        ggml_tensor* ffn_gate = ggml_view_2d(ctx0, ffn_h, inner, n_latents, ffn_h->nb[1], 0);
        ggml_tensor* ffn_up = ggml_view_2d(ctx0, ffn_h, inner, n_latents, ffn_h->nb[1], inner * sizeof(float));
        ffn_gate = ggml_cont(ctx0, ffn_gate);
        ffn_up = ggml_cont(ctx0, ffn_up);
        ggml_tensor* ffn_act = ggml_mul(ctx0, ggml_gelu(ctx0, ffn_gate), ffn_up);

        // down: [1706, 32] → [1280, 32]
        snprintf(key, sizeof(key), "perc.layers.%d.1.2.weight", il);
        ggml_tensor* ffn_down_w = core_gguf::try_get(ts, key);
        snprintf(key, sizeof(key), "perc.layers.%d.1.2.bias", il);
        ggml_tensor* ffn_down_b = core_gguf::try_get(ts, key);
        ggml_tensor* ffn_out = ggml_mul_mat(ctx0, ffn_down_w, ffn_act);
        if (ffn_down_b) {
            ffn_out = ggml_add(ctx0, ffn_out, ffn_down_b);
        }

        latents = ggml_add(ctx0, latents, ffn_out);
    }

    // Final RMSNorm
    ggml_tensor* norm_gamma = core_gguf::try_get(ts, "perc.norm.gamma");
    if (norm_gamma) {
        latents = ggml_rms_norm(ctx0, latents, 1e-8f);
        latents = ggml_mul(ctx0, latents, norm_gamma);
    }

    // Output: [1280, 32] = [d_model, n_latents]
    ggml_set_name(latents, "perceiver_out");
    ggml_set_output(latents);
    ggml_build_forward_expand(gf, latents);
    ggml_free(ctx0);
    return gf;
}

// ── Run conditioning pipeline ───────────────────────────────────
//
// ref_pcm → mel → Conformer → Perceiver → 32 latent vectors [1280, 32]
//
// Returns the 32 conditioning latent vectors as a flat float array of
// size (perceiver_n_latents * d_model). Stored in indextts_context for
// reuse across prefill and latent passes.

static bool run_conditioning(indextts_context* c, const float* ref_pcm, int ref_n_samples) {
    const int D = (int)c->hp.d_model;
    const int n_latents = (int)c->hp.perceiver_n_latents;
    const int cond_d = (int)c->hp.cond_d_model;

    // Check if conditioning tensors exist
    if (!core_gguf::try_get(c->tensors, "cond_enc.embed.conv.0.weight")) {
        if (c->params.verbosity >= 1) {
            fprintf(stderr, "indextts: conditioning encoder tensors not found, using zero latents\n");
        }
        c->cond_latents.assign((size_t)n_latents * D, 0.0f);
        return true;
    }

    // Allocate conditioning compute metadata if needed
    if (c->cond_compute_meta.empty()) {
        c->cond_compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    // Step 1: Compute reference mel spectrogram
    int T_mel = 0;
    auto mel = compute_ref_mel(ref_pcm, ref_n_samples, &T_mel);
    if (mel.empty() || T_mel <= 0) {
        fprintf(stderr, "indextts: failed to compute reference mel\n");
        c->cond_latents.assign((size_t)n_latents * D, 0.0f);
        return true;
    }

    // Limit mel to ~5s reference for ggml_conv_2d stability
    if (T_mel > 500) {
        T_mel = 500;
        mel.resize((size_t)100 * T_mel); // MelsTime layout: [n_mels, T]
    }

    if (c->params.verbosity >= 1) {
        fprintf(stderr, "indextts: ref mel: %d frames x 100 bands\n", T_mel);
    }

    // Debug: check mel values
    {
        int nnan = 0;
        float mrms = 0;
        for (size_t i = 0; i < mel.size(); i++) {
            if (std::isnan(mel[i]) || std::isinf(mel[i]))
                nnan++;
            else
                mrms += mel[i] * mel[i];
        }
        mrms = sqrtf(mrms / std::max((size_t)1, mel.size() - nnan));
        fprintf(stderr, "indextts: mel values rms=%.4f nan=%d/%zu min=%.4f max=%.4f\n", mrms, nnan, mel.size(),
                *std::min_element(mel.begin(), mel.end()), *std::max_element(mel.begin(), mel.end()));
    }

    // Step 2: Run Conformer encoder
    int T_enc = ((T_mel - 3) / 2) + 1;
    {
        ggml_cgraph* gf = build_cond_enc_graph(c, T_mel);
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
            fprintf(stderr, "indextts: failed to alloc Conformer graph\n");
            c->cond_latents.assign((size_t)n_latents * D, 0.0f);
            return true;
        }

        // mel is [n_mels, T_mel] in MelsTime layout — set as 4D [T_mel, 100, 1, 1]
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_mel"), mel.data(), 0, mel.size() * sizeof(float));

        // Set positional encoding
        auto pos_data = core_conformer::make_pos_enc(cond_d, T_enc);
        ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "cond_pos_enc");
        if (pos_t) {
            ggml_backend_tensor_set(pos_t, pos_data.data(), 0, pos_data.size() * sizeof(float));
        }

        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "indextts: Conformer compute failed\n");
            c->cond_latents.assign((size_t)n_latents * D, 0.0f);
            return true;
        }

        // Debug: check conv2d output
        {
            ggml_tensor* dbg = ggml_graph_get_tensor(gf, "dbg_conv2d_out");
            if (dbg) {
                int n = (int)ggml_nelements(dbg);
                std::vector<float> d(n);
                ggml_backend_tensor_get(dbg, d.data(), 0, n * sizeof(float));
                int nnan = 0;
                float s2 = 0;
                for (float v : d) {
                    if (std::isnan(v) || std::isinf(v))
                        nnan++;
                    else
                        s2 += v * v;
                }
                fprintf(stderr, "indextts: conv2d_out rms=%.4f nan=%d/%d ne=(%lld,%lld,%lld,%lld)\n",
                        sqrtf(s2 / std::max(1, n - nnan)), nnan, n, (long long)dbg->ne[0], (long long)dbg->ne[1],
                        (long long)dbg->ne[2], (long long)dbg->ne[3]);
            }
        }
        // Debug: check pre-transformer output
        {
            ggml_tensor* dbg = ggml_graph_get_tensor(gf, "dbg_pre_transformer");
            if (dbg) {
                int n = (int)ggml_nelements(dbg);
                std::vector<float> d(n);
                ggml_backend_tensor_get(dbg, d.data(), 0, n * sizeof(float));
                int nnan = 0;
                float s2 = 0;
                for (float v : d) {
                    if (std::isnan(v) || std::isinf(v))
                        nnan++;
                    else
                        s2 += v * v;
                }
                fprintf(stderr, "indextts: pre-transformer rms=%.4f nan=%d/%d\n", sqrtf(s2 / std::max(1, n - nnan)),
                        nnan, n);
            }
        }

        // Read conformer output: [512, T_enc]
        ggml_tensor* out = ggml_graph_get_tensor(gf, "conformer_out");
        std::vector<float> conf_out((size_t)cond_d * T_enc);
        ggml_backend_tensor_get(out, conf_out.data(), 0, conf_out.size() * sizeof(float));

        {
            float crms = 0;
            int n_nan = 0;
            for (size_t i = 0; i < conf_out.size(); i++) {
                if (std::isnan(conf_out[i]) || std::isinf(conf_out[i]))
                    n_nan++;
                else
                    crms += conf_out[i] * conf_out[i];
            }
            crms = sqrtf(crms / std::max((size_t)1, conf_out.size()));
            if (c->params.verbosity >= 1) {
                fprintf(stderr,
                        "indextts: Conformer output: [%d, %d] rms=%.4f nan=%d first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                        cond_d, T_enc, crms, n_nan, conf_out[0], conf_out[1], conf_out[2], conf_out[3], conf_out[4]);
            }
        }

        // Step 3: Run Perceiver resampler
        ggml_cgraph* pgf = build_perceiver_graph(c, T_enc);
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, pgf)) {
            fprintf(stderr, "indextts: failed to alloc Perceiver graph\n");
            c->cond_latents.assign((size_t)n_latents * D, 0.0f);
            return true;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(pgf, "perc_context"), conf_out.data(), 0,
                                conf_out.size() * sizeof(float));

        if (ggml_backend_sched_graph_compute(c->sched, pgf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "indextts: Perceiver compute failed\n");
            c->cond_latents.assign((size_t)n_latents * D, 0.0f);
            return true;
        }

        // Read perceiver output: [1280, 32]
        ggml_tensor* perc_out = ggml_graph_get_tensor(pgf, "perceiver_out");
        c->cond_latents.resize((size_t)n_latents * D);
        ggml_backend_tensor_get(perc_out, c->cond_latents.data(), 0, c->cond_latents.size() * sizeof(float));

        if (c->params.verbosity >= 1) {
            float norm = 0.0f;
            for (float v : c->cond_latents) {
                norm += v * v;
            }
            fprintf(stderr, "indextts: Perceiver output: [%d, %d] norm=%.4f\n", D, n_latents, sqrtf(norm));
        }
    }

    return true;
}

// ── KV cache allocation ─────────────────────────────────────────

static bool kv_alloc(indextts_context* c, int max_ctx) {
    if (c->kv_ctx && max_ctx <= c->kv_max_ctx) {
        return true;
    }

    // Free existing
    if (c->kv_buf) {
        ggml_backend_buffer_free(c->kv_buf);
    }
    if (c->kv_ctx) {
        ggml_free(c->kv_ctx);
    }
    c->kv_buf = nullptr;
    c->kv_ctx = nullptr;

    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_h = (int)hp.n_heads; // MHA: n_kv_heads == n_heads
    const int nl = (int)hp.n_layers;
    c->kv_max_ctx = max_ctx;

    // KV shape: (head_dim, max_ctx, n_heads, n_layers)
    struct ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    c->kv_ctx = ggml_init(ip);
    if (!c->kv_ctx) {
        return false;
    }

    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_h, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_h, nl);

    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf) {
        fprintf(stderr, "indextts: failed to allocate KV cache\n");
        ggml_free(c->kv_ctx);
        c->kv_ctx = nullptr;
        return false;
    }

    if (c->params.verbosity >= 1) {
        size_t kb = ggml_nbytes(c->kv_k);
        size_t vb = ggml_nbytes(c->kv_v);
        fprintf(stderr, "indextts: kv cache %d MiB (hd=%d max=%d n_h=%d nl=%d)\n", (int)((kb + vb) / 1048576), hd,
                max_ctx, n_h, nl);
    }

    return true;
}

// ── GPT-2 graph builder ──────────────────────────────────────────

static ggml_cgraph* build_gpt2_kv_graph(indextts_context* c, int n_past, int n_tokens) {
    const auto& hp = c->hp;
    const int D = (int)hp.d_model;
    const int n_h = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->model.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // Fused QKV: x @ c_attn_w + c_attn_b → (3*D, T)
        ggml_tensor* qkv = ggml_mul_mat(ctx0, b.attn_qkv_w, x);
        qkv = ggml_add(ctx0, qkv, b.attn_qkv_b);

        // Split into Q, K, V — each (D, T)
        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor* Q = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts);
        ggml_tensor* V = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts);
        if (T > 1) {
            Q = ggml_cont(ctx0, Q);
            K = ggml_cont(ctx0, K);
            V = ggml_cont(ctx0, V);
        }

        // Reshape to (hd, n_h, T)
        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);

        // No RoPE — GPT-2 uses learned position embeddings

        // Permute new K/V to (hd, T, n_h) for cache write
        ggml_tensor* K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor* V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Write into KV cache at [n_past, n_past+T)
        ggml_tensor* k_view = ggml_view_4d(ctx0, c->kv_k, hd, T, n_h, 1, c->kv_k->nb[1], c->kv_k->nb[2], c->kv_k->nb[3],
                                           (size_t)il * c->kv_k->nb[3] + (size_t)n_past * c->kv_k->nb[1]);
        ggml_tensor* v_view = ggml_view_4d(ctx0, c->kv_v, hd, T, n_h, 1, c->kv_v->nb[1], c->kv_v->nb[2], c->kv_v->nb[3],
                                           (size_t)il * c->kv_v->nb[3] + (size_t)n_past * c->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));

        // Read full K/V history
        ggml_tensor* k_layer_view =
            ggml_view_3d(ctx0, c->kv_k, hd, Lk, n_h, c->kv_k->nb[1], c->kv_k->nb[2], (size_t)il * c->kv_k->nb[3]);
        ggml_tensor* v_layer_view =
            ggml_view_3d(ctx0, c->kv_v, hd, Lk, n_h, c->kv_v->nb[1], c->kv_v->nb[2], (size_t)il * c->kv_v->nb[3]);
        ggml_tensor* Kfull = ggml_cont(ctx0, k_layer_view);
        ggml_tensor* Vfull = ggml_cont(ctx0, v_layer_view);

        // Permute Q to (hd, T, n_h)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        // Flash attention
        ggml_tensor* attn =
            ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, b.attn_proj_w, attn);
        attn = ggml_add(ctx0, attn, b.attn_proj_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);

        // GELU FFN: c_fc → gelu → c_proj
        ggml_tensor* mlp = ggml_mul_mat(ctx0, b.ffn_fc_w, x);
        mlp = ggml_add(ctx0, mlp, b.ffn_fc_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, b.ffn_proj_w, mlp);
        mlp = ggml_add(ctx0, mlp, b.ffn_proj_b);

        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, c->model.final_norm_w);
    cur = ggml_add(ctx0, cur, c->model.final_norm_b);

    // Take last token for prefill
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // Mel head
    cur = ggml_mul_mat(ctx0, c->model.mel_head_w, cur);
    if (c->model.mel_head_b) {
        cur = ggml_add(ctx0, cur, c->model.mel_head_b);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Run GPT-2 forward pass ───────────────────────────────────────

static float* run_gpt2_kv(indextts_context* c, const float* embeds, int n_tokens, int n_past) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "indextts: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const int D = (int)c->hp.d_model;
    const int vocab = (int)c->hp.mel_vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_gpt2_kv_graph(c, n_past, n_tokens);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "indextts: failed to alloc GPT-2 graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)D * n_tokens * sizeof(float));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "indextts: GPT-2 compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// ── GPT-2 latent extraction (return_latent pass) ────────────────
//
// Build a full-prefill graph that processes ALL tokens (conditioning +
// text + mel codes) in a single forward pass and returns the hidden
// states after final_norm for the mel code positions only. This is the
// "return_latent" second pass from IndexTTS Python: instead of going
// through mel_head to get logits, we extract the raw normalized hidden
// states that the BigVGAN vocoder needs.
//
// Returns a float vector of shape [n_mel_codes, d_model].

static ggml_cgraph* build_gpt2_latent_graph(indextts_context* c, int n_total, int n_mel_codes) {
    const auto& hp = c->hp;
    const int D = (int)hp.d_model;
    const int n_h = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_total;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "latent_embeds");
    ggml_set_input(embeds);

    // Full causal mask
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(causal_mask, "latent_mask");
    ggml_set_input(causal_mask);

    ggml_tensor* cur = embeds;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->model.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // Fused QKV
        ggml_tensor* qkv = ggml_mul_mat(ctx0, b.attn_qkv_w, x);
        qkv = ggml_add(ctx0, qkv, b.attn_qkv_b);

        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor* Q = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts);
        ggml_tensor* V = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts);
        Q = ggml_cont(ctx0, Q);
        K = ggml_cont(ctx0, K);
        V = ggml_cont(ctx0, V);

        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);

        // Permute to (hd, T, n_h) for flash attention
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, causal_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        attn = ggml_mul_mat(ctx0, b.attn_proj_w, attn);
        attn = ggml_add(ctx0, attn, b.attn_proj_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);

        ggml_tensor* mlp = ggml_mul_mat(ctx0, b.ffn_fc_w, x);
        mlp = ggml_add(ctx0, mlp, b.ffn_fc_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, b.ffn_proj_w, mlp);
        mlp = ggml_add(ctx0, mlp, b.ffn_proj_b);

        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, c->model.final_norm_w);
    cur = ggml_add(ctx0, cur, c->model.final_norm_b);

    // Extract hidden states for mel positions: start_mel through the second-to-last
    // mel code. Python returns mel_logits[:, :-2] on mel_emb which has (k+2) tokens
    // [start_mel, m1..mk, stop_mel], so the latent has k positions starting at start_mel.
    // These are the positions [start_mel, m1, ..., mk-1] — the hidden state at each
    // position predicts the next mel code, which the vocoder uses as its input.
    int mel_start = T - n_mel_codes - 1; // -1 to include start_mel position
    cur = ggml_view_2d(ctx0, cur, D, n_mel_codes, cur->nb[1], (size_t)mel_start * cur->nb[1]);
    cur = ggml_cont(ctx0, cur);

    ggml_set_name(cur, "latent_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the second GPT forward pass to extract hidden states for vocoder.
// Takes the full token sequence including the generated mel codes and returns
// normalized hidden states for the mel code positions.
// Returns a malloc'd float array of shape [n_mel_codes, d_model].
static float* run_gpt_latent(indextts_context* c, const std::vector<int32_t>& text_tokens,
                             const std::vector<int32_t>& mel_codes) {
    const int D = (int)c->hp.d_model;
    const int cond_len = (int)c->hp.perceiver_n_latents;
    // Python wraps text with start_text(0) + stop_text(1)
    const int text_len = (int)text_tokens.size() + 2; // +start_text +stop_text
    const int n_mel = (int)mel_codes.size();
    // Full sequence: [cond(32)] [start_text, t1..tn, stop_text] [start_mel, m1..mk]
    // (no stop_mel appended — Python strips it via :-2)
    const int total_len = cond_len + text_len + 1 + n_mel;
    // Latent: extract from start_mel position through mk-1 = n_mel positions
    // (hidden state at start_mel predicts m1, at m1 predicts m2, etc.)
    const int n_latent = n_mel; // matches Python's mel_emb.shape[1] - 2

    if (c->params.verbosity >= 1) {
        fprintf(stderr, "indextts: latent pass: total=%d (cond=%d text=%d mel=%d+1) n_latent=%d\n", total_len, cond_len,
                text_len, n_mel, n_latent);
    }

    // Build full embedding sequence
    std::vector<float> text_emb_table((size_t)c->hp.text_vocab_size * D);
    tensor_get_f32(c->model.text_emb_w, text_emb_table.data(), text_emb_table.size());

    std::vector<float> text_pos_table((size_t)c->hp.text_pos_size * D);
    tensor_get_f32(c->model.text_pos_emb_w, text_pos_table.data(), text_pos_table.size());

    std::vector<float> mel_emb_table((size_t)c->hp.mel_vocab_size * D);
    tensor_get_f32(c->model.mel_emb_w, mel_emb_table.data(), mel_emb_table.size());

    std::vector<float> mel_pos_table((size_t)c->hp.mel_pos_size * D);
    tensor_get_f32(c->model.mel_pos_emb_w, mel_pos_table.data(), mel_pos_table.size());

    std::vector<float> embeds((size_t)total_len * D, 0.0f);
    int pos = 0;

    // 1. Conditioning latents from Conformer+Perceiver (or zeros)
    if (!c->cond_latents.empty() && (int)c->cond_latents.size() == cond_len * D) {
        std::memcpy(embeds.data(), c->cond_latents.data(), (size_t)cond_len * D * sizeof(float));
    }
    pos += cond_len;

    // 2. Text embeddings: [start_text, t1..tn, stop_text]
    // Python: build_aligned_inputs_and_targets prepends start_text_token=0
    // and the text_inputs already has stop_text_token=1 appended via F.pad
    {
        int start_text = 0; // GPT config: start_text_token=0
        int stop_text = 1;  // GPT config: stop_text_token=1

        // Start text token
        int tpi = 0;
        for (int j = 0; j < D; j++)
            embeds[pos * D + j] = text_emb_table[(size_t)start_text * D + j] + text_pos_table[(size_t)tpi * D + j];
        pos++;

        // Text tokens
        for (int i = 0; i < (int)text_tokens.size(); i++) {
            int tok = text_tokens[i];
            if (tok < 0 || tok >= (int)c->hp.text_vocab_size)
                tok = 0;
            tpi = std::min(i + 1, (int)c->hp.text_pos_size - 1);
            for (int j = 0; j < D; j++)
                embeds[(pos + i) * D + j] = text_emb_table[(size_t)tok * D + j] + text_pos_table[(size_t)tpi * D + j];
        }
        pos += (int)text_tokens.size();

        // Stop text token
        tpi = std::min((int)text_tokens.size() + 1, (int)c->hp.text_pos_size - 1);
        for (int j = 0; j < D; j++)
            embeds[pos * D + j] = text_emb_table[(size_t)stop_text * D + j] + text_pos_table[(size_t)tpi * D + j];
        pos++;
    }

    // 3. Start mel token + mel pos 0
    {
        int start_tok = (int)c->hp.start_mel_token;
        if (start_tok >= (int)c->hp.mel_vocab_size)
            start_tok = 0;
        for (int j = 0; j < D; j++) {
            embeds[pos * D + j] = mel_emb_table[(size_t)start_tok * D + j] + mel_pos_table[j];
        }
        pos++;
    }

    // 4. Mel code tokens
    for (int i = 0; i < n_mel; i++) {
        int tok = mel_codes[i];
        if (tok < 0 || tok >= (int)c->hp.mel_vocab_size)
            tok = 0;
        int mpi = std::min(i + 1, (int)c->hp.mel_pos_size - 1);
        for (int j = 0; j < D; j++) {
            embeds[(pos + i) * D + j] = mel_emb_table[(size_t)tok * D + j] + mel_pos_table[(size_t)mpi * D + j];
        }
    }

    // Build causal mask
    std::vector<ggml_fp16_t> mask((size_t)total_len * total_len, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < total_len; q++) {
        for (int k = q + 1; k < total_len; k++) {
            mask[(size_t)q * total_len + k] = neg_inf;
        }
    }

    // Build and run the latent extraction graph (no KV cache — single prefill)
    ggml_cgraph* gf = build_gpt2_latent_graph(c, total_len, n_latent);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "indextts: failed to alloc latent graph\n");
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "latent_embeds"), embeds.data(), 0,
                            (size_t)total_len * D * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "latent_mask"), mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "indextts: latent compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "latent_out");
    if (!out) {
        fprintf(stderr, "indextts: latent_out tensor not found\n");
        return nullptr;
    }

    float* result = (float*)malloc((size_t)n_latent * D * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)n_latent * D * sizeof(float));

    // Compute latent RMS for debugging
    float sum2 = 0;
    for (int i = 0; i < n_latent * D; i++)
        sum2 += result[i] * result[i];
    float lat_rms = sqrtf(sum2 / (float)(n_latent * D));

    if (c->params.verbosity >= 1) {
        // Per-position RMS to check if later positions have speech structure
        float rms_first = 0, rms_last = 0;
        for (int j = 0; j < D; j++) {
            rms_first += result[j] * result[j];
            rms_last += result[(n_latent - 1) * D + j] * result[(n_latent - 1) * D + j];
        }
        rms_first = sqrtf(rms_first / D);
        rms_last = sqrtf(rms_last / D);
        fprintf(stderr, "indextts: latent: [%d, %d] rms=%.4f pos0_rms=%.4f posN_rms=%.4f\n", n_latent, D, lat_rms,
                rms_first, rms_last);
    }

    return result;
}

// ── Build prefill embeddings ─────────────────────────────────────

// Build the embedding sequence: [cond_latents(32)] [text_embs+text_pos] [start_mel+mel_pos_0]
// For Phase 1, cond_latents are dummy zeros.
static std::vector<float> build_prefill_embeds(indextts_context* c, const std::vector<int32_t>& text_tokens) {
    const int D = (int)c->hp.d_model;
    const auto& m = c->model;

    // Read embedding tables
    std::vector<float> text_emb_table((size_t)c->hp.text_vocab_size * D);
    tensor_get_f32(m.text_emb_w, text_emb_table.data(), text_emb_table.size());

    std::vector<float> text_pos_table((size_t)c->hp.text_pos_size * D);
    tensor_get_f32(m.text_pos_emb_w, text_pos_table.data(), text_pos_table.size());

    std::vector<float> mel_emb_table((size_t)c->hp.mel_vocab_size * D);
    tensor_get_f32(m.mel_emb_w, mel_emb_table.data(), mel_emb_table.size());

    std::vector<float> mel_pos_table((size_t)c->hp.mel_pos_size * D);
    tensor_get_f32(m.mel_pos_emb_w, mel_pos_table.data(), mel_pos_table.size());

    const int cond_len = (int)c->hp.perceiver_n_latents;
    const int text_len = (int)text_tokens.size();
    const int total_len = cond_len + text_len + 2 + 1; // +2 for start/stop_text, +1 for start_mel

    std::vector<float> embeds((size_t)total_len * D, 0.0f);

    int pos = 0;

    // 1. Conditioning latents from Conformer+Perceiver (or zeros if not computed)
    if (!c->cond_latents.empty() && (int)c->cond_latents.size() == cond_len * D) {
        // cond_latents is [D, n_latents] in ggml layout (ne[0]=D, ne[1]=n_latents)
        // i.e. flat: latent[i][j] = cond_latents[j + i*D] — already row-major per-latent
        // embeds wants [total_len, D] row-major, so embeds[pos*D..pos*D+D-1] = latent[pos]
        // ggml [D, N] memory: element(d, n) = data[d + n*D]
        // So latent n is at cond_latents[n*D .. n*D+D-1] — matches row-major per-latent
        std::memcpy(embeds.data(), c->cond_latents.data(), (size_t)cond_len * D * sizeof(float));
    }
    pos += cond_len;

    // 2. Text embeddings: [start_text(0), t1..tn, stop_text(1)]
    // Matches Python's build_aligned_inputs_and_targets + F.pad(stop_text)
    {
        int start_text = 0, stop_text = 1;

        // Start text
        for (int j = 0; j < D; j++)
            embeds[pos * D + j] = text_emb_table[(size_t)start_text * D + j] + text_pos_table[j];
        pos++;

        // Text tokens
        for (int i = 0; i < text_len; i++) {
            int tok = text_tokens[i];
            if (tok < 0 || tok >= (int)c->hp.text_vocab_size)
                tok = 0;
            int tpi = std::min(i + 1, (int)c->hp.text_pos_size - 1);
            for (int j = 0; j < D; j++)
                embeds[(pos + i) * D + j] = text_emb_table[(size_t)tok * D + j] + text_pos_table[(size_t)tpi * D + j];
        }
        pos += text_len;

        // Stop text
        int tpi = std::min(text_len + 1, (int)c->hp.text_pos_size - 1);
        for (int j = 0; j < D; j++)
            embeds[pos * D + j] = text_emb_table[(size_t)stop_text * D + j] + text_pos_table[(size_t)tpi * D + j];
        pos++;
    }

    // 3. Start mel token: mel_emb(start_mel) + mel_pos_emb(0)
    {
        int start_tok = (int)c->hp.start_mel_token;
        if (start_tok >= (int)c->hp.mel_vocab_size) {
            start_tok = 0;
        }
        for (int j = 0; j < D; j++) {
            embeds[pos * D + j] = mel_emb_table[(size_t)start_tok * D + j] + mel_pos_table[j];
        }
    }

    return embeds;
}

// Build embedding for a single mel token at a given mel position.
static std::vector<float> build_mel_token_embed(indextts_context* c, int32_t token_id, int mel_pos) {
    const int D = (int)c->hp.d_model;
    std::vector<float> embed(D);

    std::vector<float> mel_emb_table((size_t)c->hp.mel_vocab_size * D);
    tensor_get_f32(c->model.mel_emb_w, mel_emb_table.data(), mel_emb_table.size());

    std::vector<float> mel_pos_table((size_t)c->hp.mel_pos_size * D);
    tensor_get_f32(c->model.mel_pos_emb_w, mel_pos_table.data(), mel_pos_table.size());

    int tid = (token_id >= 0 && token_id < (int)c->hp.mel_vocab_size) ? token_id : 0;
    int pid = (mel_pos >= 0 && mel_pos < (int)c->hp.mel_pos_size) ? mel_pos : 0;
    for (int j = 0; j < D; j++) {
        embed[j] = mel_emb_table[(size_t)tid * D + j] + mel_pos_table[(size_t)pid * D + j];
    }
    return embed;
}

} // namespace

// ── Public C ABI ────────────────────────────────────────────────

extern "C" struct indextts_context_params indextts_context_default_params(void) {
    indextts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.8f;
    p.top_p = 0.9f;
    p.top_k = 50;
    p.repetition_penalty = 1.2f;
    p.max_mel_tokens = 1500;
    return p;
}

extern "C" struct indextts_context* indextts_init_from_file(const char* path_model,
                                                            struct indextts_context_params params) {
    auto* c = new indextts_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: metadata
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
        fprintf(stderr, "indextts: GPT-2 %uL d=%u h=%u hd=%u ff=%u text_vocab=%u mel_vocab=%u\n", c->hp.n_layers,
                c->hp.d_model, c->hp.n_heads, c->hp.head_dim, c->hp.ff_dim, c->hp.text_vocab_size,
                c->hp.mel_vocab_size);
        fprintf(stderr, "indextts: tokenizer %s (%zu tokens)\n",
                c->tokenizer.loaded ? "loaded from GGUF" : "fallback (byte-level)", c->tokenizer.id_to_token.size());
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "indextts: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        if (params.verbosity >= 1 && params.use_gpu) {
            fprintf(stderr, "indextts: GPU backend unavailable, falling back to CPU\n");
        }
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, c->backend, "indextts", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    // Bind tensors
    if (!bind_model(c)) {
        fprintf(stderr, "indextts: failed to bind model tensors\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend != c->backend_cpu) {
            backends[n_be++] = c->backend_cpu;
        }
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    return c;
}

extern "C" int indextts_set_vocoder_path(struct indextts_context* ctx, const char* path) {
    if (!ctx || !path) {
        return -1;
    }
    ctx->vocoder_path = path;

    // Free existing vocoder if reloading
    if (ctx->voc) {
        indextts_voc_free(ctx->voc);
        ctx->voc = nullptr;
    }

    ctx->voc = indextts_voc_init(path, ctx->n_threads, ctx->params.use_gpu);
    if (!ctx->voc) {
        fprintf(stderr, "indextts: failed to load BigVGAN vocoder from '%s'\n", path);
        return -1;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: BigVGAN vocoder loaded from '%s'\n", path);
    }
    return 0;
}

extern "C" int32_t* indextts_generate_mel_codes(struct indextts_context* ctx, const char* text, const float* ref_pcm,
                                                int ref_n_samples, int* out_n) {
    if (!ctx || !text || !out_n) {
        return nullptr;
    }
    *out_n = 0;

    // Run conditioning pipeline if reference audio is provided
    if (ref_pcm && ref_n_samples > 0) {
        if (!run_conditioning(ctx, ref_pcm, ref_n_samples)) {
            fprintf(stderr, "indextts: conditioning pipeline failed\n");
        }
    } else {
        // No reference audio — use zero conditioning
        const int D = (int)ctx->hp.d_model;
        const int n_lat = (int)ctx->hp.perceiver_n_latents;
        ctx->cond_latents.assign((size_t)n_lat * D, 0.0f);
    }

    // 1. Tokenize text
    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.loaded) {
        text_tokens = tokenize_bpe(ctx->tokenizer, text);
    } else {
        text_tokens = tokenize_fallback(ctx->tokenizer, text, ctx->hp.text_vocab_size);
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: text \"%s\" -> %zu tokens\n", text, text_tokens.size());
    }

    // 2. Allocate KV cache
    const int max_mel = ctx->params.max_mel_tokens;
    const int cond_len = (int)ctx->hp.perceiver_n_latents;
    int max_ctx = cond_len + (int)text_tokens.size() + 1 + max_mel + 16;
    if (!kv_alloc(ctx, max_ctx)) {
        fprintf(stderr, "indextts: failed to allocate KV cache\n");
        return nullptr;
    }

    // 3. Build prefill embeddings
    std::vector<float> prefill_embeds = build_prefill_embeds(ctx, text_tokens);
    int prefill_len = (int)(prefill_embeds.size() / ctx->hp.d_model);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: prefill %d tokens (cond=%d text=%zu start_mel=1) max_mel=%d\n", prefill_len,
                cond_len, text_tokens.size(), max_mel);
    }

    // 4. Run prefill
    int n_past = 0;
    float* logits = run_gpt2_kv(ctx, prefill_embeds.data(), prefill_len, n_past);
    if (!logits) {
        fprintf(stderr, "indextts: prefill failed\n");
        return nullptr;
    }
    n_past += prefill_len;

    // Debug: compare prefill logits against Python reference
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: prefill logits[0:5] = [%.3f, %.3f, %.3f, %.3f, %.3f]\n", logits[0], logits[1],
                logits[2], logits[3], logits[4]);
        // Find top-5
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < (int)ctx->hp.mel_vocab_size; i++)
            scored.push_back({logits[i], i});
        std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        fprintf(stderr, "indextts: prefill top5:");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, " %d(%.3f)", scored[i].second, scored[i].first);
        fprintf(stderr, "\n");
    }

    // 5. AR decode loop
    std::vector<int32_t> mel_codes;
    const int mel_vocab = (int)ctx->hp.mel_vocab_size;
    const int stop_token = (int)ctx->hp.stop_mel_token;
    std::vector<int32_t> recent_tokens;

    for (int step = 0; step < max_mel; step++) {
        // Sample from logits
        int32_t next_tok =
            sample_top_k_top_p(logits, mel_vocab, ctx->params.temperature, ctx->params.top_k, ctx->params.top_p,
                               ctx->params.repetition_penalty, recent_tokens, ctx->rng_state);
        free(logits);
        logits = nullptr;

        if (next_tok == stop_token) {
            if (ctx->params.verbosity >= 1) {
                fprintf(stderr, "indextts: stop token at step %d\n", step);
            }
            break;
        }

        mel_codes.push_back(next_tok);
        recent_tokens.push_back(next_tok);
        // Keep recent tokens window for rep penalty
        if (recent_tokens.size() > 64) {
            recent_tokens.erase(recent_tokens.begin());
        }

        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "  step %d: mel code %d\n", step, (int)next_tok);
        }

        // Build embedding for next token
        // mel_pos starts at 1 (0 was used for start_mel in prefill)
        std::vector<float> tok_embed = build_mel_token_embed(ctx, next_tok, step + 1);

        // Run single-token decode
        logits = run_gpt2_kv(ctx, tok_embed.data(), 1, n_past);
        if (!logits) {
            fprintf(stderr, "indextts: decode step %d failed\n", step);
            break;
        }
        n_past += 1;
    }

    if (logits) {
        free(logits);
    }

    if (mel_codes.empty()) {
        fprintf(stderr, "indextts: no mel codes generated\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: generated %zu mel codes\n", mel_codes.size());
    }

    // Return a copy
    *out_n = (int)mel_codes.size();
    int32_t* result = (int32_t*)malloc(mel_codes.size() * sizeof(int32_t));
    std::memcpy(result, mel_codes.data(), mel_codes.size() * sizeof(int32_t));
    return result;
}

extern "C" float* indextts_synthesize(struct indextts_context* ctx, const char* text, const float* ref_pcm,
                                      int ref_n_samples, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) {
        return nullptr;
    }
    *out_n_samples = 0;

    // Step 1: Generate mel codes via AR decode
    int n_codes = 0;
    int32_t* codes = indextts_generate_mel_codes(ctx, text, ref_pcm, ref_n_samples, &n_codes);
    if (!codes || n_codes == 0) {
        return nullptr;
    }

    // Debug: override mel codes from file if INDEXTTS_MEL_CODES_FILE is set
    const char* mc_file = getenv("INDEXTTS_MEL_CODES_FILE");
    if (mc_file && mc_file[0]) {
        FILE* f = fopen(mc_file, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            n_codes = (int)(sz / sizeof(int32_t));
            free(codes);
            codes = (int32_t*)malloc(sz);
            fread(codes, 1, sz, f);
            fclose(f);
            fprintf(stderr, "indextts: DEBUG loaded %d mel codes from %s\n", n_codes, mc_file);
        }
    }

    // Check if vocoder is loaded
    if (!ctx->voc) {
        fprintf(stderr,
                "indextts: synthesize() generated %d mel codes but no BigVGAN vocoder loaded.\n"
                "         Pass --codec-model PATH or place indextts-bigvgan.gguf next to the GPT model.\n",
                n_codes);
        indextts_codes_free(codes);
        return nullptr;
    }

    // Step 2: Re-tokenize text to rebuild the full sequence for latent pass
    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.loaded) {
        text_tokens = tokenize_bpe(ctx->tokenizer, text);
    } else {
        text_tokens = tokenize_fallback(ctx->tokenizer, text, ctx->hp.text_vocab_size);
    }

    std::vector<int32_t> mel_codes_vec(codes, codes + n_codes);
    indextts_codes_free(codes);
    codes = nullptr;

    // Step 3: Run GPT latent extraction (second forward pass)
    float* latent = run_gpt_latent(ctx, text_tokens, mel_codes_vec);
    if (!latent) {
        fprintf(stderr, "indextts: latent extraction failed\n");
        return nullptr;
    }

    // Step 4: Compute speaker embedding from reference audio via ECAPA-TDNN
    // ECAPA-TDNN speaker embedding for vocoder conditioning.
    // NOTE: currently disabled — the ECAPA output has 3.4x the norm of
    // Python's reference (2.89 vs 0.85), which overwhelms the vocoder.
    // The Conformer+Perceiver conditioning for the GPT still works.
    // TODO: debug ECAPA BatchNorm / output scaling to match Python.
    float* spk_emb = nullptr;
    (void)ref_pcm;
    (void)ref_n_samples;

    // Debug: override latent from file if INDEXTTS_LATENT_FILE is set
    const char* lat_file = getenv("INDEXTTS_LATENT_FILE");
    if (lat_file && lat_file[0]) {
        FILE* f = fopen(lat_file, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            int n_elem = (int)(sz / sizeof(float));
            int D = 1280;
            int T = n_elem / D;
            free(latent);
            n_codes = T;
            latent = (float*)malloc(sz);
            fread(latent, 1, sz, f);
            fclose(f);
            fprintf(stderr, "indextts: DEBUG loaded latent from %s: [%d, %d]\n", lat_file, T, D);
        }
    }

    // Step 5: Run BigVGAN vocoder
    // latent is [n_codes, 1280] — the vocoder expects this as-is.
    int n_audio = 0;
    float* pcm = indextts_voc_generate(ctx->voc, latent, n_codes, spk_emb, &n_audio);
    free(latent);
    free(spk_emb);

    if (!pcm || n_audio <= 0) {
        fprintf(stderr, "indextts: BigVGAN vocoder failed\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "indextts: synthesized %d samples (%.2f sec @ 24kHz)\n", n_audio, (float)n_audio / 24000.0f);
    }

    *out_n_samples = n_audio;
    return pcm;
}

extern "C" void indextts_codes_free(int32_t* codes) {
    free(codes);
}

extern "C" void indextts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void indextts_free(struct indextts_context* ctx) {
    delete ctx;
}

extern "C" void indextts_set_n_threads(struct indextts_context* ctx, int n_threads) {
    if (ctx) {
        ctx->n_threads = n_threads > 0 ? n_threads : 4;
    }
}
