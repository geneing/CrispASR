// chatterbox_s3gen.cpp — S3Gen (flow matching) + HiFTGenerator vocoder.
//
// This file implements the second and third stages of the Chatterbox pipeline:
//   Stage 2: Speech tokens → mel-spectrogram via conditional flow matching
//   Stage 3: Mel → 24 kHz waveform via HiFT-GAN vocoder
//
// Architecture (from chatterbox/models/s3gen/):
//   - UpsampleConformerEncoder: 6 pre-upsample + 4 post-upsample conformer
//     blocks with relative positional self-attention (512D, 8 heads, 2048 FFN)
//   - ConditionalDecoder: UNet1D with causal conv1d, 1 down + 12 mid + 1 up
//     blocks, each containing CausalResnetBlock1D + 4 BasicTransformerBlocks
//   - CausalConditionalCFM: Euler ODE solver, 10 steps, cosine t-schedule
//   - HiFTGenerator: F0 prediction → SineGen → ConvTranspose1D chain → iSTFT
//
// Weight loading: reads from chatterbox-s3gen-f16.gguf produced by
// models/convert-chatterbox-to-gguf.py. All tensor names are prefixed
// with "s3." matching the converter's map_s3gen_name().

#include "chatterbox_s3gen.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Context ──────────────────────────────────────────────────────

struct chatterbox_s3gen_context {
    int n_threads = 4;
    int verbosity = 1;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ~chatterbox_s3gen_context() {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx_w) ggml_free(ctx_w);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu) ggml_backend_free(backend);
        if (backend_cpu) ggml_backend_free(backend_cpu);
    }
};

// ── Tensor lookup helper ─────────────────────────────────────────

static ggml_tensor* T(chatterbox_s3gen_context* c, const char* name) {
    return core_gguf::try_get(c->tensors, name);
}

static ggml_tensor* TR(chatterbox_s3gen_context* c, const char* name) {
    return core_gguf::require(c->tensors, name, "s3gen");
}

// ── Public API ──────────────────────────────────────────────────

extern "C" struct chatterbox_s3gen_context* chatterbox_s3gen_init_from_file(
    const char* path, int n_threads, int verbosity
) {
    auto* c = new chatterbox_s3gen_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;
    c->verbosity = verbosity;

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "s3gen: failed to init CPU backend\n");
        delete c; return nullptr;
    }
    c->backend = c->backend_cpu;

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "s3gen", wl)) {
        delete c; return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (verbosity >= 1) {
        fprintf(stderr, "s3gen: loaded %zu tensors from %s\n", c->tensors.size(), path);
    }

    // Verify critical tensors exist
    if (!TR(c, "s3.flow.input_embedding.weight") ||
        !TR(c, "s3.flow.encoder_proj.weight") ||
        !TR(c, "s3.flow.spk_embed_affine_layer.weight")) {
        fprintf(stderr, "s3gen: missing critical tensors\n");
        delete c; return nullptr;
    }

    // Scheduler
    {
        ggml_backend_t backends[] = { c->backend };
        c->sched = ggml_backend_sched_new(backends, nullptr, 1, 32768, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    }

    return c;
}

// ── Conformer encoder forward (CPU fallback) ────────────────────
//
// The conformer encoder is complex (relative positional attention,
// feedforward, pre-lookahead conv). For the initial implementation,
// we run it on CPU using direct weight reads, similar to how the T3
// prefill embeddings are built.
//
// This is simpler than building a full ggml graph for the conformer
// but slower. A ggml graph version is a follow-up optimization.

static std::vector<float> run_conformer_encoder_cpu(
    chatterbox_s3gen_context* c,
    const int32_t* speech_tokens, int n_tokens,
    const int32_t* prompt_tokens, int n_prompt
) {
    const int D = 512;  // encoder hidden dim
    const int total = n_prompt + n_tokens;

    // 1. Embed all tokens: flow.input_embedding (6561, 512)
    ggml_tensor* emb_w = TR(c, "s3.flow.input_embedding.weight");
    std::vector<float> emb_table(6561 * D);
    ggml_backend_tensor_get(emb_w, emb_table.data(), 0, emb_table.size() * sizeof(float));

    std::vector<float> input(total * D, 0.0f);
    // Prompt tokens first, then speech tokens
    for (int i = 0; i < n_prompt; i++) {
        int tok = prompt_tokens[i];
        if (tok >= 0 && tok < 6561) {
            std::memcpy(&input[i * D], &emb_table[tok * D], D * sizeof(float));
        }
    }
    for (int i = 0; i < n_tokens; i++) {
        int tok = speech_tokens[i];
        if (tok >= 0 && tok < 6561) {
            std::memcpy(&input[(n_prompt + i) * D], &emb_table[tok * D], D * sizeof(float));
        }
    }

    // 2. Linear embed layer: embed.out.0 (Linear 512→512) + embed.out.1 (LayerNorm)
    // For simplicity in this initial version, we skip the full conformer
    // forward and just do: embed → linear → project to 80D.
    // This won't produce correct results but lets us test the pipeline.
    // TODO: implement full conformer forward with relative attention.

    // Linear: embed.out.0 (512, 512)
    ggml_tensor* lin_w = T(c, "s3.fe.embed.out.0.weight");
    ggml_tensor* lin_b = T(c, "s3.fe.embed.out.0.bias");
    if (lin_w && lin_b) {
        std::vector<float> w(D * D);
        std::vector<float> b(D);
        ggml_backend_tensor_get(lin_w, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_get(lin_b, b.data(), 0, b.size() * sizeof(float));

        std::vector<float> out(total * D, 0.0f);
        for (int t = 0; t < total; t++) {
            for (int i = 0; i < D; i++) {
                float sum = b[i];
                for (int j = 0; j < D; j++) {
                    sum += w[i * D + j] * input[t * D + j];
                }
                out[t * D + i] = sum;
            }
        }
        input = std::move(out);
    }

    // 3. Upsample 2x: nearest-neighbor interpolation + conv
    int up_len = total * 2;
    std::vector<float> upsampled(up_len * D, 0.0f);
    for (int t = 0; t < total; t++) {
        std::memcpy(&upsampled[(t * 2) * D], &input[t * D], D * sizeof(float));
        std::memcpy(&upsampled[(t * 2 + 1) * D], &input[t * D], D * sizeof(float));
    }

    // 4. Project encoder output to 80D: encoder_proj (80, 512)
    ggml_tensor* proj_w = TR(c, "s3.flow.encoder_proj.weight");
    ggml_tensor* proj_b = T(c, "s3.flow.encoder_proj.bias");
    std::vector<float> pw(80 * D);
    std::vector<float> pb(80, 0.0f);
    ggml_backend_tensor_get(proj_w, pw.data(), 0, pw.size() * sizeof(float));
    if (proj_b) ggml_backend_tensor_get(proj_b, pb.data(), 0, pb.size() * sizeof(float));

    // Output: (up_len, 80) stored as (80, up_len) for channel-first layout
    std::vector<float> h(80 * up_len, 0.0f);
    for (int t = 0; t < up_len; t++) {
        for (int i = 0; i < 80; i++) {
            float sum = pb[i];
            for (int j = 0; j < D; j++) {
                sum += pw[i * D + j] * upsampled[t * D + j];
            }
            h[i * up_len + t] = sum;
        }
    }

    return h; // (80, up_len) channel-first
}

// ── Sinusoidal positional embedding ──────────────────────────────

static std::vector<float> sinusoidal_embedding(float t_val, int dim) {
    // Same as SinusoidalPosEmb in matcha/decoder.py
    std::vector<float> emb(dim);
    int half = dim / 2;
    float log_term = std::log(10000.0f) / (float)(half - 1);
    for (int i = 0; i < half; i++) {
        float freq = std::exp(-(float)i * log_term);
        emb[i] = std::sin(t_val * freq);
        emb[half + i] = std::cos(t_val * freq);
    }
    return emb;
}

// ── CFM Euler solver (CPU) ──────────────────────────────────────
//
// For the initial implementation, the denoiser UNet1D forward is a
// stub that returns the mu (encoder output) directly. This produces
// a rough approximation that at least lets the vocoder run.
// TODO: implement full UNet1D denoiser forward.

static std::vector<float> cfm_euler_solve(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mu,       // (80, T) encoder output
    const std::vector<float>& cond,     // (80, T) conditioning (prompt mel + zeros)
    const std::vector<float>& spk_emb,  // (80,) projected speaker embedding
    int T_mel,
    int n_steps,
    float cfg_rate
) {
    // Generate cosine time schedule
    std::vector<float> t_span(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float t = (float)i / (float)n_steps;
        t_span[i] = 1.0f - std::cos(t * 0.5f * (float)M_PI);
    }

    // Start from noise
    std::vector<float> x(80 * T_mel);
    // Simple noise (deterministic seed for reproducibility)
    uint64_t rng = 42;
    for (size_t i = 0; i < x.size(); i++) {
        // Box-Muller transform for normal distribution
        float u1 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        float u2 = (float)((rng = rng * 6364136223846793005ULL + 1) >> 33) / (float)(1ULL << 31);
        if (u1 < 1e-7f) u1 = 1e-7f;
        x[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
    }

    // Euler steps
    for (int step = 0; step < n_steps; step++) {
        float t = t_span[step];
        float r = t_span[step + 1];
        float dt = r - t;

        // TODO: run the actual UNet1D denoiser here.
        // For now, use a simple linear interpolation as placeholder:
        // velocity ≈ (mu - x) which drives x toward mu
        for (size_t i = 0; i < x.size(); i++) {
            float velocity = mu[i] - x[i];
            x[i] += dt * velocity;
        }
    }

    return x; // (80, T_mel)
}

// ── HiFTGenerator vocoder (CPU stub) ────────────────────────────
//
// The full HiFTGenerator has:
//   1. F0 predictor (ConvRNN, 5 conv layers + linear classifier)
//   2. SineGen (harmonic source from F0)
//   3. ConvTranspose1D upsampling (rates 8,5,3 = 120x total)
//   4. Snake activation + ResBlocks
//   5. iSTFT for final waveform
//
// For the initial stub, we use Griffin-Lim as a simple mel→wav
// approximation. The proper HiFTGenerator implementation is a
// follow-up.

static std::vector<float> hift_vocoder_cpu(
    chatterbox_s3gen_context* c,
    const std::vector<float>& mel, // (80, T_mel) channel-first
    int T_mel
) {
    // Mel to waveform via simple Griffin-Lim approximation.
    // The actual HiFTGenerator uses F0-conditioned iSTFT — this is
    // a placeholder that produces audible (but low quality) output.

    const int sample_rate = 24000;
    const int hop_length = 480; // 24000 / 50 Hz mel frame rate
    const int n_samples = T_mel * hop_length;

    std::vector<float> wav(n_samples, 0.0f);

    // Very simple: treat mel as energy envelope, generate noise shaped by it
    for (int t = 0; t < T_mel; t++) {
        // Compute energy from mel bands
        float energy = 0.0f;
        for (int b = 0; b < 80; b++) {
            float m = mel[b * T_mel + t];
            energy += std::exp(m); // mel is in log scale
        }
        energy = std::sqrt(energy / 80.0f) * 0.1f;

        // Fill hop with shaped noise
        for (int s = 0; s < hop_length && (t * hop_length + s) < n_samples; s++) {
            float phase = (float)(t * hop_length + s) / (float)sample_rate;
            // Mix harmonics at common speech frequencies
            wav[t * hop_length + s] = energy * (
                0.5f * std::sin(2.0f * (float)M_PI * 150.0f * phase) +
                0.3f * std::sin(2.0f * (float)M_PI * 300.0f * phase) +
                0.2f * std::sin(2.0f * (float)M_PI * 450.0f * phase)
            );
        }
    }

    return wav;
}

// ── Full pipeline ───────────────────────────────────────────────

extern "C" float* chatterbox_s3gen_synthesize(
    struct chatterbox_s3gen_context* ctx,
    const int32_t* speech_tokens, int n_speech_tokens,
    const int32_t* prompt_tokens, int n_prompt_tokens,
    const float* prompt_feat, int prompt_feat_len,
    const float* spk_embedding,
    int n_cfm_steps,
    int* out_n_samples
) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (n_cfm_steps <= 0) n_cfm_steps = 10;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: %d speech tokens + %d prompt tokens, %d CFM steps\n",
                n_speech_tokens, n_prompt_tokens, n_cfm_steps);
    }

    // 1. Conformer encoder: tokens → (80, T_mel)
    std::vector<float> h = run_conformer_encoder_cpu(
        ctx, speech_tokens, n_speech_tokens,
        prompt_tokens, n_prompt_tokens);

    int T_mel_total = (n_prompt_tokens + n_speech_tokens) * 2; // 2x upsample
    int T_mel_prompt = n_prompt_tokens * 2;
    int T_mel_gen = n_speech_tokens * 2;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: encoder output T_mel=%d (prompt=%d, gen=%d)\n",
                T_mel_total, T_mel_prompt, T_mel_gen);
    }

    // 2. Build conditioning: prompt mel + zeros for generation region
    std::vector<float> cond(80 * T_mel_total, 0.0f);
    if (prompt_feat && prompt_feat_len > 0) {
        int copy_len = std::min(prompt_feat_len, T_mel_prompt);
        // prompt_feat is (T, 80) row-major, convert to (80, T) channel-first
        for (int t = 0; t < copy_len; t++) {
            for (int b = 0; b < 80; b++) {
                cond[b * T_mel_total + t] = prompt_feat[t * 80 + b];
            }
        }
    }

    // 3. Project speaker embedding: spk_embed_affine_layer (80, 192)
    std::vector<float> spk_proj(80, 0.0f);
    if (spk_embedding) {
        ggml_tensor* spk_w = TR(ctx, "s3.flow.spk_embed_affine_layer.weight");
        ggml_tensor* spk_b = T(ctx, "s3.flow.spk_embed_affine_layer.bias");
        std::vector<float> sw(80 * 192);
        std::vector<float> sb(80, 0.0f);
        ggml_backend_tensor_get(spk_w, sw.data(), 0, sw.size() * sizeof(float));
        if (spk_b) ggml_backend_tensor_get(spk_b, sb.data(), 0, sb.size() * sizeof(float));

        // Normalize embedding (L2 norm)
        float norm = 0.0f;
        for (int i = 0; i < 192; i++) norm += spk_embedding[i] * spk_embedding[i];
        norm = std::sqrt(norm + 1e-12f);

        for (int i = 0; i < 80; i++) {
            float sum = sb[i];
            for (int j = 0; j < 192; j++) {
                sum += sw[i * 192 + j] * (spk_embedding[j] / norm);
            }
            spk_proj[i] = sum;
        }
    }

    // 4. CFM Euler solver: noise → mel
    std::vector<float> mel = cfm_euler_solve(
        ctx, h, cond, spk_proj, T_mel_total, n_cfm_steps, 0.7f);

    // 5. Extract generated portion (skip prompt region)
    std::vector<float> gen_mel(80 * T_mel_gen);
    for (int b = 0; b < 80; b++) {
        std::memcpy(&gen_mel[b * T_mel_gen],
                     &mel[b * T_mel_total + T_mel_prompt],
                     T_mel_gen * sizeof(float));
    }

    // 6. Vocoder: mel → waveform
    std::vector<float> wav = hift_vocoder_cpu(ctx, gen_mel, T_mel_gen);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "s3gen: generated %zu samples (%.2f sec @ 24kHz)\n",
                wav.size(), (float)wav.size() / 24000.0f);
    }

    // Copy to malloc'd buffer
    float* out = (float*)malloc(wav.size() * sizeof(float));
    if (!out) return nullptr;
    std::memcpy(out, wav.data(), wav.size() * sizeof(float));
    *out_n_samples = (int)wav.size();
    return out;
}

extern "C" void chatterbox_s3gen_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void chatterbox_s3gen_free(struct chatterbox_s3gen_context* ctx) {
    delete ctx;
}
