#pragma once

// Chatterbox public C ABI.
//
// ResembleAI/chatterbox (MIT) — a multi-stage TTS pipeline:
//   1. T3 (520M Llama AR) — text + conditioning → speech tokens @25 Hz
//   2. S3Gen (CFM) — speech tokens → mel-spectrogram via flow matching
//   3. HiFTGenerator — mel → 24 kHz waveform
//
// Voice cloning uses a reference WAV processed through:
//   - VoiceEncoder (3-layer LSTM) → 256D speaker embedding for T3
//   - S3Tokenizer → speech tokens for T3 prompt conditioning
//   - CAMPPlus → 192D x-vector for S3Gen speaker conditioning
//
// The default built-in voice uses precomputed conditioning baked into
// the T3 GGUF (from conds.pt), so no reference audio is needed for
// basic synthesis.
//
// Two GGUFs are required:
//   - chatterbox-t3-f16.gguf   (T3 model + VE + tokenizer + conds)
//   - chatterbox-s3gen-f16.gguf (S3Gen flow + vocoder + CAMPPlus)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct chatterbox_context;

struct chatterbox_context_params {
    int n_threads;
    int verbosity;        // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;    // AR sampling temperature (default 0.8)
    float cfg_weight;     // classifier-free guidance weight (default 0.5)
    float exaggeration;   // emotion exaggeration factor (default 0.5)
    float repetition_penalty; // repetition penalty (default 1.2)
    float min_p;          // min_p sampling (default 0.05)
    float top_p;          // top_p sampling (default 1.0)
    int max_speech_tokens;// upper bound on T3 AR decode (default 1000)
    int cfm_steps;        // number of CFM Euler steps (default 10)
};

struct chatterbox_context_params chatterbox_context_default_params(void);

// Initialise from the T3 GGUF (arch="chatterbox", produced by
// models/convert-chatterbox-to-gguf.py).
struct chatterbox_context* chatterbox_init_from_file(
    const char* path_model,
    struct chatterbox_context_params params);

// Point the runtime at the S3Gen GGUF (arch="chatterbox-s3gen").
// Required before the first chatterbox_synthesize call. Returns 0
// on success.
int chatterbox_set_s3gen_path(struct chatterbox_context* ctx, const char* path);

// Synthesise text → 24 kHz mono float32 PCM using the built-in voice.
// Caller frees with chatterbox_pcm_free. *out_n_samples is set on
// success. Returns nullptr on error.
float* chatterbox_synthesize(
    struct chatterbox_context* ctx,
    const char* text,
    int* out_n_samples);

// Run only the T3 stage: text → speech tokens. Caller frees with
// chatterbox_tokens_free. *out_n is set to token count.
int32_t* chatterbox_synthesize_tokens(
    struct chatterbox_context* ctx,
    const char* text,
    int* out_n);

// Set voice from a reference WAV path for voice cloning.
// Requires VE (in T3 GGUF) + S3Tokenizer + CAMPPlus (in S3Gen GGUF).
// Returns 0 on success.
int chatterbox_set_voice_from_wav(
    struct chatterbox_context* ctx,
    const char* wav_path);

// Set the emotion exaggeration factor (0.0–2.0).
void chatterbox_set_exaggeration(struct chatterbox_context* ctx, float exaggeration);

// Set classifier-free guidance weight.
void chatterbox_set_cfg_weight(struct chatterbox_context* ctx, float cfg_weight);

// Set number of CFM denoising steps (1–100).
void chatterbox_set_cfm_steps(struct chatterbox_context* ctx, int steps);

void chatterbox_tokens_free(int32_t* tokens);
void chatterbox_pcm_free(float* pcm);

void chatterbox_free(struct chatterbox_context* ctx);
void chatterbox_set_n_threads(struct chatterbox_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
