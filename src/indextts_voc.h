#pragma once

// indextts_voc.h -- BigVGAN vocoder for IndexTTS-1.5.
//
// Converts GPT hidden states (d=1280) to 24 kHz waveform using the BigVGAN
// neural vocoder architecture with SnakeBeta activations.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct indextts_voc_context;

// Load BigVGAN vocoder from a GGUF file.
// Returns nullptr on failure.
struct indextts_voc_context* indextts_voc_init(const char* path, int n_threads, bool use_gpu);

// Generate waveform from GPT hidden states.
// latent: [T, 1280] float array (T time steps, 1280 hidden dim) in row-major order.
// spk_emb: [512] float speaker embedding (nullptr for zero/default).
// out_n: receives the number of output audio samples.
// Returns malloc'd float array, caller frees with free().
float* indextts_voc_generate(struct indextts_voc_context* ctx, const float* latent, int T, const float* spk_emb,
                             int* out_n);

void indextts_voc_free(struct indextts_voc_context* ctx);

#ifdef __cplusplus
}
#endif
