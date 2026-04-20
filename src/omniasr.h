// omniasr.h — Facebook OmniASR-CTC (wav2vec2-style encoder + CTC head).
//
// Architecture: 7-layer CNN frontend + 24L Transformer encoder + CTC projection.
// Input: raw 16kHz mono PCM. Output: text via SentencePiece tokenizer.
// 325M params (300M variant), 1600+ languages, Apache-2.0.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct omniasr_context;

struct omniasr_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
};

struct omniasr_context_params omniasr_context_default_params(void);

struct omniasr_context* omniasr_init_from_file(const char* path_model, struct omniasr_context_params params);

void omniasr_free(struct omniasr_context* ctx);

// Transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* omniasr_transcribe(struct omniasr_context* ctx, const float* samples, int n_samples);

#ifdef __cplusplus
}
#endif
