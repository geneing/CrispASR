#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moonshine_streaming_context;

struct moonshine_streaming_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
};

struct moonshine_streaming_context_params moonshine_streaming_context_default_params(void);

// Initialize from a GGUF file. Returns nullptr on failure.
struct moonshine_streaming_context* moonshine_streaming_init_from_file(
    const char* path_model, struct moonshine_streaming_context_params params);

// Transcribe PCM audio (16kHz mono float32). Returns malloc'd UTF-8 string (caller frees).
char* moonshine_streaming_transcribe(struct moonshine_streaming_context* ctx, const float* pcm, int n_samples);

// Variant that additionally returns per-emitted-token ids + softmax probs.
// Free with moonshine_streaming_result_free.
struct moonshine_streaming_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct moonshine_streaming_result* moonshine_streaming_transcribe_with_probs(
    struct moonshine_streaming_context* ctx, const float* pcm, int n_samples);
void moonshine_streaming_result_free(struct moonshine_streaming_result* r);

// Single-id detokenize. Returns a thread-local buffer valid until the next
// call. Empty string for special / out-of-range ids.
const char* moonshine_streaming_token_text(struct moonshine_streaming_context* ctx, int id);

// Free context and all associated memory.
void moonshine_streaming_free(struct moonshine_streaming_context* ctx);

// Set thread count after init.
void moonshine_streaming_set_n_threads(struct moonshine_streaming_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
