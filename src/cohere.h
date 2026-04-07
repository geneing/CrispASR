#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cohere_context;

struct cohere_context_params {
    int   n_threads;   // default: number of physical cores
    bool  use_flash;   // flash attention in decoder (default: false for now)
    // Output verbosity:
    //   0 = silent  — only hard errors (failed/cannot) go to stderr
    //   1 = normal  — model loading info printed (default)
    //   2 = verbose — per-inference timing, per-step tokens, performance report
    int   verbosity;
};

struct cohere_context_params cohere_context_default_params(void);

// Load model from GGUF file produced by export_gguf.py
struct cohere_context * cohere_init_from_file(const char * path_model,
                                              struct cohere_context_params params);

void cohere_free(struct cohere_context * ctx);

// Transcribe raw 16 kHz mono PCM.
// Returns a newly allocated UTF-8 string (caller must free()).
// lang: ISO-639-1 code e.g. "en", "fr", "de" (NULL → autodetect, not implemented yet)
char * cohere_transcribe(struct cohere_context * ctx,
                         const float * samples, int n_samples,
                         const char * lang);

// Vocabulary helpers
int         cohere_n_vocab(struct cohere_context * ctx);
const char* cohere_token_to_str(struct cohere_context * ctx, int token_id);
int         cohere_str_to_token(struct cohere_context * ctx, const char * str);

#ifdef __cplusplus
}
#endif
