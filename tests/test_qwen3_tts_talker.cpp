// test_qwen3_tts_talker.cpp — sanity check the Qwen3-TTS talker forward.
//
// Usage:
//   ./build/tests/test_qwen3_tts_talker <model.gguf> "<text>"
//
// Loads the talker, runs greedy AR decode of codebook-0, prints the
// resulting code stream. Not auto-registered with ctest — needs a
// real GGUF + the converter on disk.

#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <text>\n", argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* text = argv[2];

    qwen3_tts_context_params p = qwen3_tts_context_default_params();
    p.verbosity = 1;
    p.use_gpu = true;

    qwen3_tts_context* ctx = qwen3_tts_init_from_file(model, p);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 2;
    }

    int n = 0;
    int32_t* codes = qwen3_tts_synthesize_codes(ctx, text, &n);
    if (!codes) {
        fprintf(stderr, "synth failed\n");
        qwen3_tts_free(ctx);
        return 3;
    }

    fprintf(stderr, "got %d codes\n", n);
    fprintf(stdout, "[");
    for (int i = 0; i < n; i++)
        fprintf(stdout, i + 1 < n ? "%d, " : "%d", codes[i]);
    fprintf(stdout, "]\n");

    qwen3_tts_codes_free(codes);
    qwen3_tts_free(ctx);
    return 0;
}
