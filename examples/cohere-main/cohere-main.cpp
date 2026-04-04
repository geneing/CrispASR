// cohere-main.cpp — CLI for Cohere Transcribe via ggml
//
// Usage:
//   cohere-main -m cohere-transcribe.gguf -f audio.wav [-l en] [-t 4]

#include "cohere.h"
#include "common.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Minimal WAV reader (16-bit PCM, mono or stereo → mono downmix)
static bool read_wav(const char * path, std::vector<float> & out, int target_sr = 16000) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cohere-main: cannot open '%s'\n", path); return false; }

    // RIFF header
    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); fprintf(stderr, "cohere-main: not a RIFF file\n"); return false; }
    fseek(f, 4, SEEK_CUR); // file size
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); fprintf(stderr, "cohere-main: not a WAVE file\n"); return false; }

    // Chunk loop
    int channels = 1, sample_rate = 16000, bits = 16;
    bool data_found = false;
    while (!data_found) {
        char chunk_id[4]; if (fread(chunk_id, 1, 4, f) != 4) break;
        int chunk_size; fread(&chunk_size, 4, 1, f);
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            short audio_fmt; fread(&audio_fmt, 2, 1, f);
            short nch;       fread(&nch, 2, 1, f);      channels    = nch;
            int   sr;        fread(&sr,  4, 1, f);      sample_rate = sr;
            fseek(f, 4, SEEK_CUR);  // byte rate
            fseek(f, 2, SEEK_CUR);  // block align
            short bps;       fread(&bps, 2, 1, f);      bits = bps;
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            int n_samples = chunk_size / (bits / 8) / channels;
            out.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                float v = 0.0f;
                for (int c = 0; c < channels; c++) {
                    if (bits == 16) {
                        short s; fread(&s, 2, 1, f); v += s / 32768.0f;
                    } else if (bits == 32) {
                        float s; fread(&s, 4, 1, f); v += s;
                    }
                }
                out[i] = v / channels; // downmix to mono
            }
            if (sample_rate != target_sr)
                fprintf(stderr, "cohere-main: warning: audio is %d Hz, model expects %d Hz\n", sample_rate, target_sr);
            data_found = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);
    return data_found;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m MODEL.gguf -f AUDIO.wav [-l LANG] [-t THREADS]\n", prog);
    fprintf(stderr, "  -m   path to cohere-transcribe.gguf\n");
    fprintf(stderr, "  -f   input WAV file (16 kHz mono PCM recommended)\n");
    fprintf(stderr, "  -l   language code (default: en)\n");
    fprintf(stderr, "  -t   number of threads (default: 4)\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * audio_path = nullptr;
    const char * lang       = "en";
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) audio_path = argv[++i];
        else if (strcmp(argv[i], "-l") == 0 && i+1 < argc) lang       = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads  = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (!model_path || !audio_path) {
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    struct cohere_context_params params = cohere_context_default_params();
    params.n_threads = n_threads;

    fprintf(stderr, "cohere-main: loading model from '%s'...\n", model_path);
    struct cohere_context * ctx = cohere_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "cohere-main: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "cohere-main: model loaded, vocab size = %d\n", cohere_n_vocab(ctx));

    // Load audio
    std::vector<float> samples;
    fprintf(stderr, "cohere-main: loading audio from '%s'...\n", audio_path);
    if (!read_wav(audio_path, samples)) {
        cohere_free(ctx);
        return 1;
    }
    fprintf(stderr, "cohere-main: audio: %d samples (%.1fs)\n",
            (int)samples.size(), samples.size() / 16000.0f);

    // Transcribe
    const int64_t t_start_ms = ggml_time_ms();
    char * text = cohere_transcribe(ctx, samples.data(), (int)samples.size(), lang);
    const int64_t t_end_ms = ggml_time_ms();

    if (text) {
        printf("%s\n", text);
        free(text);
    } else {
        fprintf(stderr, "cohere-main: transcription failed\n");
    }

    const double t_inference_s = (t_end_ms - t_start_ms) / 1000.0;
    const double audio_duration_s = samples.size() / 16000.0f;
    fprintf(stderr, "cohere-main: inference took %.2fs (%.2fx realtime)\n",
            t_inference_s, audio_duration_s / t_inference_s);

    cohere_free(ctx);
    return 0;
}
