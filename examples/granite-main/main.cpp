// granite-main — CLI for ibm-granite/granite-4.0-1b-speech.
//
// Usage:
//   granite-main -m granite-speech-1b.gguf -f audio.wav [-t 4]
//   granite-main -m auto -f audio.wav

#include "granite_speech.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };
    char riff[4]; f.read(riff, 4); uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff,"RIFF",4)!=0||memcmp(wave,"WAVE",4)!=0) return false;
    uint16_t afmt=0,nchan=0,bps=0; uint32_t sr=0; std::vector<uint8_t> data;
    while (f) {
        char cid[4]; f.read(cid,4); uint32_t csz; read32(csz); if(!f) break;
        if (!memcmp(cid,"fmt ",4)) {
            read16(afmt); read16(nchan); read32(sr);
            uint32_t br; read32(br); uint16_t ba; read16(ba); read16(bps);
            if (csz > 16) f.seekg(csz-16, std::ios::cur);
        } else if (!memcmp(cid,"data",4)) {
            data.resize(csz); f.read((char*)data.data(), csz); break;
        } else f.seekg(csz, std::ios::cur);
    }
    if (afmt!=1||bps!=16) return false;
    if (sr!=16000) { fprintf(stderr, "%s: need 16kHz (got %u)\n", path.c_str(), sr); return false; }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t ns = data.size()/2;
    if (nchan == 1) {
        out.resize(ns); for (size_t i=0;i<ns;i++) out[i]=pcm[i]/32768.0f;
    } else {
        size_t nf = ns/nchan; out.resize(nf);
        for (size_t i=0;i<nf;i++) { float s=0; for(int c=0;c<nchan;c++) s+=pcm[i*nchan+c]/32768.0f; out[i]=s/nchan; }
    }
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf -f AUDIO.wav [options]\n\n"
        "options:\n"
        "  -m  FNAME        granite speech GGUF model (required, or 'auto')\n"
        "  -f  FNAME        input audio, 16 kHz mono WAV (required)\n"
        "  -t  N            threads (default: 4)\n"
        "  -n  N            max new tokens (default: 200)\n"
        "  -np              suppress stderr info\n"
        "  -v               verbose debug output\n\n", prog);
}

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path;
    int n_threads = 4, max_new = 200, verbosity = 1;
    bool no_prints = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="-m"  && i+1<argc) model_path = argv[++i];
        else if (a=="-f"  && i+1<argc) audio_path = argv[++i];
        else if (a=="-t"  && i+1<argc) n_threads  = std::atoi(argv[++i]);
        else if (a=="-n"  && i+1<argc) max_new    = std::atoi(argv[++i]);
        else if (a=="-np")             { no_prints = true; verbosity = 0; }
        else if (a=="-v")              verbosity = 2;
        else if (a=="-h"||a=="--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown: %s\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()||audio_path.empty()) { print_usage(argv[0]); return 1; }

    // Auto-download
    if (model_path == "auto" || model_path == "default") {
        const char * home = getenv("HOME");
        std::string cache_dir = std::string(home ? home : "/tmp") + "/.cache/crispasr";
        std::string cached = cache_dir + "/granite-speech-1b-q4_k.gguf";
        if (access(cached.c_str(), F_OK) == 0) {
            model_path = cached;
            if (!no_prints) fprintf(stderr, "using cached model: %s\n", cached.c_str());
        } else {
            fprintf(stderr, "downloading granite-speech-1b-q4_k.gguf...\n");
            std::string cmd = "mkdir -p '" + cache_dir + "' && "
                "hf download cstr/granite-speech-4.0-1b-GGUF "
                "granite-speech-1b-q4_k.gguf "
                "--local-dir '" + cache_dir + "' 2>&1";
            int rc = system(cmd.c_str());
            if (rc != 0) { fprintf(stderr, "download failed (rc=%d)\n", rc); return 1; }
            model_path = cached;
        }
    }

    // Load audio
    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    if (!no_prints) fprintf(stderr, "audio: %.2f s (%zu samples)\n",
                            samples.size()/16000.0, samples.size());

    // Init model
    auto cp = granite_speech_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = verbosity;
    auto * ctx = granite_speech_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    auto t0 = clk::now();

    // Mel spectrogram
    int n_mels = 0, T_mel = 0;
    float * mel = granite_speech_compute_mel(ctx, samples.data(), (int)samples.size(),
                                              &n_mels, &T_mel);
    if (!mel) { fprintf(stderr, "mel failed\n"); granite_speech_free(ctx); return 4; }
    auto t1 = clk::now();
    if (!no_prints) fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", T_mel, n_mels, ms(t0,t1));

    // Encoder
    int N_enc = 0, enc_dim = 0;
    float * enc = granite_speech_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &enc_dim);
    free(mel);
    if (!enc) { fprintf(stderr, "encoder failed\n"); granite_speech_free(ctx); return 5; }
    auto t2 = clk::now();
    if (!no_prints) fprintf(stderr, "encoder: %d × %d  (%.0f ms)\n", N_enc, enc_dim, ms(t1,t2));

    // Projector
    int N_proj = 0, proj_dim = 0;
    float * proj = granite_speech_run_projector(ctx, enc, N_enc, enc_dim, &N_proj, &proj_dim);
    free(enc);
    if (!proj) { fprintf(stderr, "projector failed\n"); granite_speech_free(ctx); return 6; }
    auto t3 = clk::now();
    if (!no_prints) fprintf(stderr, "projector: %d × %d  (%.0f ms)\n", N_proj, proj_dim, ms(t2,t3));

    // Build prompt: "USER: <audio>*N question\n ASSISTANT:"
    // Correct prompt from HF chat template (no BOS):
    // prefix: "USER: " = [6584, 25, 220]
    // suffix: "can you transcribe the speech into a written format?\n ASSISTANT:" = 16 tokens
    const int AUDIO_TOK = 100352;
    const int EOS       = 100257;
    int32_t prefix[] = {6584, 25, 220};
    int32_t suffix[] = {4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198, 36660, 3931, 2891, 25};
    const int n_prefix = 3, n_suffix = 16;
    const int total_prompt = n_prefix + N_proj + n_suffix;

    std::vector<int32_t> prompt_ids;
    for (int i = 0; i < n_prefix; i++) prompt_ids.push_back(prefix[i]);
    for (int i = 0; i < N_proj; i++) prompt_ids.push_back(AUDIO_TOK);
    for (int i = 0; i < n_suffix; i++) prompt_ids.push_back(suffix[i]);

    // Embed all tokens (raw, no multiplier — LLM forward applies it)
    float * all_embeds = granite_speech_embed_tokens(ctx, prompt_ids.data(), total_prompt);
    if (!all_embeds) { free(proj); granite_speech_free(ctx); return 7; }

    // Replace audio positions with projector output
    for (int i = 0; i < N_proj; i++)
        std::memcpy(all_embeds + (size_t)(n_prefix + i) * proj_dim,
                     proj + (size_t)i * proj_dim, proj_dim * sizeof(float));
    free(proj);

    // KV cache + prefill
    if (!granite_speech_kv_init(ctx, 4096)) {
        free(all_embeds); granite_speech_free(ctx); return 8;
    }
    granite_speech_kv_reset(ctx);

    auto t4 = clk::now();
    int vocab = 0;
    float * logits = granite_speech_run_llm_kv(ctx, all_embeds, total_prompt, 0, nullptr, &vocab);
    free(all_embeds);
    if (!logits) { fprintf(stderr, "prefill failed\n"); granite_speech_free(ctx); return 9; }
    auto t5 = clk::now();
    if (!no_prints) fprintf(stderr, "prefill: %d tokens  (%.0f ms)\n", total_prompt, ms(t4,t5));

    // Greedy decode
    int next = 0;
    { float mx = -1e30f; for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; next = k; } }
    free(logits);

    int n_past = total_prompt;
    std::vector<int32_t> gen_ids;
    auto t6 = clk::now();

    for (int step = 0; step < max_new; step++) {
        if (next == EOS) break;
        gen_ids.push_back(next);

        int32_t tok = next;
        float * emb = granite_speech_embed_tokens(ctx, &tok, 1);
        float * lg = granite_speech_run_llm_kv(ctx, emb, 1, n_past, nullptr, nullptr);
        free(emb);
        if (!lg) break;
        n_past++;

        next = 0; float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > mx) { mx = lg[k]; next = k; }
        free(lg);
    }
    if (next == EOS) gen_ids.push_back(next);

    auto t7 = clk::now();

    // Decode token IDs to text (simple: look up each token via the model's tokenizer)
    // For now, just use granite_speech_embed_tokens to verify token validity,
    // and print token IDs that can be decoded externally.
    // TODO: integrate tokenizer for direct text output.

    if (!no_prints) {
        fprintf(stderr, "decode: %zu tokens  (%.0f ms, %.0f ms/tok)\n",
                gen_ids.size(), ms(t6,t7),
                gen_ids.empty() ? 0 : ms(t6,t7)/gen_ids.size());
        fprintf(stderr, "total: %.0f ms (mel %.0f + enc %.0f + proj %.0f + prefill %.0f + decode %.0f)\n",
                ms(t0,t7), ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t4,t5), ms(t6,t7));
    }

    // Filter out EOS from gen_ids
    std::vector<int32_t> text_ids;
    for (int32_t id : gen_ids) if (id != EOS) text_ids.push_back(id);

    // Decode tokens to text
    char * text = granite_speech_decode_tokens(ctx, text_ids.data(), (int)text_ids.size());
    if (text) {
        printf("%s\n", text);
        free(text);
    } else {
        // Fallback: print token IDs
        for (int32_t id : text_ids) printf("%d ", id);
        printf("\n");
    }

    granite_speech_free(ctx);
    return 0;
}
