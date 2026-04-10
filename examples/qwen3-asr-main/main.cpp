// qwen3-asr-main — CLI for Qwen/Qwen3-ASR-0.6B ggml runtime.
//
// Reads a 16 kHz mono WAV, runs the full audio→text pipeline (mel → encoder
// → splice into ChatML prompt → Qwen3 LLM forward with KV cache → greedy
// decode), and prints the transcript.
//
// Optional word-level timestamps via a CTC aligner second pass (-am flag).
//
// Usage:
//   qwen3-asr-main -m qwen3-asr-0.6b.gguf -f audio.wav [-t 4]
//   qwen3-asr-main -m model.gguf -f audio.wav -am aligner.gguf -timestamps

#include "qwen3_asr.h"
#include "canary_ctc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Tiny WAV reader: 16-bit PCM, mono, 16 kHz only.
// ---------------------------------------------------------------------------
static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };

    char riff[4]; f.read(riff, 4);
    uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a RIFF/WAVE file\n", path.c_str()); return false;
    }

    uint16_t audio_format = 0, n_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data;

    while (f) {
        char chunk_id[4]; f.read(chunk_id, 4);
        uint32_t chunk_sz; read32(chunk_sz);
        if (!f) break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            read16(audio_format); read16(n_channels);
            read32(sample_rate);
            uint32_t byte_rate; read32(byte_rate);
            uint16_t block_align; read16(block_align);
            read16(bits_per_sample);
            if (chunk_sz > 16) f.seekg(chunk_sz - 16, std::ios::cur);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data.resize(chunk_sz);
            f.read((char*)data.data(), chunk_sz);
            break;
        } else {
            f.seekg(chunk_sz, std::ios::cur);
        }
    }
    if (audio_format != 1 || bits_per_sample != 16) {
        fprintf(stderr, "%s: only 16-bit PCM supported (got format=%d bits=%d)\n",
                path.c_str(), audio_format, bits_per_sample); return false;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "%s: only 16 kHz supported (got %u). Pre-convert with:\n"
                        "  ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav\n",
                path.c_str(), sample_rate); return false;
    }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t n_samples_total = data.size() / 2;
    if (n_channels == 1) {
        out.resize(n_samples_total);
        for (size_t i = 0; i < n_samples_total; i++) out[i] = pcm[i] / 32768.0f;
    } else {
        size_t n_frames = n_samples_total / n_channels;
        out.resize(n_frames);
        for (size_t i = 0; i < n_frames; i++) {
            float s = 0;
            for (int c = 0; c < n_channels; c++) s += pcm[i * n_channels + c] / 32768.0f;
            out[i] = s / n_channels;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build the ChatML prompt token IDs by tokenising the template via the
// model's BPE encoder. This replaces the previous hardcoded ID list and
// makes it trivial to add an explicit language hint or system prompt.
// ---------------------------------------------------------------------------
static std::vector<int32_t> build_prompt_ids(qwen3_asr_context * ctx, int n_audio_pad) {
    std::string text =
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|>";
    text.reserve(text.size() + 13 * n_audio_pad + 64);
    for (int i = 0; i < n_audio_pad; i++) text += "<|audio_pad|>";
    text +=
        "<|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n";

    int n = 0;
    int32_t * raw = qwen3_asr_tokenize(ctx, text.c_str(), &n);
    std::vector<int32_t> ids;
    if (raw) {
        ids.assign(raw, raw + n);
        free(raw);
    }
    return ids;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level decoder
// ---------------------------------------------------------------------------
static std::vector<int> & byte_decoder() {
    static std::vector<int> dec(0x200, -1);
    static bool initialized = false;
    if (initialized) return dec;
    std::vector<int> bs, cs;
    for (int b = 0x21; b <= 0x7e; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xa1; b <= 0xac; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xae; b <= 0xff; b++) { bs.push_back(b); cs.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool present = false;
        for (int x : bs) if (x == b) { present = true; break; }
        if (!present) { bs.push_back(b); cs.push_back(256 + n); n++; }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        if ((size_t)cs[i] < dec.size()) dec[cs[i]] = bs[i];
    }
    initialized = true;
    return dec;
}

static std::string decode_token(const std::string & s) {
    auto & dec = byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int cp = 0, len = 1;
        if      (c < 0x80)            { cp = c;          len = 1; }
        else if ((c & 0xE0) == 0xC0)  { cp = c & 0x1F;   len = 2; }
        else if ((c & 0xF0) == 0xE0)  { cp = c & 0x0F;   len = 3; }
        else if ((c & 0xF8) == 0xF0)  { cp = c & 0x07;   len = 4; }
        else { i++; continue; }
        if (i + len > s.size()) break;
        for (int k = 1; k < len; k++) cp = (cp << 6) | (s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < (int)dec.size() && dec[cp] >= 0) {
            out.push_back((char)dec[cp]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Word tokenizer for CTC alignment
// ---------------------------------------------------------------------------
static std::vector<std::string> tokenise_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ---------------------------------------------------------------------------
// SRT / VTT output helpers
// ---------------------------------------------------------------------------
static std::string format_time_srt(int64_t cs) {
    int h = (int)(cs / 360000);
    int m = (int)((cs % 360000) / 6000);
    int s = (int)((cs % 6000) / 100);
    int ms = (int)(cs % 100) * 10;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, s, ms);
    return buf;
}

static std::string format_time_vtt(int64_t cs) {
    int h = (int)(cs / 360000);
    int m = (int)((cs % 360000) / 6000);
    int s = (int)((cs % 6000) / 100);
    int ms = (int)(cs % 100) * 10;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, s, ms);
    return buf;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf -f AUDIO.wav [options]\n\n"
        "options:\n"
        "  -h, --help       show this help\n"
        "  -m  FNAME        qwen3-asr GGUF model (required)\n"
        "  -f  FNAME        input audio, 16 kHz mono WAV (required)\n"
        "  -t  N            threads (default: 4)\n"
        "  -n  N            max new tokens (default: 256)\n"
        "  -am FNAME        CTC aligner GGUF (canary-ctc-aligner) for timestamps\n"
        "  -timestamps      enable word-level timestamps (requires -am)\n"
        "  -osrt            output SRT subtitle file (to stdout)\n"
        "  -ovtt            output VTT subtitle file (to stdout)\n"
        "  -np              no prints (suppress stderr info)\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path, aligner_path;
    int n_threads = 4;
    int max_new = 256;
    bool timestamps = false;
    bool out_srt = false;
    bool out_vtt = false;
    bool no_prints = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m"  && i+1 < argc) model_path   = argv[++i];
        else if (a == "-f"  && i+1 < argc) audio_path   = argv[++i];
        else if (a == "-t"  && i+1 < argc) n_threads    = std::atoi(argv[++i]);
        else if (a == "-n"  && i+1 < argc) max_new      = std::atoi(argv[++i]);
        else if (a == "-am" && i+1 < argc) aligner_path = argv[++i];
        else if (a == "-timestamps")       timestamps    = true;
        else if (a == "-osrt")           { timestamps = true; out_srt = true; }
        else if (a == "-ovtt")           { timestamps = true; out_vtt = true; }
        else if (a == "-np")               no_prints    = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || audio_path.empty()) {
        fprintf(stderr, "missing -m or -f. -h for help.\n"); return 1;
    }

    // Auto-download: "-m auto" downloads the default Q4_K from HuggingFace
    if (model_path == "auto" || model_path == "default") {
        const char * home = getenv("HOME");
        std::string cache_dir = std::string(home ? home : "/tmp") + "/.cache/crispasr";
        std::string cached = cache_dir + "/qwen3-asr-0.6b-q4_k.gguf";
        if (access(cached.c_str(), F_OK) == 0) {
            model_path = cached;
            if (!no_prints) fprintf(stderr, "using cached model: %s\n", cached.c_str());
        } else {
            fprintf(stderr, "downloading qwen3-asr-0.6b-q4_k.gguf (~516 MB)...\n");
            std::string cmd = "mkdir -p '" + cache_dir + "' && "
                "hf download cstr/qwen3-asr-0.6b-GGUF "
                "qwen3-asr-0.6b-q4_k.gguf "
                "--local-dir '" + cache_dir + "' 2>&1";
            int rc = system(cmd.c_str());
            if (rc != 0) {
                fprintf(stderr, "download failed (rc=%d). Install hf CLI: pip install huggingface_hub\n", rc);
                return 1;
            }
            model_path = cached;
        }
    }

    if (timestamps && aligner_path.empty()) {
        fprintf(stderr, "error: -timestamps / -osrt / -ovtt require -am ALIGNER.gguf\n");
        return 1;
    }

    // ----- Load WAV -----
    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    if (!no_prints)
        fprintf(stderr, "audio: %.2f s (%zu samples)\n", samples.size() / 16000.0, samples.size());

    // ----- Init context -----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = no_prints ? 0 : 1;
    auto * ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    auto t0 = std::chrono::steady_clock::now();

    // ----- Mel -----
    int n_mels = 0, T_mel = 0;
    float * mel = qwen3_asr_compute_mel(ctx, samples.data(), (int)samples.size(),
                                        &n_mels, &T_mel);
    if (!mel) { fprintf(stderr, "mel failed\n"); qwen3_asr_free(ctx); return 4; }
    auto t_mel = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", n_mels, T_mel,
                std::chrono::duration<double, std::milli>(t_mel - t0).count());

    // ----- Encoder -----
    int N_enc = 0, pdim = 0;
    float * audio_embeds = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) { fprintf(stderr, "encoder failed\n"); qwen3_asr_free(ctx); return 5; }
    auto t_enc = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "encoder: %d frames × %d dim  (%.0f ms)\n", N_enc, pdim,
                std::chrono::duration<double, std::milli>(t_enc - t_mel).count());

    // ----- Build prompt + embed text + splice audio -----
    auto ids = build_prompt_ids(ctx, N_enc);
    int T_prompt = (int)ids.size();
    if (!no_prints)
        fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    float * text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { fprintf(stderr, "embed failed\n"); free(audio_embeds); qwen3_asr_free(ctx); return 6; }

    const int AUDIO_PAD = 151676;
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == AUDIO_PAD) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);
    if (!no_prints)
        fprintf(stderr, "spliced %d audio frames\n", spliced);

    // ----- KV cache + prefill -----
    if (!qwen3_asr_kv_init(ctx, 4096)) {
        fprintf(stderr, "kv init failed\n"); free(text_embeds); qwen3_asr_free(ctx); return 7;
    }
    qwen3_asr_kv_reset(ctx);

    auto t_pf0 = std::chrono::steady_clock::now();
    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    auto t_pf1 = std::chrono::steady_clock::now();
    if (!logits) { fprintf(stderr, "prefill failed\n"); free(text_embeds); qwen3_asr_free(ctx); return 8; }
    if (!no_prints)
        fprintf(stderr, "prefill: %.0f ms\n",
                std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count());
    free(text_embeds);

    int next = 0; float mx = -1e30f;
    for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; next = k; }
    free(logits);

    // ----- Greedy decode loop -----
    const int EOS = 151645;
    std::vector<int32_t> gen;
    gen.push_back(next);

    auto t_dec0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    while ((int)gen.size() < max_new && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = qwen3_asr_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        int n_t2 = 0, v2 = 0;
        float * lg = qwen3_asr_run_llm_kv(ctx, tail, 1, n_past, &n_t2, &v2);
        free(tail);
        if (!lg) break;
        n_past += 1;
        int nx = 0; float lmx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > lmx) { lmx = lg[k]; nx = k; }
        free(lg);
        gen.push_back(nx);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    if (!no_prints)
        fprintf(stderr, "decode: %zu tokens in %.0f ms (%.0f ms/token)\n",
                gen.size() - 1, dec_ms, dec_ms / std::max((size_t)1, gen.size() - 1));

    // ----- Decode token IDs to text -----
    std::string transcript;
    std::string detected_language;
    bool capture_language = false;
    for (auto id : gen) {
        if (id == EOS) break;
        std::string raw = qwen3_asr_token_text(ctx, id);
        if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') continue;
        if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>') continue;  // <asr_text> etc.
        if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D') continue;
        std::string txt = decode_token(raw);
        if (txt == "language") { capture_language = true; continue; }
        if (capture_language) {
            size_t s = 0;
            while (s < txt.size() && (txt[s] == ' ' || txt[s] == '\t')) s++;
            detected_language = txt.substr(s);
            capture_language = false;
            continue;
        }
        transcript += txt;
    }
    // Trim leading whitespace
    size_t start = 0;
    while (start < transcript.size() && (transcript[start] == ' ' || transcript[start] == '\t')) start++;
    transcript = transcript.substr(start);

    if (!no_prints && !detected_language.empty()) {
        fprintf(stderr, "language: %s\n", detected_language.c_str());
    }

    auto t_total = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "total: %.0f ms\n",
                std::chrono::duration<double, std::milli>(t_total - t0).count());

    // ----- Timestamps via CTC aligner second pass -----
    if (timestamps) {
        auto t_align0 = std::chrono::steady_clock::now();

        canary_ctc_context_params acp = canary_ctc_context_default_params();
        acp.n_threads = n_threads;
        canary_ctc_context * actx = canary_ctc_init_from_file(aligner_path.c_str(), acp);
        if (!actx) { fprintf(stderr, "failed to load aligner model\n"); qwen3_asr_free(ctx); return 9; }

        float * ctc_logits = nullptr;
        int T_ctc = 0, V_ctc = 0;
        int rc = canary_ctc_compute_logits(actx, samples.data(), (int)samples.size(),
                                           &ctc_logits, &T_ctc, &V_ctc);
        if (rc != 0) {
            fprintf(stderr, "CTC logits failed (rc=%d)\n", rc);
            canary_ctc_free(actx);
            qwen3_asr_free(ctx);
            return 10;
        }

        auto words = tokenise_words(transcript);
        if (words.empty()) {
            if (!no_prints) fprintf(stderr, "no words to align\n");
            printf("%s\n", transcript.c_str());
        } else {
            std::vector<canary_ctc_word> out_words(words.size());
            std::vector<const char *> word_ptrs(words.size());
            for (size_t i = 0; i < words.size(); i++) word_ptrs[i] = words[i].c_str();

            rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc,
                                        word_ptrs.data(), (int)words.size(),
                                        out_words.data());
            auto t_align1 = std::chrono::steady_clock::now();
            if (!no_prints)
                fprintf(stderr, "align: %zu words in %.0f ms\n", words.size(),
                        std::chrono::duration<double, std::milli>(t_align1 - t_align0).count());

            if (rc != 0) {
                fprintf(stderr, "alignment failed (rc=%d), printing plain transcript\n", rc);
                printf("%s\n", transcript.c_str());
            } else if (out_srt) {
                for (size_t i = 0; i < out_words.size(); i++) {
                    printf("%zu\n%s --> %s\n%s\n\n",
                           i + 1,
                           format_time_srt(out_words[i].t0).c_str(),
                           format_time_srt(out_words[i].t1).c_str(),
                           out_words[i].text);
                }
            } else if (out_vtt) {
                printf("WEBVTT\n\n");
                for (size_t i = 0; i < out_words.size(); i++) {
                    printf("%s --> %s\n%s\n\n",
                           format_time_vtt(out_words[i].t0).c_str(),
                           format_time_vtt(out_words[i].t1).c_str(),
                           out_words[i].text);
                }
            } else {
                for (const auto & w : out_words) {
                    printf("[%8.2fs → %8.2fs]  %s\n",
                           w.t0 / 100.0, w.t1 / 100.0, w.text);
                }
            }
        }
        free(ctc_logits);
        canary_ctc_free(actx);
    } else {
        printf("%s\n", transcript.c_str());
    }

    qwen3_asr_free(ctx);
    return 0;
}
