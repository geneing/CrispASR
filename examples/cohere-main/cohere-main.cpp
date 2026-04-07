// cohere-main.cpp — CLI for Cohere Transcribe
//
// Usage matches whisper-cli conventions:
//   cohere-main -m MODEL.gguf -f audio.wav [-l en] [-t 4] [--verbose]
//   cohere-main -m MODEL.gguf -f audio.wav -ts -vad-model ggml-silero-vad.bin
//   cohere-main -m MODEL.gguf -f audio.wav -ts -vad-model ggml-silero-vad.bin -pc -osrt
//
// By default only the transcript is written to stdout; all progress info
// goes to stderr and is suppressed unless --verbose is passed.

#include "cohere.h"
#include "common.h"
#include "common-whisper.h"
#include "ggml.h"
#include "whisper.h"   // for Silero VAD API

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct cohere_params {
    std::string model;
    std::string fname_inp;
    std::string language       = "en";
    int         n_threads      = std::min(4, (int)std::thread::hardware_concurrency());
    int         verbosity      = 1;    // 0=silent 1=normal(loading only) 2=verbose
    bool        use_flash      = false;
    bool        no_prints      = false;
    bool        debug          = false;
    bool        no_punctuation = false;
    bool        diarize        = false; // -tdrz: enable speaker diarization (experimental)
    bool        output_txt     = false; // -ot  : write transcript to <audio>.txt
    bool        timestamps     = false; // -ts  : enable timestamp output
    bool        print_colors   = false; // -pc  : ANSI confidence color-coding
    int         max_len        = 0;    // -ml N: max chars per output segment (0=unlimited)
    bool        output_srt     = false; // -osrt: write .srt file
    bool        output_vtt     = false; // -ovtt: write .vtt file
    std::string vad_model;             // -vad-model: path to ggml-silero-vad.bin
    float       vad_thold      = 0.5f; // -vad-thold: speech probability threshold
    int         vad_min_speech_ms  = 250;
    int         vad_min_silence_ms = 100;
    float       vad_speech_pad_ms  = 30.0f;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] -m MODEL -f AUDIO\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,         --help              show this help message\n");
    fprintf(stderr, "  -m FNAME,   --model FNAME        path to cohere-transcribe.gguf\n");
    fprintf(stderr, "  -f FNAME,   --file FNAME         input audio file (WAV 16 kHz mono)\n");
    fprintf(stderr, "  -l LANG,    --language LANG      language code (default: en)\n");
    fprintf(stderr, "  -t N,       --threads N          number of threads (default: %d)\n",
            std::min(4, (int)std::thread::hardware_concurrency()));
    fprintf(stderr, "  -ot,        --output-txt         write plain transcript to <audio>.txt\n");
    fprintf(stderr, "  -ts,        --timestamps         output with timestamps (VAD recommended)\n");
    fprintf(stderr, "  -ml N,      --max-len N          max chars per output segment (0=unlimited)\n");
    fprintf(stderr, "  -pc,        --print-colors       color-code output by token confidence\n");
    fprintf(stderr, "  -osrt,      --output-srt         write SRT subtitle file\n");
    fprintf(stderr, "  -ovtt,      --output-vtt         write WebVTT subtitle file\n");
    fprintf(stderr, "  -vad-model FNAME                 path to ggml-silero-vad.bin for VAD\n");
    fprintf(stderr, "  -vad-thold F                     VAD speech threshold (default: 0.5)\n");
    fprintf(stderr, "  -tdrz,      --diarize            enable speaker diarization (experimental)\n");
    fprintf(stderr, "  -npnc,      --no-punctuation     disable punctuation in output\n");
    fprintf(stderr, "  -v,         --verbose            show timing info and per-step tokens\n");
    fprintf(stderr, "  -np,        --no-prints          suppress all informational output\n");
    fprintf(stderr, "  -d,         --debug              enable COHERE_DEBUG and COHERE_PROF\n");
    fprintf(stderr, "  --flash                          enable flash attention in decoder\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "supported languages:\n");
    fprintf(stderr, "  ar de el en es fr it ja ko nl pl pt vi zh\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "timestamp modes:\n");
    fprintf(stderr, "  -ts                             one segment = full audio [00:00:00 --> dur]\n");
    fprintf(stderr, "  -ts -vad-model ggml-silero-vad.bin   VAD splits audio into speech segments\n");
    fprintf(stderr, "  -ts -ml 1                       approx word-level timestamps\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "environment:\n");
    fprintf(stderr, "  COHERE_DEVICE=metal|cuda|cpu    force backend selection\n");
    fprintf(stderr, "  COHERE_THREADS=N                override thread count\n");
    fprintf(stderr, "  COHERE_DEBUG=1                  verbose tensor/graph logging\n");
    fprintf(stderr, "  COHERE_PROF=1                   per-op profiling\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "input must be 16 kHz mono WAV; convert with:\n");
    fprintf(stderr, "  ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio.wav\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "VAD model download:\n");
    fprintf(stderr, "  ./models/download-ggml-model.sh silero-vad\n");
    fprintf(stderr, "  (or: cp /path/to/ggml-silero-vad.bin .)\n");
    fprintf(stderr, "\n");
}

static bool parse_args(int argc, char ** argv, cohere_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if ((arg == "-m" || arg == "--model")    && i+1 < argc) { p.model          = argv[++i];
        } else if ((arg == "-f" || arg == "--file")     && i+1 < argc) { p.fname_inp      = argv[++i];
        } else if ((arg == "-l" || arg == "--language") && i+1 < argc) { p.language       = argv[++i];
        } else if ((arg == "-t" || arg == "--threads")  && i+1 < argc) { p.n_threads      = std::atoi(argv[++i]);
        } else if ((arg == "-ml" || arg == "--max-len") && i+1 < argc) { p.max_len        = std::atoi(argv[++i]);
        } else if (arg == "-vad-model"                 && i+1 < argc)  { p.vad_model      = argv[++i];
        } else if (arg == "-vad-thold"                 && i+1 < argc)  { p.vad_thold      = std::atof(argv[++i]);
        } else if (arg == "-ot"   || arg == "--output-txt")   { p.output_txt     = true;
        } else if (arg == "-ts"   || arg == "--timestamps")   { p.timestamps     = true;
        } else if (arg == "-pc"   || arg == "--print-colors") { p.print_colors   = true;
        } else if (arg == "-osrt" || arg == "--output-srt")   { p.output_srt     = true;
        } else if (arg == "-ovtt" || arg == "--output-vtt")   { p.output_vtt     = true;
        } else if (arg == "-tdrz" || arg == "--diarize")        { p.diarize        = true;
        } else if (arg == "-npnc" || arg == "--no-punctuation"){ p.no_punctuation = true;
        } else if (arg == "-v"    || arg == "--verbose")       { p.verbosity      = 2;
        } else if (arg == "-np"   || arg == "--no-prints")     { p.no_prints      = true;
        } else if (arg == "-d"    || arg == "--debug")         { p.debug          = true;
        } else if (arg == "--flash")                           { p.use_flash      = true;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (p.model.empty() || p.fname_inp.empty()) {
        fprintf(stderr, "error: -m MODEL and -f AUDIO are required\n\n");
        print_usage(argv[0]);
        return false;
    }

    if (p.no_prints) p.verbosity = 0;

    return true;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

// to_timestamp is provided by common-whisper.h (centiseconds → HH:MM:SS.mmm)

// Confidence color for a token probability p in [0,1].
// Uses the same 7-color gradient as whisper-cli (red=low → green=high).
// k_colors is defined in common.h as std::vector<std::string>.
static const char * ANSI_RESET = "\033[0m";

static std::string token_color(float p) {
    int n = (int)k_colors.size();
    int col = (int)(std::pow(p, 3.0f) * n);
    col = std::max(0, std::min(n - 1, col));
    return k_colors[col];
}

// Derive output path: strip known audio extensions, append suffix
static std::string make_output_path(const std::string & audio_path, const std::string & ext) {
    std::string base = audio_path;
    for (const char * e : {".wav", ".WAV", ".mp3", ".MP3", ".flac", ".FLAC",
                            ".ogg", ".OGG", ".m4a", ".M4A", ".opus", ".OPUS"}) {
        if (base.size() > strlen(e) &&
            base.compare(base.size() - strlen(e), strlen(e), e) == 0) {
            base = base.substr(0, base.size() - strlen(e));
            break;
        }
    }
    return base + ext;
}

// ---------------------------------------------------------------------------
// Segment rendering
//
// A "display segment" is one line of output: [t0 --> t1]  text
// With -ml N we split token arrays into sub-segments of max N chars.
// ---------------------------------------------------------------------------

struct disp_segment {
    int64_t t0, t1;   // centiseconds
    std::string text;
    // per-token probabilities for color output (parallel to words in text)
    std::vector<std::pair<std::string, float>> colored_tokens;
};

// Build display segments from a cohere_result.
// t0_seg / t1_seg: absolute segment bounds in centiseconds (from VAD or full audio).
// max_len: max chars per display segment (0 = unlimited).
static std::vector<disp_segment> make_disp_segments(
        const struct cohere_result * r,
        int64_t t0_seg, int64_t t1_seg,
        int max_len)
{
    std::vector<disp_segment> out;

    if (!r || r->n_tokens == 0) {
        if (r && r->text && r->text[0]) {
            disp_segment ds;
            ds.t0   = t0_seg;
            ds.t1   = t1_seg;
            ds.text = r->text;
            out.push_back(ds);
        }
        return out;
    }

    // Group tokens into display segments
    disp_segment cur;
    cur.t0 = r->tokens[0].t0;

    auto flush = [&]() {
        if (!cur.text.empty()) {
            // trim leading space
            if (cur.text[0] == ' ') cur.text = cur.text.substr(1);
            out.push_back(cur);
        }
        cur = disp_segment{};
    };

    for (int i = 0; i < r->n_tokens; i++) {
        const cohere_token_data & td = r->tokens[i];
        int tlen = (int)strlen(td.text);
        if (tlen == 0) continue;

        bool will_overflow = (max_len > 0) &&
                             (!cur.text.empty()) &&
                             ((int)cur.text.size() + tlen > max_len);

        if (will_overflow) {
            cur.t1 = td.t0;  // end at the start of this token
            flush();
        }

        if (cur.text.empty()) {
            cur.t0 = td.t0;
        }
        cur.t1 = td.t1;
        cur.text += td.text;
        cur.colored_tokens.push_back({td.text, td.p});
    }
    flush();

    return out;
}

// Print a single display segment to a FILE (stdout or file)
static void print_disp_segment(
        FILE * fp,
        const disp_segment & ds,
        bool show_timestamps,
        bool print_colors,
        bool to_file  // suppress ANSI codes when writing to file
) {
    if (show_timestamps) {
        fprintf(fp, "[%s --> %s]  ",
                to_timestamp(ds.t0).c_str(),
                to_timestamp(ds.t1).c_str());
    }

    if (print_colors && !to_file && !ds.colored_tokens.empty()) {
        for (const auto & [tok, p] : ds.colored_tokens) {
            fprintf(fp, "%s%s%s", token_color(p).c_str(), tok.c_str(), ANSI_RESET);
        }
    } else {
        fprintf(fp, "%s", ds.text.c_str());
    }
    fprintf(fp, "\n");
}

// Write SRT file from display segments
static void write_srt(const std::string & path,
                      const std::vector<disp_segment> & segs,
                      int verbosity, const char * prog)
{
    std::ofstream ofs(path);
    if (!ofs) {
        fprintf(stderr, "%s: warning: could not write SRT to '%s'\n", prog, path.c_str());
        return;
    }
    for (int i = 0; i < (int)segs.size(); i++) {
        ofs << (i + 1) << "\n";
        ofs << to_timestamp(segs[i].t0, /*comma=*/true)
            << " --> "
            << to_timestamp(segs[i].t1, /*comma=*/true) << "\n";
        ofs << segs[i].text << "\n\n";
    }
    if (verbosity >= 1)
        fprintf(stderr, "%s: SRT written to '%s'\n", prog, path.c_str());
}

// Write WebVTT file from display segments
static void write_vtt(const std::string & path,
                      const std::vector<disp_segment> & segs,
                      int verbosity, const char * prog)
{
    std::ofstream ofs(path);
    if (!ofs) {
        fprintf(stderr, "%s: warning: could not write VTT to '%s'\n", prog, path.c_str());
        return;
    }
    ofs << "WEBVTT\n\n";
    for (const auto & ds : segs) {
        ofs << to_timestamp(ds.t0) << " --> " << to_timestamp(ds.t1) << "\n";
        ofs << ds.text << "\n\n";
    }
    if (verbosity >= 1)
        fprintf(stderr, "%s: VTT written to '%s'\n", prog, path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    cohere_params p;
    if (!parse_args(argc, argv, p)) return 1;

    if (p.debug) {
#if defined(_WIN32)
        _putenv_s("COHERE_DEBUG", "1");
        _putenv_s("COHERE_PROF",  "1");
#else
        setenv("COHERE_DEBUG", "1", 1);
        setenv("COHERE_PROF",  "1", 1);
#endif
        p.verbosity = std::max(p.verbosity, 2);
    }

    // Build context params
    struct cohere_context_params params = cohere_context_default_params();
    params.n_threads      = p.n_threads;
    params.use_flash      = p.use_flash;
    params.no_punctuation = p.no_punctuation;
    params.diarize        = p.diarize;
    params.verbosity      = p.verbosity;

    if (p.verbosity >= 1)
        fprintf(stderr, "%s: loading model '%s'\n", argv[0], p.model.c_str());

    struct cohere_context * ctx = cohere_init_from_file(p.model.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load model '%s'\n", argv[0], p.model.c_str());
        return 1;
    }

    // Load audio
    std::vector<float> samples;
    std::vector<std::vector<float>> samples_stereo;
    if (!read_audio_data(p.fname_inp, samples, samples_stereo, /*stereo=*/false)) {
        fprintf(stderr, "%s: failed to read audio '%s'\n", argv[0], p.fname_inp.c_str());
        cohere_free(ctx);
        return 1;
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads\n",
                argv[0], p.fname_inp.c_str(),
                (int)samples.size(), (float)samples.size() / 16000.0f,
                p.n_threads);
    }

    const int SR = 16000;

    // -----------------------------------------------------------------------
    // Build list of audio slices to transcribe.
    // Each slice has: [sample_start, sample_end), absolute t0_cs/t1_cs.
    // Without VAD: one slice = full audio.
    // With VAD: one slice per speech segment.
    // -----------------------------------------------------------------------
    struct audio_slice {
        int     start;    // sample index
        int     end;      // sample index (exclusive)
        int64_t t0_cs;    // absolute start, centiseconds
        int64_t t1_cs;    // absolute end,   centiseconds
    };
    std::vector<audio_slice> slices;

    struct whisper_vad_context * vctx = nullptr;

    if (p.timestamps && !p.vad_model.empty()) {
        // Init Silero VAD
        struct whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = p.n_threads;
        vctx = whisper_vad_init_from_file_with_params(p.vad_model.c_str(), vcp);
        if (!vctx) {
            fprintf(stderr, "%s: warning: failed to load VAD model '%s', falling back to single segment\n",
                    argv[0], p.vad_model.c_str());
        }
    }

    if (vctx) {
        struct whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold             = p.vad_thold;
        vp.min_speech_duration_ms  = p.vad_min_speech_ms;
        vp.min_silence_duration_ms = p.vad_min_silence_ms;
        vp.speech_pad_ms           = p.vad_speech_pad_ms;

        struct whisper_vad_segments * vseg =
            whisper_vad_segments_from_samples(vctx, vp, samples.data(), (int)samples.size());

        if (!vseg || whisper_vad_segments_n_segments(vseg) == 0) {
            fprintf(stderr, "%s: VAD found no speech segments\n", argv[0]);
            whisper_vad_free_segments(vseg);
            whisper_vad_free(vctx);
            cohere_free(ctx);
            return 1;
        }

        int n_vad = whisper_vad_segments_n_segments(vseg);
        if (p.verbosity >= 1)
            fprintf(stderr, "%s: VAD detected %d speech segment(s)\n", argv[0], n_vad);

        for (int i = 0; i < n_vad; i++) {
            float t0_s = whisper_vad_segments_get_segment_t0(vseg, i);
            float t1_s = whisper_vad_segments_get_segment_t1(vseg, i);
            int start = std::max(0, (int)(t0_s * SR));
            int end   = std::min((int)samples.size(), (int)(t1_s * SR));
            if (end <= start) continue;
            slices.push_back({start, end,
                              (int64_t)(t0_s * 100.0f),
                              (int64_t)(t1_s * 100.0f)});
        }
        whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);
        vctx = nullptr;
    } else {
        // Single slice — full audio
        int64_t dur_cs = (int64_t)((double)samples.size() / SR * 100.0);
        slices.push_back({0, (int)samples.size(), 0, dur_cs});
    }

    // -----------------------------------------------------------------------
    // Transcribe all slices and accumulate display segments
    // -----------------------------------------------------------------------
    std::vector<disp_segment> all_segs;
    std::string plain_text;

    for (const auto & sl : slices) {
        struct cohere_result * r = cohere_transcribe_ex(
                ctx,
                samples.data() + sl.start,
                sl.end - sl.start,
                p.language.c_str(),
                sl.t0_cs);

        if (!r) {
            fprintf(stderr, "%s: transcription failed for segment [%s --> %s]\n",
                    argv[0],
                    to_timestamp(sl.t0_cs).c_str(),
                    to_timestamp(sl.t1_cs).c_str());
            continue;
        }

        if (p.timestamps) {
            auto segs = make_disp_segments(r, sl.t0_cs, sl.t1_cs, p.max_len);
            for (auto & ds : all_segs) (void)ds;  // suppress unused
            all_segs.insert(all_segs.end(), segs.begin(), segs.end());
        } else {
            // No timestamps: accumulate plain text
            if (r->text) {
                if (!plain_text.empty()) plain_text += " ";
                plain_text += r->text;
            }
        }

        cohere_result_free(r);
    }

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------

    if (p.timestamps) {
        // Print to stdout
        for (const auto & ds : all_segs)
            print_disp_segment(stdout, ds, /*show_ts=*/true, p.print_colors, /*to_file=*/false);

        // SRT
        if (p.output_srt)
            write_srt(make_output_path(p.fname_inp, ".srt"), all_segs, p.verbosity, argv[0]);

        // VTT
        if (p.output_vtt)
            write_vtt(make_output_path(p.fname_inp, ".vtt"), all_segs, p.verbosity, argv[0]);

        // Plain .txt (no timestamps, just text)
        if (p.output_txt) {
            const std::string txt_path = make_output_path(p.fname_inp, ".txt");
            std::ofstream ofs(txt_path);
            if (ofs) {
                for (const auto & ds : all_segs) ofs << ds.text << "\n";
                if (p.verbosity >= 1)
                    fprintf(stderr, "%s: transcript written to '%s'\n", argv[0], txt_path.c_str());
            } else {
                fprintf(stderr, "%s: warning: could not write to '%s'\n", argv[0], txt_path.c_str());
            }
        }
    } else {
        // Plain output (no timestamps)
        if (p.print_colors) {
            // Re-run but print colored tokens inline — need results again.
            // Since we already freed results above, we redo, but only for single segment.
            // For simplicity: color output requires -ts (print_colors without -ts
            // falls back to plain text with a warning).
            fprintf(stderr, "%s: note: --print-colors requires --timestamps (-ts); use -ts -pc\n", argv[0]);
        }
        printf("%s\n", plain_text.c_str());

        if (p.output_txt) {
            const std::string txt_path = make_output_path(p.fname_inp, ".txt");
            std::ofstream ofs(txt_path);
            if (ofs) {
                ofs << plain_text << "\n";
                if (p.verbosity >= 1)
                    fprintf(stderr, "%s: transcript written to '%s'\n", argv[0], txt_path.c_str());
            } else {
                fprintf(stderr, "%s: warning: could not write to '%s'\n", argv[0], txt_path.c_str());
            }
        }
    }

    if (p.verbosity >= 1)
        fprintf(stderr, "\n%s: done\n", argv[0]);

    cohere_free(ctx);
    return 0;
}
