// crispasr_run.cpp — top-level dispatch for non-whisper backends.
//
// Called from cli.cpp main() when params.backend is a non-whisper backend.
// Drives the pipeline: resolve model -> detect backend -> load audio ->
// segment via VAD (or fixed chunks) -> transcribe -> print + write outputs.
//
// The whisper code path in cli.cpp is left completely untouched so the
// historical whisper-cli behaviour is bit-identical.

#include "crispasr_backend.h"
#include "crispasr_vad.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr.h"
#include "crispasr_aligner.h"
#include "crispasr_lid.h"
#include "crispasr_diarize.h"
#include "whisper_params.h"

#include "common-whisper.h" // read_audio_data

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

// Capability-vs-request check. For each requested feature, warn on stderr
// when the backend doesn't support it. Not fatal — the feature is silently
// ignored. Returns the number of warnings emitted.
int warn_unsupported(const CrispasrBackend & backend, const whisper_params & p) {
    const uint32_t caps = backend.capabilities();
    int warns = 0;

    auto warn = [&](const char * feature) {
        fprintf(stderr,
                "crispasr: warning: backend '%s' does not support %s — ignoring\n",
                backend.name(), feature);
        warns++;
    };

    // Diarize is now handled at the dispatcher level via the generic
    // crispasr_apply_diarize() post-step (energy / xcorr / future
    // pyannote / ecapa), so no warning even when the backend itself
    // doesn't claim CAP_DIARIZE — the dispatcher will label the
    // segments after transcribe() returns. Tinydiarize still requires
    // backend support (whisper-only).
    if (p.tinydiarize  && !(caps & CAP_DIARIZE))          warn("--tinydiarize");
    if (p.translate    && !(caps & CAP_TRANSLATE))        warn("--translate");
    if (!p.grammar.empty() && !(caps & CAP_GRAMMAR))      warn("--grammar");
    if (p.temperature != 0.0f && !(caps & CAP_TEMPERATURE)) warn("--temperature");
    if (!p.punctuation && !(caps & CAP_PUNCTUATION_TOGGLE)) warn("--no-punctuation");
    if (!p.source_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--source-lang");
    if (!p.target_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--target-lang");
    if (p.n_processors > 1 && !(caps & CAP_PARALLEL_PROCESSORS))
        warn("--processors > 1");

    return warns;
}

// Merge individual-slice results into a flat list preserving time order.
std::vector<crispasr_segment> merge_segments(
    std::vector<std::vector<crispasr_segment>> && per_slice)
{
    std::vector<crispasr_segment> out;
    size_t total = 0;
    for (auto & v : per_slice) total += v.size();
    out.reserve(total);
    for (auto & v : per_slice) {
        for (auto & s : v) out.push_back(std::move(s));
    }
    return out;
}

} // namespace

int crispasr_run_backend(const whisper_params & params_in) {
    whisper_params params = params_in;

    // Resolve backend name: explicit --backend takes priority; otherwise
    // auto-detect from the GGUF file. Defaults are handled in cli.cpp.
    std::string backend_name = params.backend;
    if (backend_name.empty() || backend_name == "auto") {
        backend_name = crispasr_detect_backend_from_gguf(params.model);
        if (backend_name.empty()) {
            fprintf(stderr,
                    "crispasr: error: could not auto-detect backend from '%s'. "
                    "Use --backend NAME to force one.\n",
                    params.model.c_str());
            return 10;
        }
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: detected backend '%s' from GGUF metadata\n",
                    backend_name.c_str());
        }
    }

    // Resolve "-m auto" via the model registry + curl/wget download.
    const std::string resolved = crispasr_resolve_model(
        params.model, backend_name, params.no_prints);
    if (resolved.empty()) {
        return 11;
    }
    params.model = resolved;

    // Create and init the backend.
    std::unique_ptr<CrispasrBackend> backend = crispasr_create_backend(backend_name);
    if (!backend) {
        fprintf(stderr, "crispasr: error: backend '%s' is not available in this build\n",
                backend_name.c_str());
        return 12;
    }

    warn_unsupported(*backend, params);

    if (!backend->init(params)) {
        fprintf(stderr, "crispasr: error: failed to initialise backend '%s'\n",
                backend_name.c_str());
        return 13;
    }

    // Process every input file.
    int rc = 0;
    for (const auto & fname_inp : params.fname_inp) {
        std::vector<float> samples;
        std::vector<std::vector<float>> stereo;
        // Request stereo split when --diarize is set. Diarize is now
        // a generic dispatcher post-step (crispasr_diarize.cpp), so we
        // try it for every backend rather than only those that
        // advertise CAP_DIARIZE — the backend itself doesn't have to
        // know anything about stereo; the dispatcher labels its
        // segments after transcribe() returns.
        const bool want_stereo = params.diarize;
        if (!read_audio_data(fname_inp, samples, stereo, want_stereo)) {
            fprintf(stderr, "crispasr: error: failed to read audio '%s'\n",
                    fname_inp.c_str());
            rc = 20;
            continue;
        }
        bool have_stereo = want_stereo &&
            stereo.size() == 2 &&
            !stereo[0].empty() &&
            stereo[0].size() == stereo[1].size();
        // miniaudio duplicates mono -> both channels when we ask for
        // stereo, so a mono input file gives us pcmf32s[0] == pcmf32s[1].
        // Detect that and downgrade to mono so the diarize post-step
        // takes the mono-friendly path (vad-turns) instead of the
        // tie-only energy path.
        if (have_stereo) {
            const size_t n = stereo[0].size();
            const size_t check = std::min<size_t>(n, 4096);
            bool channels_equal = true;
            for (size_t i = 0; i < check; i++) {
                if (stereo[0][i] != stereo[1][i]) { channels_equal = false; break; }
            }
            if (channels_equal) have_stereo = false;
        }

        constexpr int SR = 16000;
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n",
                    (int)samples.size(),
                    (double)samples.size() / SR, SR, params.n_threads);
        }

        // Optional language-identification pre-step. Fires only when the
        // user asked for auto language (either --detect-language or
        // --language auto) AND the chosen backend can't detect language
        // natively (qwen3/whisper/parakeet already do). The detected ISO
        // code is written into `params.language` and, if empty, into
        // `params.source_lang` so canary can pick it up as well.
        const bool want_auto_lang = params.detect_language ||
                                    params.language == "auto";
        const bool has_native_lid = (backend->capabilities() & CAP_LANGUAGE_DETECT) != 0;
        const bool lid_disabled   = params.lid_backend == "off" ||
                                    params.lid_backend == "none";
        if (want_auto_lang && !has_native_lid && !lid_disabled) {
            crispasr_lid_result lid;
            if (crispasr_detect_language(samples.data(), (int)samples.size(),
                                          params, lid)) {
                params.language = lid.lang_code;
                if (params.source_lang.empty()) {
                    params.source_lang = lid.lang_code;
                }
                if (!params.no_prints) {
                    fprintf(stderr,
                            "crispasr: LID -> language = '%s' (%s, p=%.3f)\n",
                            lid.lang_code.c_str(), lid.source.c_str(),
                            lid.confidence);
                }
            } else if (!params.no_prints) {
                fprintf(stderr,
                        "crispasr: LID failed, falling back to params.language='%s'\n",
                        params.language.c_str());
            }
        }

        // Slice into chunks (VAD or fixed-window fallback).
        const auto slices = crispasr_compute_audio_slices(
            samples.data(), (int)samples.size(), SR,
            params.chunk_seconds, params);

        if (slices.empty()) {
            fprintf(stderr, "crispasr: warning: no speech detected in '%s'\n",
                    fname_inp.c_str());
            continue;
        }

        if (!params.no_prints && slices.size() > 1) {
            fprintf(stderr, "crispasr: processing %zu slice(s)\n", slices.size());
        }

        // Transcribe each slice.
        std::vector<std::vector<crispasr_segment>> per_slice;
        per_slice.reserve(slices.size());
        for (size_t i = 0; i < slices.size(); i++) {
            const auto & sl = slices[i];
            // Always transcribe in mono — every backend takes mono PCM
            // and the diarize step happens later as a generic post-pass.
            std::vector<crispasr_segment> segs = backend->transcribe(
                samples.data() + sl.start,
                sl.end - sl.start,
                sl.t0_cs,
                params);

            // Apply the generic diarize post-step. Stereo-only methods
            // (energy, xcorr) need have_stereo == true; mono-friendly
            // methods (vad-turns, future sherpa/pyannote) work either
            // way. Pass both channel buffers and an is_stereo hint;
            // when have_stereo is false we point both at the mono
            // buffer so the helper has data to look at without
            // special-casing.
            if (params.diarize && !segs.empty()) {
                if (have_stereo) {
                    std::vector<float> sl_l(stereo[0].begin() + sl.start,
                                            stereo[0].begin() + sl.end);
                    std::vector<float> sl_r(stereo[1].begin() + sl.start,
                                            stereo[1].begin() + sl.end);
                    crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true,
                                           sl.t0_cs, segs, params);
                } else {
                    std::vector<float> mono_slice(samples.begin() + sl.start,
                                                  samples.begin() + sl.end);
                    crispasr_apply_diarize(mono_slice, mono_slice,
                                           /*is_stereo=*/false,
                                           sl.t0_cs, segs, params);
                }
            }

            // Optional CTC forced alignment to attach word-level timestamps.
            // Applies to backends that expose CAP_TIMESTAMPS_CTC and don't
            // already have words populated. Runs per slice so absolute
            // timestamps come out right.
            const bool want_align =
                !params.aligner_model.empty() &&
                (backend->capabilities() & CAP_TIMESTAMPS_CTC);
            if (want_align) {
                for (auto & seg : segs) {
                    if (!seg.words.empty()) continue; // already aligned
                    auto words = crispasr_ctc_align(
                        params.aligner_model,
                        seg.text,
                        samples.data() + sl.start,
                        sl.end - sl.start,
                        sl.t0_cs,
                        params.n_threads);
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }

            per_slice.push_back(std::move(segs));
        }
        auto all_segs = merge_segments(std::move(per_slice));

        // Optional post-processing: strip punctuation when --no-punctuation
        // is set. Cohere and canary pass p.punctuation through to their C
        // APIs natively and will usually return text that's already clean,
        // but this second pass is idempotent so the double application is
        // harmless. For the LLM backends (voxtral/voxtral4b/qwen3/granite)
        // this is the only way punctuation control happens — the models
        // don't take a "no punctuation" flag, they just generate whatever
        // the prompt pushes them towards.
        if (!params.punctuation) {
            for (auto & seg : all_segs) {
                crispasr_strip_punctuation(seg);
            }
        }

        // Build display segments.
        const auto disp = crispasr_make_disp_segments(all_segs, params.max_len);

        // Print to stdout.
        const bool show_timestamps =
            !params.no_timestamps &&
            (params.output_srt || params.output_vtt ||
             params.max_len > 0  || params.print_colors ||
             params.diarize);
        crispasr_print_stdout(disp, show_timestamps);

        // Write output files.
        if (params.output_txt)
            crispasr_write_txt(crispasr_make_out_path(fname_inp, ".txt"), disp);
        if (params.output_srt)
            crispasr_write_srt(crispasr_make_out_path(fname_inp, ".srt"), disp);
        if (params.output_vtt)
            crispasr_write_vtt(crispasr_make_out_path(fname_inp, ".vtt"), disp);
        if (params.output_csv)
            crispasr_write_csv(crispasr_make_out_path(fname_inp, ".csv"), disp);
        if (params.output_lrc)
            crispasr_write_lrc(crispasr_make_out_path(fname_inp, ".lrc"), disp);
        if (params.output_jsn)
            crispasr_write_json(
                crispasr_make_out_path(fname_inp, ".json"),
                all_segs, backend->name(), params.model, params.language,
                params.output_jsn_full);
    }

    backend->shutdown();
    return rc;
}
