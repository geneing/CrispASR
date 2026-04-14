// crispasr_lid.h — optional language identification pre-step.
//
// Several backends in the crispasr lineup can't auto-detect language on
// their own (cohere is English-only anyway, canary/granite/voxtral/
// voxtral4b all require an explicit ISO code up-front). For those the
// dispatch layer in crispasr_run.cpp can run a lightweight LID pass
// before transcribe() and fill in params.language / params.source_lang
// based on the result.
//
// Two backends are supported:
//
//   * "whisper"  — uses the whisper.cpp encoder + language head on a
//                  multilingual ggml-*.bin model (default ggml-tiny.bin).
//                  Pure C++, zero extra dependencies beyond what crispasr
//                  already links in. Fastest, most accurate of the two
//                  for trained languages.
//
//   * "silero"   — subprocess shim that invokes a short Python one-liner
//                  loading silero-lang-detector via torch.hub. Slower
//                  (Python startup) but works without a whisper weights
//                  file and covers Silero's trained language set. Only
//                  available where python3 + torch are installed.
//
// Select via `--lid-backend NAME`. When --lid-backend is omitted but the
// active transcription backend can't detect language natively, the
// default is "whisper" (model ggml-tiny.bin). Passing an empty string
// disables the pre-step entirely and lets backends that don't support
// lang-detect fall back to params.language as usual.

#pragma once

#include <string>

struct whisper_params; // fwd

struct crispasr_lid_result {
    std::string lang_code;    // ISO 639-1 (e.g. "en", "de"). Empty on failure.
    float confidence = -1.0f; // [0, 1], -1 if unknown
    std::string source;       // "whisper" | "silero" | ""
};

// Run language detection on a 16 kHz mono PCM buffer. `params.lid_backend`
// picks the implementation; `params.lid_model` optionally overrides the
// model path (empty = sensible default). Returns true on success, false
// on any failure (reason printed to stderr unless params.no_prints).
bool crispasr_detect_language(const float* samples, int n_samples, const whisper_params& params,
                              crispasr_lid_result& out);
