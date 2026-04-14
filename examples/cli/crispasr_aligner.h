// crispasr_aligner.h — shared CTC forced-alignment helper.
//
// LLM-based backends (qwen3, voxtral, voxtral4b, granite) emit plain text
// without per-word timestamps. A second pass through the canary CTC
// aligner produces frame-aligned word timings over a 16k SentencePiece
// vocabulary that covers 25+ European languages. Works with any
// transcript/audio pair — the aligner re-tokenises the words through
// its own vocab before running Viterbi.
//
// See src/canary_ctc.h for the underlying library API. This helper wraps
// it into a crispasr_word vector so backends can attach timestamps to
// their segments with one function call.

#pragma once

#include "crispasr_backend.h"

#include <string>
#include <vector>

// Run CTC forced alignment on a transcript + audio pair.
//
// aligner_model: path to canary-ctc-aligner GGUF file
// transcript:    plain text produced by the backend
// samples:       16 kHz mono PCM of the audio slice
// n_samples:     sample count
// t_offset_cs:   absolute start of this slice in centiseconds (added to
//                each returned word's t0/t1)
// n_threads:     inference thread count
//
// Returns a vector of crispasr_word with absolute (t0, t1) centisecond
// timestamps. Returns an empty vector on failure (with an error message
// on stderr).
//
// The aligner model is loaded and freed inside each call. For batch use
// we may want to cache the context later, but for typical "one file at
// a time" CLI usage the load cost is negligible compared to LLM decode.
std::vector<crispasr_word> crispasr_ctc_align(const std::string& aligner_model, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs, int n_threads);
