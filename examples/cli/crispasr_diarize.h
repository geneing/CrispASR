// crispasr_diarize.h — generic stereo speaker diarization post-step.
//
// Diarization used to live inside the whisper wrapper as a special case
// because whisper was the only backend that knew about stereo PCM.
// That was always the wrong layering: stereo→speaker is a property of
// the audio, not the transcription model. This module pulls the energy
// comparison out into a dispatcher-level post-step that runs AFTER any
// backend's transcribe() call, so every backend supports `--diarize`
// for free as long as the dispatcher feeds it the L/R channel buffers.
//
// Same shape as crispasr_lid.h: a single entry point + a small enum
// for the method, easy to extend with new diarizers.
//
// Implemented diarizers (all pure C++, no extra deps):
//
//   * "energy"   — default. For each segment, computes |L| and |R|
//                  energy in the segment's time range and labels the
//                  segment "(speaker 0)" or "(speaker 1)" based on
//                  which channel dominates. Same algorithm as the
//                  historical whisper-cli `--diarize`. Works only on
//                  proper stereo input where each channel is one
//                  speaker.
//
//   * "xcorr"    — cross-correlation. Computes the lag at peak
//                  cross-correlation between L and R and labels the
//                  segment based on the sign of the lag (positive =
//                  voice-arrives-at-L-first = "0", negative = "1").
//                  Slightly more robust than pure energy when both
//                  speakers are picked up by both mics at different
//                  delays (typical conference-room setup).
//
// Future / pending:
//
//   * "pyannote" — Pyannote.audio segmentation ported to GGUF (~5 MB
//                  ONNX -> GGUF). Single-channel, so works on mono
//                  input too. Tracked in TODO.md.
//
//   * "ecapa"    — SpeechBrain ECAPA-TDNN speaker embeddings + simple
//                  agglomerative clustering. Mono-friendly, more
//                  accurate but heavier (22 MB GGUF). Also TODO.

#pragma once

#include "crispasr_backend.h"

#include <string>
#include <vector>

struct whisper_params; // fwd

// Mutate `segs` in-place, filling each `seg.speaker` based on the
// configured diarizer. Returns false on unknown method; otherwise
// always succeeds (mono inputs are handled by mono-friendly methods,
// stereo-only methods degrade to a single-speaker label).
//
// `left` and `right` are per-channel slice buffers when stereo is
// available. For mono input, both vectors point at the same data
// and `is_stereo` is false; the dispatcher should call this anyway
// so the mono-friendly methods (vad-turns, sherpa, pyannote) can
// still run.
bool crispasr_apply_diarize(
    const std::vector<float> & left,
    const std::vector<float> & right,
    bool                       is_stereo,
    int64_t                    slice_t0_cs,
    std::vector<crispasr_segment> & segs,
    const whisper_params & params);
