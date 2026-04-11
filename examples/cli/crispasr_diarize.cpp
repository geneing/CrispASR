// crispasr_diarize.cpp — implementation of the generic diarize post-step.
// See crispasr_diarize.h for the interface contract.

#include "crispasr_diarize.h"
#include "whisper_params.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Convert an absolute centisecond timestamp into the per-channel sample
// index inside the current slice. Slice-relative because the energy
// arrays cover only [t0_cs, t1_cs), not the whole input.
inline int64_t cs_to_sample_in_slice(int64_t cs_abs, int64_t slice_t0_cs) {
    int64_t cs_local = cs_abs - slice_t0_cs;
    if (cs_local < 0) cs_local = 0;
    return (cs_local * 16000) / 100;
}

// -----------------------------------------------------------------------
// Method 1: energy-based comparison
// -----------------------------------------------------------------------
//
// For each segment, sum |L[i]| and |R[i]| across the segment's sample
// range and pick the louder channel. Margin = 1.1× to avoid flapping
// on near-equal energy. Same threshold the historical whisper-cli
// `--diarize` path uses, kept for consistency with downstream tools
// that grep for the literal "(speaker 0)" / "(speaker 1)" prefix.
void apply_energy(
    const float * left, const float * right, int n_samples,
    int64_t slice_t0_cs,
    std::vector<crispasr_segment> & segs)
{
    for (auto & seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1, slice_t0_cs);
        if (is0 < 0) is0 = 0;
        if (is1 > n_samples) is1 = n_samples;
        if (is0 >= is1) continue;

        double e_l = 0.0, e_r = 0.0;
        for (int64_t j = is0; j < is1; j++) {
            e_l += std::fabs((double)left [j]);
            e_r += std::fabs((double)right[j]);
        }
        std::string spk;
        if      (e_l > 1.1 * e_r) spk = "0";
        else if (e_r > 1.1 * e_l) spk = "1";
        else                      spk = "?";
        seg.speaker = "(speaker " + spk + ") ";
    }
}

// -----------------------------------------------------------------------
// Method 2: cross-correlation lag (TDOA-style)
// -----------------------------------------------------------------------
//
// Compute the cross-correlation of L and R within each segment over a
// short search window (±5 ms = ±80 samples at 16 kHz, generous enough
// to cover the head-shadow inter-aural delay for any normal mic
// spacing). The lag at the correlation peak's sign tells us which
// channel the voice is closest to: positive lag = peak when L is
// shifted ahead = voice arrived at R first = speaker 1; negative lag
// = speaker 0. Falls back to energy when the peak isn't strong enough
// (e.g. silent segment).
void apply_xcorr(
    const float * left, const float * right, int n_samples,
    int64_t slice_t0_cs,
    std::vector<crispasr_segment> & segs)
{
    constexpr int MAX_LAG = 80; // ±5 ms at 16 kHz
    for (auto & seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1, slice_t0_cs);
        if (is0 < 0) is0 = 0;
        if (is1 > n_samples) is1 = n_samples;
        if (is1 - is0 < 2 * MAX_LAG) {
            // Segment is too short to estimate a lag; fall back to
            // single-frame energy comparison so we still emit something.
            double e_l = 0.0, e_r = 0.0;
            for (int64_t j = is0; j < is1; j++) {
                e_l += std::fabs((double)left[j]);
                e_r += std::fabs((double)right[j]);
            }
            seg.speaker = (e_l >= e_r) ? "(speaker 0) " : "(speaker 1) ";
            continue;
        }

        const int64_t hi = is1 - MAX_LAG;
        const int64_t lo = is0 + MAX_LAG;
        double best = -1e30;
        int    best_lag = 0;
        for (int lag = -MAX_LAG; lag <= MAX_LAG; lag++) {
            double sum = 0.0;
            for (int64_t j = lo; j < hi; j++) {
                sum += (double)left[j] * (double)right[j + lag];
            }
            if (sum > best) { best = sum; best_lag = lag; }
        }
        std::string spk;
        if      (best_lag <  0) spk = "0";
        else if (best_lag >  0) spk = "1";
        else                    spk = "?";
        seg.speaker = "(speaker " + spk + ") ";
    }
}

} // namespace

// -----------------------------------------------------------------------
// Method 3: VAD-turn segmentation (mono-friendly)
// -----------------------------------------------------------------------
//
// Walk the segments in time order and assign a new "(speaker N)" label
// every time we see a gap > MIN_TURN_GAP_CS centiseconds since the
// previous segment. This is a "speaker turn" detector — it's not real
// speaker identification, just a useful proxy for "the conversation
// changed track here". Works on any input regardless of channel count
// because it only looks at segment timestamps. Default min gap is
// 60 cs (= 600 ms), which is the conventional pause threshold used by
// pyannote / NeMo for natural conversation turns.
namespace {
constexpr int64_t MIN_TURN_GAP_CS = 60;
} // namespace

void apply_vad_turns(std::vector<crispasr_segment> & segs) {
    if (segs.empty()) return;
    int speaker = 0;
    int64_t prev_t1 = -1;
    for (auto & seg : segs) {
        if (prev_t1 >= 0 && (seg.t0 - prev_t1) > MIN_TURN_GAP_CS) {
            speaker = 1 - speaker; // alternate 0 / 1
        }
        seg.speaker = "(speaker " + std::to_string(speaker) + ") ";
        prev_t1 = seg.t1;
    }
}

bool crispasr_apply_diarize(
    const std::vector<float> & left,
    const std::vector<float> & right,
    bool                       is_stereo,
    int64_t                    slice_t0_cs,
    std::vector<crispasr_segment> & segs,
    const whisper_params & params)
{
    if (segs.empty()) return true;

    // Method dispatch. Default depends on whether we have stereo:
    //   stereo input  -> "energy"  (cheap, accurate per channel)
    //   mono input    -> "vad-turns" (cheap, mono-friendly, just turns)
    std::string method = params.diarize_method;
    if (method.empty()) {
        method = is_stereo ? "energy" : "vad-turns";
    }

    if (method == "energy") {
        if (!is_stereo) {
            fprintf(stderr,
                    "crispasr[diarize]: --diarize-method energy needs stereo input — "
                    "falling back to vad-turns for this mono clip\n");
            apply_vad_turns(segs);
            return true;
        }
        apply_energy(left.data(), right.data(), (int)left.size(),
                     slice_t0_cs, segs);
        return true;
    }
    if (method == "xcorr" || method == "cross-correlation") {
        if (!is_stereo) {
            fprintf(stderr,
                    "crispasr[diarize]: --diarize-method xcorr needs stereo input — "
                    "falling back to vad-turns for this mono clip\n");
            apply_vad_turns(segs);
            return true;
        }
        apply_xcorr(left.data(), right.data(), (int)left.size(),
                    slice_t0_cs, segs);
        return true;
    }
    if (method == "vad-turns" || method == "turns") {
        apply_vad_turns(segs);
        return true;
    }
    if (method == "sherpa" || method == "sherpa-onnx") {
        fprintf(stderr,
                "crispasr[diarize]: --diarize-method sherpa needs the optional\n"
                "                   sherpa-onnx integration (build with\n"
                "                   -DCRISPASR_SHERPA=ON). Not yet wired —\n"
                "                   falling back to %s.\n",
                is_stereo ? "energy" : "vad-turns");
        if (is_stereo) apply_energy(left.data(), right.data(),
                                    (int)left.size(), slice_t0_cs, segs);
        else           apply_vad_turns(segs);
        return true;
    }
    if (method == "pyannote" || method == "ecapa") {
        fprintf(stderr,
                "crispasr[diarize]: --diarize-method '%s' is not implemented yet.\n"
                "                   pyannote v3 segmentation is MIT-licensed and a\n"
                "                   native GGUF port is on the TODO. Falling back\n"
                "                   to %s for this run.\n",
                method.c_str(),
                is_stereo ? "energy" : "vad-turns");
        if (is_stereo) apply_energy(left.data(), right.data(),
                                    (int)left.size(), slice_t0_cs, segs);
        else           apply_vad_turns(segs);
        return true;
    }

    fprintf(stderr,
            "crispasr[diarize]: unknown --diarize-method '%s' "
            "(expected energy|xcorr|vad-turns|sherpa|pyannote|ecapa)\n",
            method.c_str());
    return false;
}
