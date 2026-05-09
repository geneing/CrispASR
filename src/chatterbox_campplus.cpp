// chatterbox_campplus.cpp — CAMPPlus speaker encoder forward (Module 4).
//
// See chatterbox_campplus.h for the full pipeline contract. This first
// commit ships the Kaldi fbank front-end and the per-utterance mean
// subtract; the FCM head + xvector chain land in a follow-up.

#include "chatterbox_campplus.h"

#include "core/kaldi_fbank.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace chatterbox_campplus {

std::vector<float> compute_fbank(const float* pcm_16k, int n_samples, int& T_frames_out) {
    T_frames_out = 0;
    if (!pcm_16k || n_samples <= 0)
        return {};

    core_kaldi::FbankParams p;
    p.sample_rate = 16000;
    p.n_mels = 80;
    p.frame_length_ms = 25;
    p.frame_shift_ms = 10;
    p.low_freq = 20.0f;
    p.high_freq = 0.0f; // Nyquist
    p.preemph = 0.97f;
    p.remove_dc_offset = true;
    p.int16_scale = false; // raw [-1, 1] — torchaudio's default consumed by CAMPPlus

    int T = 0;
    auto feats = core_kaldi::compute_fbank(pcm_16k, n_samples, p, T);
    if (feats.empty() || T <= 0) {
        return {};
    }

    // Per-utterance mean subtract along time (xvector.py:extract_feature).
    // feats is (T, 80) row-major. Compute mean over T for each mel bin, then
    // subtract from each row.
    const int n_mels = p.n_mels;
    std::vector<double> mean((size_t)n_mels, 0.0);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            mean[(size_t)m] += (double)feats[(size_t)t * (size_t)n_mels + (size_t)m];
        }
    }
    const double inv_T = 1.0 / (double)T;
    for (int m = 0; m < n_mels; m++)
        mean[(size_t)m] *= inv_T;
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            feats[(size_t)t * (size_t)n_mels + (size_t)m] -= (float)mean[(size_t)m];
        }
    }

    T_frames_out = T;
    return feats;
}

} // namespace chatterbox_campplus
