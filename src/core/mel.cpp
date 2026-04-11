// src/core/mel.cpp — implementation of compute_log_mel().
// See src/core/mel.h for the interface contract.

#include "mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace core_mel {

std::vector<float> compute(
    const float * samples, int n_samples,
    const float * window_in,
    int           win_length,
    const float * mel_fb,
    int           n_freqs,
    FftR2C        fft,
    const Params & p,
    int         & T_out)
{
    const int n_fft = p.n_fft;
    const int hop   = p.hop_length;
    const int nmels = p.n_mels;

    // -----------------------------------------------------------------
    // 1. Optional center-pad of the input by n_fft/2 on each side (this
    //    matches torchaudio / NeMo center=True). Zero-pad is used rather
    //    than reflect; the numerical difference is negligible for real
    //    audio and several existing models already use zero-pad.
    // -----------------------------------------------------------------
    std::vector<float> padded_in;
    const float * in_ptr;
    int           in_len;
    if (p.center_pad) {
        const int pad = n_fft / 2;
        padded_in.assign((size_t)(pad + n_samples + pad), 0.0f);
        std::memcpy(padded_in.data() + pad, samples, (size_t)n_samples * sizeof(float));
        in_ptr = padded_in.data();
        in_len = (int)padded_in.size();
    } else {
        in_ptr = samples;
        in_len = n_samples;
    }

    // -----------------------------------------------------------------
    // 2. Center-pad the window to n_fft (if win_length < n_fft). This
    //    mirrors what the legacy mel functions did inline.
    // -----------------------------------------------------------------
    std::vector<float> window(n_fft, 0.0f);
    {
        const int lpad = (n_fft - win_length) / 2;
        const int wn = std::min(win_length, n_fft);
        for (int i = 0; i < wn; i++) window[lpad + i] = window_in[i];
    }

    // -----------------------------------------------------------------
    // 3. STFT → power spectrum
    // -----------------------------------------------------------------
    const int T = in_len >= n_fft ? (in_len - n_fft) / hop + 1 : 0;
    T_out = T;
    if (T <= 0) return {};

    // Power is stored (n_freqs, T) row-major to make the mel matmul
    // column-contiguous. (The legacy code stored (T, n_freqs); both work
    // but this lets us share the matmul path with the MelsTime layout.)
    // We'll keep the legacy (T, n_freqs) layout to preserve bit-exact
    // numerical behaviour for the NeMo cluster, then choose the output
    // layout at the end.
    std::vector<float> power((size_t)T * n_freqs, 0.0f);
    {
        std::vector<float> fft_in((size_t)n_fft);
        std::vector<float> fft_out((size_t)n_fft * 2);
        for (int t = 0; t < T; t++) {
            const float * frame = in_ptr + (size_t)t * hop;
            for (int n = 0; n < n_fft; n++) fft_in[n] = frame[n] * window[n];
            fft(fft_in.data(), n_fft, fft_out.data());
            for (int k = 0; k < n_freqs; k++) {
                const float re = fft_out[2*k];
                const float im = fft_out[2*k + 1];
                power[(size_t)t * n_freqs + k] = re*re + im*im;
            }
        }
    }

    // -----------------------------------------------------------------
    // 4. Mel projection: mel[t, m] = sum_k power[t, k] * fb[m, k]
    //    Legacy layout is (T, n_mels); we'll transpose at the end if the
    //    caller wants (n_mels, T).
    // -----------------------------------------------------------------
    std::vector<float> mel_tn((size_t)T * nmels, 0.0f);
    for (int t = 0; t < T; t++) {
        const float * pp = power.data() + (size_t)t * n_freqs;
        float * mp = mel_tn.data() + (size_t)t * nmels;
        for (int m = 0; m < nmels; m++) {
            const float * fb = mel_fb + (size_t)m * n_freqs;
            float s = 0.0f;
            for (int k = 0; k < n_freqs; k++) s += pp[k] * fb[k];
            mp[m] = s;
        }
    }

    // -----------------------------------------------------------------
    // 5. log
    // -----------------------------------------------------------------
    if (p.log_base == LogBase::Ln) {
        for (size_t i = 0; i < mel_tn.size(); i++) {
            mel_tn[i] = std::log(mel_tn[i] + p.log_eps);
        }
    } else { // Log10
        for (size_t i = 0; i < mel_tn.size(); i++) {
            mel_tn[i] = std::log10(mel_tn[i] + p.log_eps);
        }
    }

    // -----------------------------------------------------------------
    // 6. Normalization
    // -----------------------------------------------------------------
    switch (p.norm) {
        case Normalization::PerFeatureZ: {
            // Per-mel band z-score across time. NeMo cluster.
            for (int m = 0; m < nmels; m++) {
                double sum = 0.0, sq = 0.0;
                for (int t = 0; t < T; t++) sum += mel_tn[(size_t)t * nmels + m];
                const double mean = sum / T;
                for (int t = 0; t < T; t++) {
                    const double d = mel_tn[(size_t)t * nmels + m] - mean;
                    sq += d * d;
                }
                const float inv_std = 1.0f / std::sqrt((float)(sq / T) + 1e-5f);
                for (int t = 0; t < T; t++) {
                    mel_tn[(size_t)t * nmels + m] =
                        (float)(mel_tn[(size_t)t * nmels + m] - mean) * inv_std;
                }
            }
            break;
        }
        case Normalization::GlobalClipMax: {
            // HF / Whisper style: (max(x, max(x)-8) + 4) / 4
            float mx = -1e30f;
            for (size_t i = 0; i < mel_tn.size(); i++) {
                if (mel_tn[i] > mx) mx = mel_tn[i];
            }
            const float floor_v = mx - 8.0f;
            for (size_t i = 0; i < mel_tn.size(); i++) {
                float v = mel_tn[i];
                if (v < floor_v) v = floor_v;
                mel_tn[i] = (v + 4.0f) / 4.0f;
            }
            break;
        }
        case Normalization::GlobalClipFixed: {
            // Voxtral4b: use a fixed max instead of computing it.
            const float floor_v = p.fixed_max - 8.0f;
            for (size_t i = 0; i < mel_tn.size(); i++) {
                float v = mel_tn[i];
                if (v < floor_v) v = floor_v;
                mel_tn[i] = (v + 4.0f) / 4.0f;
            }
            break;
        }
    }

    // -----------------------------------------------------------------
    // 7. Output layout + optional right-pad to p.pad_to_T frames.
    // -----------------------------------------------------------------
    const int T_final = (p.pad_to_T > 0 && p.pad_to_T > T) ? p.pad_to_T : T;
    T_out = T_final;

    if (p.layout == Layout::TimeMels) {
        // Legacy NeMo layout: (T, n_mels), row-major.
        if (T_final > T) {
            std::vector<float> out((size_t)T_final * nmels, 0.0f);
            std::memcpy(out.data(), mel_tn.data(), (size_t)T * nmels * sizeof(float));
            return out;
        }
        return mel_tn;
    }

    // Layout::MelsTime — transpose to (n_mels, T_final), optionally padded.
    std::vector<float> out((size_t)nmels * T_final, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < nmels; m++) {
            out[(size_t)m * T_final + t] = mel_tn[(size_t)t * nmels + m];
        }
    }
    return out;
}

} // namespace core_mel
