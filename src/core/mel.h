// src/core/mel.h — shared log-mel spectrogram computation.
//
// Replaces the 9 copy-pasted mel spectrogram implementations across the
// src/ model files (parakeet.cpp, canary.cpp, canary_ctc.cpp, cohere.cpp,
// qwen3_asr.cpp, voxtral.cpp, voxtral4b.cpp, granite_speech.cpp,
// whisper.cpp). Two algorithm clusters are supported via enums:
//
//   NeMo / Conformer style  — ln + per-mel z-score, (T, n_mels) output
//       used by parakeet, canary, canary_ctc, cohere
//
//   Whisper / HF style      — log10 + global clip (max-8+4)/4, (n_mels, T) output
//       used by whisper, qwen3, voxtral, voxtral4b, granite
//
// The function is parameterised rather than having two entry points
// because the STFT + mel projection steps are identical; only the log
// base, normalization, and output transpose differ. Keeping them in one
// code path means numerical differences between clusters stay localised
// to the post-processing step, not the heavy computation.
//
// Models continue to own their own FFT function — it's passed in as a
// function pointer so we don't have to unify the 9 near-identical
// Cooley-Tukey implementations in this first pass. (They can be
// consolidated in a follow-up; the win there is small compared to the
// mel extraction itself.)

#pragma once

#include <cstdint>
#include <vector>

namespace core_mel {

// Real-to-complex FFT callback signature. N is always a power of two.
// Output layout: interleaved (re, im) pairs, length 2*N floats.
// Each model passes its own FFT so we don't disturb numerical paths.
using FftR2C = void (*)(const float * in, int N, float * out);

enum class LogBase { Ln, Log10 };

enum class Normalization {
    // Per-mel band z-score across time: (x - mean) / sqrt(var + 1e-5).
    // Used by parakeet / canary / canary_ctc / cohere.
    PerFeatureZ,

    // Global clip-and-scale: y = (max(x, max(x)-8) + 4) / 4.
    // Used by whisper / qwen3 / voxtral / granite.
    GlobalClipMax,

    // Global clip with fixed ceiling (max-like value is baked into the
    // normalization rather than computed): y = (max(x, fixed_max-8) + 4) / 4.
    // Used by voxtral4b with fixed_max = 1.5.
    GlobalClipFixed,
};

enum class Layout {
    // Row-major (T, n_mels) — each frame's n_mels values contiguous.
    // Used by the NeMo cluster.
    TimeMels,

    // Row-major (n_mels, T) — each mel band's full time series contiguous.
    // Used by the HF/Whisper cluster.
    MelsTime,
};

struct Params {
    int n_fft       = 400;  // power-of-two FFT size
    int hop_length  = 160;  // frame stride in samples
    int win_length  = 400;  // window length, must be <= n_fft
    int n_mels      = 128;

    LogBase       log_base  = LogBase::Log10;
    Normalization norm      = Normalization::GlobalClipMax;
    Layout        layout    = Layout::MelsTime;

    // Small positive constant added before log() to avoid log(0).
    // NeMo uses 2^-24; Whisper uses 1e-10. Pass what the model originally used.
    float log_eps = 1e-10f;

    // For Normalization::GlobalClipFixed: the fixed ceiling used in place
    // of the per-audio max. Ignored for other normalization modes.
    float fixed_max = 1.5f;

    // Apply symmetric zero-padding of n_fft/2 samples before/after the input
    // (matches torchaudio / NeMo center=True). Set false if the caller has
    // already padded the input.
    bool center_pad = true;

    // Pad the output to exactly this many frames (zero-padding on the right).
    // Set to 0 for "don't pad". Voxtral 3B uses 3000 (= 30s at hop=160).
    int pad_to_T = 0;
};

// Compute log-mel spectrogram from raw PCM samples.
//
//   samples      : float32 PCM at the caller's sample rate (usually 16 kHz)
//   n_samples    : sample count
//   window       : float32[n_fft] Hann/Hamming window padded with zeros if
//                  win_length < n_fft (caller's responsibility). When the
//                  model stores only the win_length-sized window in its GGUF,
//                  this helper pads it inside compute().
//   mel_fb       : float32[n_mels * n_freqs] row-major filterbank with
//                  n_freqs = n_fft/2 + 1
//   fft          : model-specific FFT function pointer (see FftR2C above)
//   params       : configuration (see Params struct)
//   T_out [out]  : number of output frames
//
// Returns the flat log-mel buffer in the layout specified by params.layout.
std::vector<float> compute(
    const float * samples, int n_samples,
    const float * window,     // length win_length (we center-pad inside to n_fft)
    int           win_length,
    const float * mel_fb,     // [n_mels, n_freqs]
    int           n_freqs,
    FftR2C        fft,
    const Params & params,
    int         & T_out);

} // namespace core_mel
