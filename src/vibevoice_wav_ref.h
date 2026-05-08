// vibevoice_wav_ref.h - helpers for VibeVoice voice-reference WAV input.
//
// Pure helpers so the clone-path parsing and normalization behaviour can be
// tested without loading a VibeVoice model.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

inline uint16_t vibevoice_wav_rd_u16(const uint8_t* data, std::size_t off) {
    return (uint16_t)data[off] | ((uint16_t)data[off + 1] << 8);
}

inline uint32_t vibevoice_wav_rd_u32(const uint8_t* data, std::size_t off) {
    return (uint32_t)data[off] | ((uint32_t)data[off + 1] << 8) | ((uint32_t)data[off + 2] << 16) |
           ((uint32_t)data[off + 3] << 24);
}

inline bool vibevoice_parse_mono_pcm16_wav(const uint8_t* data, std::size_t size, std::vector<float>& pcm) {
    pcm.clear();
    if (!data || size < 12)
        return false;
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0)
        return false;

    uint32_t audio_format = 0;
    uint32_t channels = 0;
    uint32_t bits_per_sample = 0;
    std::size_t data_offset = 0;
    uint32_t data_size = 0;

    std::size_t offset = 12; // skip RIFF, size, WAVE
    while (offset + 8 <= size) {
        const uint32_t chunk_sz = vibevoice_wav_rd_u32(data, offset + 4);
        const std::size_t chunk_data = offset + 8;
        if (chunk_data + (std::size_t)chunk_sz > size)
            return false;

        if (std::memcmp(data + offset, "fmt ", 4) == 0 && chunk_sz >= 16) {
            audio_format = vibevoice_wav_rd_u16(data, chunk_data);
            channels = vibevoice_wav_rd_u16(data, chunk_data + 2);
            bits_per_sample = vibevoice_wav_rd_u16(data, chunk_data + 14);
        } else if (std::memcmp(data + offset, "data", 4) == 0) {
            data_offset = chunk_data;
            data_size = chunk_sz;
        }

        offset = chunk_data + chunk_sz + (chunk_sz & 1u);
    }

    if (data_offset == 0 || audio_format != 1 || channels != 1 || bits_per_sample != 16)
        return false;
    if (data_offset + (std::size_t)data_size > size || (data_size & 1u) != 0)
        return false;

    const std::size_t n_samples = (std::size_t)data_size / 2;
    pcm.resize(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        const std::size_t off = data_offset + i * 2;
        const int16_t s = (int16_t)vibevoice_wav_rd_u16(data, off);
        pcm[i] = (float)s / 32768.0f;
    }
    return true;
}

inline void vibevoice_normalize_ref_pcm(std::vector<float>& pcm, float target_dbfs = -25.0f) {
    if (pcm.empty())
        return;

    float rms = 0.0f;
    for (float sample : pcm)
        rms += sample * sample;
    rms = std::sqrt(rms / (float)pcm.size());
    const float target_rms = std::pow(10.0f, target_dbfs / 20.0f);
    const float scalar = target_rms / (rms + 1e-6f);

    float max_val = 0.0f;
    for (float& sample : pcm) {
        sample *= scalar;
        max_val = std::max(max_val, std::fabs(sample));
    }

    // Uniform scaling avoids hard-clipping distortion while preserving the
    // reference waveform shape.
    if (max_val > 1.0f) {
        const float clip_scalar = max_val + 1e-6f;
        for (float& sample : pcm)
            sample /= clip_scalar;
    }
}
