// crispasr_wav_writer.h — header-only WAV (16-bit PCM) serializer.
//
// Used by the TTS server route handler to wrap synthesised float32
// samples in a self-contained RIFF blob that browsers and OpenAI
// clients can play directly. Header-only so the unit tests can
// exercise it without linking the server translation unit.

#pragma once

#include <cmath>
#include <cstdint>
#include <string>

// Build a 16-bit PCM RIFF WAV from float32 samples in [-1, 1]. Mono,
// `sample_rate` Hz. Samples outside [-1, 1] are clamped before the
// int16 conversion (avoids wraparound from std::lround). The 44-byte
// RIFF/fmt/data header uses standard PCM format (1) and the
// little-endian byte order RFC 2361 prescribes — our serializer writes
// the bytes by hand so it's identical on both endian platforms (the
// project ships only LE today, but the explicit byte ordering keeps
// us portable).
inline std::string crispasr_make_wav_int16(const float* pcm, int n_samples, int sample_rate) {
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = (uint32_t)sample_rate * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t data_size = (uint32_t)(n_samples > 0 ? n_samples : 0) * block_align;
    const uint32_t riff_size = 36 + data_size;

    std::string out;
    out.reserve(44 + (size_t)data_size);
    auto put_u32 = [&](uint32_t v) {
        out.push_back((char)(v & 0xff));
        out.push_back((char)((v >> 8) & 0xff));
        out.push_back((char)((v >> 16) & 0xff));
        out.push_back((char)((v >> 24) & 0xff));
    };
    auto put_u16 = [&](uint16_t v) {
        out.push_back((char)(v & 0xff));
        out.push_back((char)((v >> 8) & 0xff));
    };
    out.append("RIFF", 4);
    put_u32(riff_size);
    out.append("WAVE", 4);
    out.append("fmt ", 4);
    put_u32(16); // PCM fmt chunk size
    put_u16(1);  // PCM format
    put_u16(num_channels);
    put_u32((uint32_t)sample_rate);
    put_u32(byte_rate);
    put_u16(block_align);
    put_u16(bits_per_sample);
    out.append("data", 4);
    put_u32(data_size);
    if (n_samples <= 0)
        return out;
    out.resize(out.size() + (size_t)data_size);
    int16_t* dst = reinterpret_cast<int16_t*>(&out[out.size() - data_size]);
    for (int i = 0; i < n_samples; i++) {
        float s = pcm[i];
        if (s > 1.0f)
            s = 1.0f;
        if (s < -1.0f)
            s = -1.0f;
        dst[i] = (int16_t)std::lround(s * 32767.0f);
    }
    return out;
}
