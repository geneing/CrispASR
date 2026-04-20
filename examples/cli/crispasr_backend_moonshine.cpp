// crispasr_backend_moonshine.cpp — Moonshine ASR backend adapter.

#include "crispasr_backend.h"
#include "moonshine.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MoonshineBackend : public CrispasrBackend {
public:
    MoonshineBackend() = default;

    const char* name() const override { return "moonshine"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD; }

    bool init(const whisper_params& params) override {
        struct moonshine_init_params mp = {};
        mp.model_path = params.model.c_str();
        mp.tokenizer_path = nullptr; // auto-detect
        mp.n_threads = params.n_threads;
        ctx_ = moonshine_init_with_params(mp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        const char* text = moonshine_transcribe(ctx_, samples, n_samples);
        if (!text || !text[0])
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            moonshine_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~MoonshineBackend() override { MoonshineBackend::shutdown(); }

private:
    moonshine_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend() {
    return std::make_unique<MoonshineBackend>();
}
