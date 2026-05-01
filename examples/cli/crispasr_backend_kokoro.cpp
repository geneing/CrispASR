// crispasr_backend_kokoro.cpp — adapter for hexgrad/Kokoro-82M and
// yl4579/StyleTTS2-LJSpeech (iSTFTNet TTS).
//
// Two-GGUF runtime: the talker GGUF (loaded from --model) and a
// per-voice GGUF (loaded via --voice). The talker carries 5 components
// (text_enc, BERT, predictor, decoder, generator); each voice GGUF
// stores one (style_pred, style_dec) reference vector indexed by
// phoneme length. Phonemizer: espeak-ng shell-out, with --tts-phonemes
// to bypass it.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "kokoro.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class KokoroBackend : public CrispasrBackend {
public:
    KokoroBackend() = default;
    ~KokoroBackend() override { KokoroBackend::shutdown(); }

    const char* name() const override { return "kokoro"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[kokoro]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        kokoro_context_params cp = kokoro_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // Map -l/--language to the espeak-ng voice. "auto" keeps the default
        // (en-us) since espeak has no auto-detect mode.
        if (!p.language.empty() && p.language != "auto") {
            std::strncpy(cp.espeak_lang, p.language.c_str(), sizeof(cp.espeak_lang) - 1);
            cp.espeak_lang[sizeof(cp.espeak_lang) - 1] = '\0';
        }
        ctx_ = kokoro_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[kokoro]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        // Voice pack: load once on first call. Required — without it, synthesis
        // returns nullptr (predictor needs a (style_pred, style_dec) reference).
        if (!voice_loaded_ && !params.tts_voice.empty()) {
            if (kokoro_load_voice_pack(ctx_, params.tts_voice.c_str()) != 0) {
                fprintf(stderr, "crispasr[kokoro]: failed to load voice pack '%s'\n", params.tts_voice.c_str());
                return {};
            }
            voice_loaded_ = true;
        }

        int n = 0;
        float* pcm = kokoro_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        kokoro_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            kokoro_free(ctx_);
            ctx_ = nullptr;
        }
        voice_loaded_ = false;
    }

private:
    kokoro_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_kokoro_backend() {
    return std::unique_ptr<CrispasrBackend>(new KokoroBackend());
}
