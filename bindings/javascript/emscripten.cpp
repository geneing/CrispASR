//
// This is the Javascript API of crispasr
//
// Very crude at the moment.
// Feel free to contribute and make this better!
//
// See tests/test-crispasr.js for sample usage.
//

#include "crispasr.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <thread>
#include <vector>

// Forward-declare the unified Session + kokoro routing C ABI exported
// by libcrispasr. The full prototypes live in src/crispasr_c_api.cpp
// and src/kokoro.h. Including those headers from emscripten would pull
// in C++/STL declarations we don't need here.
extern "C" {
struct CrispasrSession;
struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void                    crispasr_session_close(struct CrispasrSession* s);
int                     crispasr_session_set_codec_path(struct CrispasrSession* s, const char* path);
int                     crispasr_session_set_voice(struct CrispasrSession* s, const char* path,
                                                   const char* ref_text_or_null);
int                     crispasr_session_set_speaker_name(struct CrispasrSession* s, const char* name);
int                     crispasr_session_n_speakers(struct CrispasrSession* s);
const char*             crispasr_session_get_speaker_name(struct CrispasrSession* s, int i);
int                     crispasr_session_set_instruct(struct CrispasrSession* s, const char* instruct);
int                     crispasr_session_is_custom_voice(struct CrispasrSession* s);
int                     crispasr_session_is_voice_design(struct CrispasrSession* s);
float*                  crispasr_session_synthesize(struct CrispasrSession* s, const char* text,
                                                    int* out_n_samples);
void                    crispasr_pcm_free(float* pcm);
int                     crispasr_session_kokoro_clear_phoneme_cache(struct CrispasrSession* s);
int                     crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                   char* out_path, int out_path_len);
int                     crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                   char* out_path, int out_path_len,
                                                                   char* out_picked, int out_picked_len);
}

static struct CrispasrSession* g_tts_session = nullptr;

struct whisper_context* g_context;

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("init", emscripten::optional_override([](const std::string& path_model) {
                             if (g_context == nullptr) {
                                 g_context = whisper_init_from_file_with_params(path_model.c_str(),
                                                                                whisper_context_default_params());
                                 if (g_context != nullptr) {
                                     return true;
                                 } else {
                                     return false;
                                 }
                             }

                             return false;
                         }));

    emscripten::function("free", emscripten::optional_override([]() {
                             if (g_context) {
                                 whisper_free(g_context);
                                 g_context = nullptr;
                             }
                         }));

    emscripten::function(
        "full_default",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang, bool translate) {
            if (g_context == nullptr) {
                return -1;
            }

            struct whisper_full_params params =
                whisper_full_default_params(whisper_sampling_strategy::CRISPASR_SAMPLING_GREEDY);

            params.print_realtime = true;
            params.print_progress = false;
            params.print_timestamps = true;
            params.print_special = false;
            params.translate = translate;
            params.language = whisper_is_multilingual(g_context) ? lang.c_str() : "en";
            params.n_threads = std::min(8, (int)std::thread::hardware_concurrency());
            params.offset_ms = 0;

            std::vector<float> pcmf32;
            const int n = audio["length"].as<int>();

            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];

            pcmf32.resize(n);

            emscripten::val memoryView =
                audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            memoryView.call<void>("set", audio);

            // print system information
            {
                printf("\n");
                printf("system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(),
                       whisper_print_system_info());

                printf("\n");
                printf("%s: processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s ...\n",
                       __func__, int(pcmf32.size()), float(pcmf32.size()) / CRISPASR_SAMPLE_RATE, params.n_threads, 1,
                       params.language, params.translate ? "translate" : "transcribe");

                printf("\n");
            }

            // run whisper
            {
                whisper_reset_timings(g_context);
                whisper_full(g_context, params, pcmf32.data(), pcmf32.size());
                whisper_print_timings(g_context);
            }

            return 0;
        }));

    // -------------------------------------------------------------------
    // TTS surface (kokoro / vibevoice / qwen3-tts) + kokoro per-language
    // routing (PLAN #56 opt 2b).
    // -------------------------------------------------------------------

    emscripten::function("ttsOpen", emscripten::optional_override([](const std::string& model_path,
                                                                     int n_threads) {
                             if (g_tts_session != nullptr) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                             g_tts_session = crispasr_session_open(model_path.c_str(),
                                                                   n_threads <= 0 ? 1 : n_threads);
                             return g_tts_session != nullptr;
                         }));

    emscripten::function("ttsClose", emscripten::optional_override([]() {
                             if (g_tts_session) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                         }));

    emscripten::function("ttsSetCodecPath", emscripten::optional_override([](const std::string& path) {
                             return g_tts_session ? crispasr_session_set_codec_path(g_tts_session,
                                                                                    path.c_str())
                                                  : -1;
                         }));

    // Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
    emscripten::function("ttsClearPhonemeCache", emscripten::optional_override([]() {
                             return g_tts_session ? crispasr_session_kokoro_clear_phoneme_cache(g_tts_session)
                                                  : -1;
                         }));

    emscripten::function("ttsSetVoice", emscripten::optional_override([](const std::string& path,
                                                                         const std::string& ref_text) {
                             if (!g_tts_session) return -1;
                             const char* rt = ref_text.empty() ? nullptr : ref_text.c_str();
                             return crispasr_session_set_voice(g_tts_session, path.c_str(), rt);
                         }));

    // Orpheus preset speakers — set by NAME, not by file path.
    emscripten::function("ttsSetSpeakerName",
                         emscripten::optional_override([](const std::string& name) {
                             if (!g_tts_session) return -1;
                             return crispasr_session_set_speaker_name(g_tts_session, name.c_str());
                         }));

    // Returns the list of preset speaker names for the active backend
    // (orpheus today). Empty array if the backend has no preset speakers.
    emscripten::function("ttsSpeakers",
                         emscripten::optional_override([]() -> emscripten::val {
                             emscripten::val out = emscripten::val::array();
                             if (!g_tts_session) return out;
                             int n = crispasr_session_n_speakers(g_tts_session);
                             for (int i = 0; i < n; i++) {
                                 const char* name = crispasr_session_get_speaker_name(g_tts_session, i);
                                 if (name) out.call<void>("push", std::string(name));
                             }
                             return out;
                         }));

    // qwen3-tts VoiceDesign — natural-language voice description.
    emscripten::function("ttsSetInstruct",
                         emscripten::optional_override([](const std::string& instruct) {
                             if (!g_tts_session) return -1;
                             return crispasr_session_set_instruct(g_tts_session, instruct.c_str());
                         }));

    // qwen3-tts variant detection (returns false also when the active
    // backend isn't qwen3-tts).
    emscripten::function("ttsIsCustomVoice",
                         emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_custom_voice(g_tts_session) != 0;
                         }));

    emscripten::function("ttsIsVoiceDesign",
                         emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_voice_design(g_tts_session) != 0;
                         }));

    // Returns a Float32Array of 24 kHz mono PCM. Empty array on failure.
    emscripten::function("ttsSynthesize",
                         emscripten::optional_override([](const std::string& text) -> emscripten::val {
                             if (!g_tts_session) return emscripten::val::array();
                             int n = 0;
                             float* pcm = crispasr_session_synthesize(g_tts_session, text.c_str(), &n);
                             if (!pcm || n <= 0) {
                                 if (pcm) crispasr_pcm_free(pcm);
                                 return emscripten::val::array();
                             }
                             emscripten::val out = emscripten::val::global("Float32Array").new_(n);
                             emscripten::val memoryView = emscripten::val(emscripten::typed_memory_view(n, pcm));
                             out.call<void>("set", memoryView);
                             crispasr_pcm_free(pcm);
                             return out;
                         }));

    // Mirrors python crispasr.kokoro_resolve_for_lang() — returns
    // {modelPath, voicePath, voiceName, backboneSwapped}.
    emscripten::function("kokoroResolveForLang",
                         emscripten::optional_override([](const std::string& model_path,
                                                          const std::string& lang) -> emscripten::val {
                             char out_model[1024]  = {0};
                             char out_voice[1024]  = {0};
                             char out_picked[64]   = {0};

                             int rc = crispasr_kokoro_resolve_model_for_lang_abi(
                                 model_path.c_str(), lang.c_str(), out_model, sizeof(out_model));
                             bool swapped = (rc == 0);
                             std::string resolved = (out_model[0] != 0) ? std::string(out_model) : model_path;

                             std::string vp, vn;
                             rc = crispasr_kokoro_resolve_fallback_voice_abi(
                                 model_path.c_str(), lang.c_str(),
                                 out_voice, sizeof(out_voice),
                                 out_picked, sizeof(out_picked));
                             if (rc == 0) {
                                 vp = out_voice;
                                 vn = out_picked;
                             }

                             emscripten::val r = emscripten::val::object();
                             r.set("modelPath",       resolved);
                             r.set("voicePath",       vp);
                             r.set("voiceName",       vn);
                             r.set("backboneSwapped", swapped);
                             return r;
                         }));
}
