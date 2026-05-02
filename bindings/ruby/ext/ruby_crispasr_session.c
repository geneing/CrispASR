// Minimal TTS surface for the Ruby binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, orpheus) plus the kokoro per-language model + voice
// resolver (PLAN #56 opt 2b).
//
// Surface (under module `CrispASR::Session`):
//   open(model_path, n_threads) -> handle
//   close(handle)
//   set_codec_path(handle, path)
//   set_voice(handle, path, ref_text=nil)
//   set_speaker_name(handle, name)             # orpheus + qwen3-tts CV
//   speakers(handle) -> Array<String>
//   set_instruct(handle, instruct)             # qwen3-tts VoiceDesign
//   is_custom_voice(handle) -> Boolean         # qwen3-tts variant detect
//   is_voice_design(handle) -> Boolean         # qwen3-tts variant detect
//   synthesize(handle, text) -> Array<Float>   # 24 kHz mono PCM
//
// And a singleton method:
//   CrispASR::Session.kokoro_resolve_for_lang(model_path, lang)
//     -> { model_path:, voice_path:, voice_name:, backbone_swapped: }

#include <ruby.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Forward-declare the C ABI exported by libcrispasr. Full prototypes
// live in src/crispasr_c_api.cpp and src/kokoro.h.
struct CrispasrSession;
extern struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
extern void                    crispasr_session_close(struct CrispasrSession* s);
extern int                     crispasr_session_set_codec_path(struct CrispasrSession* s, const char* path);
extern int                     crispasr_session_set_voice(struct CrispasrSession* s, const char* path,
                                                          const char* ref_text_or_null);
extern int                     crispasr_session_set_speaker_name(struct CrispasrSession* s, const char* name);
extern int                     crispasr_session_n_speakers(struct CrispasrSession* s);
extern const char*             crispasr_session_get_speaker_name(struct CrispasrSession* s, int i);
extern int                     crispasr_session_set_instruct(struct CrispasrSession* s, const char* instruct);
extern int                     crispasr_session_is_custom_voice(struct CrispasrSession* s);
extern int                     crispasr_session_is_voice_design(struct CrispasrSession* s);
extern float*                  crispasr_session_synthesize(struct CrispasrSession* s, const char* text,
                                                           int* out_n_samples);
extern void                    crispasr_pcm_free(float* pcm);
extern int                     crispasr_session_kokoro_clear_phoneme_cache(struct CrispasrSession* s);
extern int                     crispasr_session_set_source_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_target_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_punctuation(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_translate(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_temperature(struct CrispasrSession* s, float temperature,
                                                                unsigned long long seed);
extern int                     crispasr_session_detect_language(struct CrispasrSession* s, const float* pcm,
                                                                int n_samples, const char* lid_model_path, int method,
                                                                char* out_lang, int out_lang_cap, float* out_prob);
extern int                     crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len);
extern int                     crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len,
                                                                          char* out_picked, int out_picked_len);

static VALUE mCrispASR;
static VALUE mSession;

static VALUE rb_session_open(VALUE self, VALUE model_path, VALUE n_threads) {
    struct CrispasrSession* s =
        crispasr_session_open(StringValueCStr(model_path), NUM2INT(n_threads));
    if (!s) rb_raise(rb_eRuntimeError, "crispasr_session_open: failed to open %s",
                     StringValueCStr(model_path));
    return ULL2NUM((uintptr_t)s);
}

static VALUE rb_session_close(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    if (s) crispasr_session_close(s);
    return Qnil;
}

static VALUE rb_session_set_codec_path(VALUE self, VALUE handle, VALUE path) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_codec_path(s, StringValueCStr(path));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_codec_path failed (rc=%d)", rc);
    return Qnil;
}

// Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
static VALUE rb_session_clear_phoneme_cache(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_kokoro_clear_phoneme_cache(s);
    if (rc != 0) rb_raise(rb_eRuntimeError, "clear_phoneme_cache failed (rc=%d)", rc);
    return Qnil;
}

// ---- Sticky session-state setters (PLAN #59 partial unblock) ----

static VALUE rb_session_set_source_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_source_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_source_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_target_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_target_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_target_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_punctuation(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_punctuation(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_punctuation failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_translate(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_translate(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_translate failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_temperature(VALUE self, VALUE handle, VALUE temperature, VALUE seed) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_temperature(s, (float)NUM2DBL(temperature),
                                              (unsigned long long)NUM2ULL(seed));
    // rc == -2 = no backend supports it; soft no-op.
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_temperature failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_detect_language(VALUE self, VALUE handle, VALUE pcm_arr, VALUE lid_path, VALUE method) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    if (!pcm) rb_raise(rb_eNoMemError, "alloc failed");
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    char out_lang[16] = {0};
    float prob = 0.0f;
    int rc = crispasr_session_detect_language(s, pcm, (int)n, StringValueCStr(lid_path),
                                              NUM2INT(method), out_lang, sizeof(out_lang), &prob);
    free(pcm);
    if (rc != 0) rb_raise(rb_eRuntimeError, "detect_language failed (rc=%d)", rc);
    VALUE pair = rb_ary_new_capa(2);
    rb_ary_push(pair, rb_str_new_cstr(out_lang));
    rb_ary_push(pair, DBL2NUM((double)prob));
    return pair;
}

static VALUE rb_session_set_voice(int argc, VALUE* argv, VALUE self) {
    VALUE handle, path, ref_text;
    rb_scan_args(argc, argv, "21", &handle, &path, &ref_text);
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    const char* rt = NIL_P(ref_text) ? NULL : StringValueCStr(ref_text);
    int rc = crispasr_session_set_voice(s, StringValueCStr(path), rt);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_voice failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_speaker_name(VALUE self, VALUE handle, VALUE name) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_speaker_name(s, StringValueCStr(name));
    if (rc == -2) rb_raise(rb_eArgError, "unknown speaker: %s", StringValueCStr(name));
    if (rc == -3) rb_raise(rb_eRuntimeError, "backend has no preset speakers; use set_voice instead");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_speaker_name failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_speakers(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = crispasr_session_n_speakers(s);
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) {
        const char* name = crispasr_session_get_speaker_name(s, i);
        rb_ary_push(arr, name ? rb_str_new_cstr(name) : rb_str_new_cstr(""));
    }
    return arr;
}

static VALUE rb_session_set_instruct(VALUE self, VALUE handle, VALUE instruct) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_instruct(s, StringValueCStr(instruct));
    if (rc == -3) rb_raise(rb_eRuntimeError,
            "backend is not a VoiceDesign variant; set_instruct only applies to qwen3-tts VoiceDesign models");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_instruct failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_is_custom_voice(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_custom_voice(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_is_voice_design(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_voice_design(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_synthesize(VALUE self, VALUE handle, VALUE text) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = 0;
    float* pcm = crispasr_session_synthesize(s, StringValueCStr(text), &n);
    if (!pcm || n <= 0) {
        if (pcm) crispasr_pcm_free(pcm);
        rb_raise(rb_eRuntimeError, "synthesize returned no audio");
    }
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) rb_ary_push(arr, DBL2NUM((double)pcm[i]));
    crispasr_pcm_free(pcm);
    return arr;
}

static VALUE rb_kokoro_resolve_for_lang(VALUE self, VALUE model_path, VALUE lang) {
    char out_model[1024]  = {0};
    char out_voice[1024]  = {0};
    char out_picked[64]   = {0};

    const char* mp = StringValueCStr(model_path);
    const char* lg = NIL_P(lang) ? "" : StringValueCStr(lang);

    int rc = crispasr_kokoro_resolve_model_for_lang_abi(mp, lg, out_model, sizeof(out_model));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_model_for_lang: buffer too small");
    int swapped = (rc == 0);
    const char* resolved = (out_model[0] != 0) ? out_model : mp;

    rc = crispasr_kokoro_resolve_fallback_voice_abi(mp, lg,
                                                    out_voice, sizeof(out_voice),
                                                    out_picked, sizeof(out_picked));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_fallback_voice: buffer too small");

    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("model_path")),       rb_str_new_cstr(resolved));
    rb_hash_aset(h, ID2SYM(rb_intern("voice_path")),       rc == 0 ? rb_str_new_cstr(out_voice) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("voice_name")),       rc == 0 ? rb_str_new_cstr(out_picked) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("backbone_swapped")), swapped ? Qtrue : Qfalse);
    return h;
}

void init_ruby_crispasr_session(VALUE* mWhisper) {
    // Define module path under the existing Whisper module so existing
    // code keeps working: CrispASR::Session, but we also alias it under
    // Whisper::CrispASR::Session.
    mCrispASR = rb_define_module_under(*mWhisper, "CrispASR");
    mSession  = rb_define_module_under(mCrispASR, "Session");

    rb_define_singleton_method(mSession, "open",                 rb_session_open,             2);
    rb_define_singleton_method(mSession, "close",                rb_session_close,            1);
    rb_define_singleton_method(mSession, "set_codec_path",       rb_session_set_codec_path,   2);
    rb_define_singleton_method(mSession, "set_voice",            rb_session_set_voice,       -1);
    rb_define_singleton_method(mSession, "set_speaker_name",     rb_session_set_speaker_name, 2);
    rb_define_singleton_method(mSession, "speakers",             rb_session_speakers,         1);
    rb_define_singleton_method(mSession, "set_instruct",         rb_session_set_instruct,     2);
    rb_define_singleton_method(mSession, "is_custom_voice",      rb_session_is_custom_voice,  1);
    rb_define_singleton_method(mSession, "is_voice_design",      rb_session_is_voice_design,  1);
    rb_define_singleton_method(mSession, "synthesize",           rb_session_synthesize,       2);
    rb_define_singleton_method(mSession, "clear_phoneme_cache",  rb_session_clear_phoneme_cache, 1);
    rb_define_singleton_method(mSession, "set_source_language",  rb_session_set_source_language, 2);
    rb_define_singleton_method(mSession, "set_target_language",  rb_session_set_target_language, 2);
    rb_define_singleton_method(mSession, "set_punctuation",      rb_session_set_punctuation, 2);
    rb_define_singleton_method(mSession, "set_translate",        rb_session_set_translate, 2);
    rb_define_singleton_method(mSession, "set_temperature",      rb_session_set_temperature, 3);
    rb_define_singleton_method(mSession, "detect_language",      rb_session_detect_language, 4);
    rb_define_singleton_method(mSession, "kokoro_resolve_for_lang", rb_kokoro_resolve_for_lang, 2);
}
