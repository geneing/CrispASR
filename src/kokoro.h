#pragma once

// Kokoro / StyleTTS2 (iSTFTNet) public C ABI.
//
// hexgrad/Kokoro-82M and yl4579/StyleTTS2-LJSpeech share the same
// architecture (custom ALBERT BERT + ProsodyPredictor + iSTFTNet
// decoder + phoneme TextEncoder) and feed through this single runtime.
// Pre-trained voices ship as separate per-voice GGUFs (arch =
// "kokoro-voice") containing one F32 tensor `voice.pack[max_phon, 1,
// 256]`. Index by phoneme length L: ref_s = voice.pack[L-1, 0, :],
// split as [predictor_style 0:128 | decoder_style 128:256].
//
// Phonemizer: espeak-ng via popen("espeak-ng -q --ipa=3 -v LANG TEXT").
// Pass already-IPA strings via kokoro_synthesize_phonemes to skip the
// shell-out. The vocab map (178 IPA symbols) lives in the GGUF as
// `tokenizer.ggml.tokens`.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kokoro_context;

struct kokoro_context_params {
    int  n_threads;
    int  verbosity;       // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    bool gen_force_metal; // KOKORO_GEN_FORCE_METAL=1 — debug only; default false
                          // pins the iSTFTNet generator to backend_cpu to avoid
                          // a known Metal hang on stride-10 ConvTranspose1d
    char espeak_lang[32]; // espeak-ng -v LANG, default "en-us"
};

struct kokoro_context_params kokoro_context_default_params(void);

// Load Kokoro / StyleTTS2 GGUF. Returns nullptr on failure.
struct kokoro_context* kokoro_init_from_file(const char* path_model, struct kokoro_context_params params);

void kokoro_free(struct kokoro_context* ctx);
void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads);

// Load a voice-pack GGUF (arch="kokoro-voice"). Each pack stores ONE
// voice — single tensor `voice.pack[max_phon, 1, 256]`, plus
// `kokoro_voice.name` metadata. Returns 0 on success.
int kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path);

// Override the espeak-ng language (default "en-us"). Returns 0 on success.
int kokoro_set_language(struct kokoro_context* ctx, const char* espeak_lang);

// Tokenise a phoneme string (already-IPA) into the model's vocab.
// Returns malloc'd int32_t[*out_n] — caller frees with free().
int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n);

// Synthesise text → 24 kHz mono float32 PCM. Runs the espeak-ng
// phonemizer first. Returns malloc'd buffer; caller frees with
// kokoro_pcm_free. *out_n_samples set on success; nullptr on failure.
float* kokoro_synthesize(struct kokoro_context* ctx, const char* text, int* out_n_samples);

// Same as kokoro_synthesize but the input is already IPA — skips espeak-ng.
float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples);

// Diff-harness stage extractor. Pass a phoneme string and a stage name;
// returns the stage's activations as malloc'd float[*out_n]. Stage names
// match the ggml_set_name labels in src/kokoro.cpp:
//
// All "L"-shaped stages below use the StyleTTS2 pad-wrap convention —
// the raw phoneme ids are wrapped as [pad, *raw, pad] before being fed
// to BERT / text_enc / predictor, so each L is the raw token count + 2.
//
//   "token_ids"        — int32 cast to float (raw tokenisation, length L_raw)
//   "bert_pooler_out"  — (768, L) BERT last_hidden_state (NOT the pooled
//                        vector — name kept for ABI stability; "pooler"
//                        weight is loaded but unused by the synth path)
//   "bert_proj_out"    — (512, L) bert_proj per-token Linear of pooler_out
//   "text_enc_out"     — (512, L) text encoder output
//   "dur_enc_out"      — (L, 512) duration-encoder output
//   "durations"        — (L,) integer durations cast to float
//   "align_out"        — (T_frames, 512) duration-aligned features
//   "f0_curve"         — (T_frames,) F0 prediction
//   "n_curve"          — (T_frames,) energy prediction
//   "dec_encode_out"   — (T_frames, 512) decoder pre-generator features
//   "dec_decode_3_out" — (T_frames, 256) last decode-stack output
//   "gen_pre_post_out" — (T_audio_frames, 512) generator output before conv_post
//   "mag"              — (11, T_audio_frames) iSTFT magnitude
//   "phase"            — (11, T_audio_frames) iSTFT phase
//   "audio_out"        — (T_samples,) final 24 kHz waveform
//
// Caller frees with free().
float* kokoro_extract_stage(struct kokoro_context* ctx, const char* phonemes, const char* stage_name, int* out_n);

void kokoro_pcm_free(float* pcm);

#ifdef __cplusplus
}
#endif
