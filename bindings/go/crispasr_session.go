package whisper

// Minimal TTS surface for the Go binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, orpheus) plus the kokoro per-language model + voice
// resolver (PLAN #56 opt 2b).

/*
// LDFLAGS for libcrispasr + all conditionally-built sub-libs are set in
// whisper.go (the canonical cgo block). Don't re-list here to avoid
// `ld: warning: ignoring duplicate libraries: '-lcrispasr'`.
#include <stdlib.h>
#include <string.h>

// Forward-declare the C ABI symbols. The full signatures live in
// src/crispasr_c_api.cpp (Session) and src/kokoro.h
// (kokoro_resolve_*_abi). They're exported by libcrispasr.dylib /
// .so and linked in via cgo's -lcrispasr.

typedef struct CrispasrSession CrispasrSession;

CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void             crispasr_session_close(CrispasrSession* s);
int              crispasr_session_set_codec_path(CrispasrSession* s, const char* path);
int              crispasr_session_set_source_language(CrispasrSession* s, const char* lang);
int              crispasr_session_set_target_language(CrispasrSession* s, const char* lang);
int              crispasr_session_set_punctuation(CrispasrSession* s, int enable);
int              crispasr_session_set_translate(CrispasrSession* s, int enable);
int              crispasr_session_set_temperature(CrispasrSession* s, float temperature, unsigned long long seed);
int              crispasr_session_detect_language(CrispasrSession* s, const float* pcm, int n_samples,
                                                  const char* lid_model_path, int method,
                                                  char* out_lang, int out_lang_cap, float* out_prob);
int              crispasr_session_set_voice(CrispasrSession* s, const char* path, const char* ref_text_or_null);
int              crispasr_session_set_speaker_name(CrispasrSession* s, const char* name);
int              crispasr_session_n_speakers(CrispasrSession* s);
const char*      crispasr_session_get_speaker_name(CrispasrSession* s, int i);
int              crispasr_session_set_instruct(CrispasrSession* s, const char* instruct);
int              crispasr_session_is_custom_voice(CrispasrSession* s);
int              crispasr_session_is_voice_design(CrispasrSession* s);
float*           crispasr_session_synthesize(CrispasrSession* s, const char* text, int* out_n_samples);
void             crispasr_pcm_free(float* pcm);
int              crispasr_session_kokoro_clear_phoneme_cache(CrispasrSession* s);

int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                               char* out_path, int out_path_len);
int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                               char* out_path, int out_path_len,
                                               char* out_picked, int out_picked_len);

// --- Streaming (PLAN #62) ---
typedef struct CrispasrStream CrispasrStream;
CrispasrStream* crispasr_session_stream_open(CrispasrSession* s, int n_threads, int step_ms,
                                             int length_ms, int keep_ms, const char* language, int translate);
int             crispasr_stream_feed(CrispasrStream* s, const float* pcm, int n_samples);
int             crispasr_stream_get_text(CrispasrStream* s, char* out_text, int out_cap,
                                         double* out_t0_s, double* out_t1_s, long long* out_counter);
int             crispasr_stream_flush(CrispasrStream* s);
void            crispasr_stream_close(CrispasrStream* s);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// CrispasrSession is a TTS-capable session (kokoro, vibevoice, qwen3-tts, orpheus).
type CrispasrSession struct {
	handle *C.CrispasrSession
}

// SessionOpen opens a backend session for the given model file.
// Detects the backend automatically from the GGUF metadata.
// Returns an error if the model can't be loaded.
func SessionOpen(modelPath string, nThreads int) (*CrispasrSession, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	h := C.crispasr_session_open(cpath, C.int(nThreads))
	if h == nil {
		return nil, errors.New("crispasr_session_open: failed to open " + modelPath)
	}
	return &CrispasrSession{handle: h}, nil
}

// Close releases the session handle.
func (s *CrispasrSession) Close() {
	if s != nil && s.handle != nil {
		C.crispasr_session_close(s.handle)
		s.handle = nil
	}
}

// ClearPhonemeCache drops the kokoro per-session phoneme cache.
// No-op for non-kokoro backends. Useful for long-running daemons that
// resynthesize across many speakers and want bounded memory. (PLAN #56 #5)
func (s *CrispasrSession) ClearPhonemeCache() error {
	rc := C.crispasr_session_kokoro_clear_phoneme_cache(s.handle)
	if rc != 0 {
		return errors.New("crispasr_session_kokoro_clear_phoneme_cache failed")
	}
	return nil
}

// ---------------------------------------------------------------------------
// Sticky session-state setters (PLAN #59 partial unblock).
// ---------------------------------------------------------------------------

// SetSourceLanguage sets the sticky source-language hint (canary, cohere,
// voxtral, whisper). Empty string clears. Per-call language args still win.
func (s *CrispasrSession) SetSourceLanguage(lang string) error {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	rc := C.crispasr_session_set_source_language(s.handle, cl)
	if rc != 0 {
		return errors.New("crispasr_session_set_source_language failed")
	}
	return nil
}

// SetTargetLanguage sets the sticky target-language. When ≠ source on
// canary/cohere, the backend emits a translation. For whisper, pair with
// SetTranslate(true).
func (s *CrispasrSession) SetTargetLanguage(lang string) error {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	rc := C.crispasr_session_set_target_language(s.handle, cl)
	if rc != 0 {
		return errors.New("crispasr_session_set_target_language failed")
	}
	return nil
}

// SetPunctuation toggles punctuation + capitalisation in the output
// (canary/cohere natively; LLM backends via post-process strip). Default true.
func (s *CrispasrSession) SetPunctuation(enable bool) error {
	v := C.int(0)
	if enable {
		v = 1
	}
	rc := C.crispasr_session_set_punctuation(s.handle, v)
	if rc != 0 {
		return errors.New("crispasr_session_set_punctuation failed")
	}
	return nil
}

// SetTranslate enables whisper sticky --translate. For canary/cohere/voxtral
// the equivalent is SetTargetLanguage ≠ source.
func (s *CrispasrSession) SetTranslate(enable bool) error {
	v := C.int(0)
	if enable {
		v = 1
	}
	rc := C.crispasr_session_set_translate(s.handle, v)
	if rc != 0 {
		return errors.New("crispasr_session_set_translate failed")
	}
	return nil
}

// SetTemperature sets the decoder temperature on backends that support
// runtime control (canary, cohere, parakeet, moonshine). Other backends
// silently no-op. seed is the RNG seed; pass 0 for time-based.
func (s *CrispasrSession) SetTemperature(temperature float32, seed uint64) error {
	rc := C.crispasr_session_set_temperature(s.handle, C.float(temperature), C.ulonglong(seed))
	// rc == -2 means no backend in this session honours it — soft no-op.
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_temperature failed")
	}
	return nil
}

// ---------------------------------------------------------------------------
// Streaming (PLAN #62) — rolling-window decoder for whisper today.
// ---------------------------------------------------------------------------

// Stream is a streaming-decoder handle returned by Session.StreamOpen.
// Feed PCM, pull text. Whisper-only at the C-ABI level today.
type Stream struct {
	handle *C.CrispasrStream
}

// StreamingUpdate is one commit from a streaming session — the latest
// concatenated text + its absolute audio-time bounds. Counter increments
// per commit; same value = no new text.
type StreamingUpdate struct {
	Text    string
	T0      float64
	T1      float64
	Counter int64
}

// StreamOpen opens a rolling-window streaming decoder for this session.
// Currently whisper-only at the C-ABI level. stepMs (default 3000) is
// how often to commit a partial transcript; lengthMs (default 10000) is
// the rolling window; keepMs (default 200) is the trailing audio carried.
func (s *CrispasrSession) StreamOpen(stepMs, lengthMs, keepMs int, language string, translate bool) (*Stream, error) {
	clang := C.CString(language)
	defer C.free(unsafe.Pointer(clang))
	tr := C.int(0)
	if translate {
		tr = 1
	}
	h := C.crispasr_session_stream_open(s.handle, C.int(4), C.int(stepMs), C.int(lengthMs), C.int(keepMs), clang, tr)
	if h == nil {
		return nil, errors.New("crispasr_session_stream_open failed (whisper-only today)")
	}
	return &Stream{handle: h}, nil
}

// Feed pushes 16 kHz mono float32 PCM. Returns 0 if still buffering, 1 if
// a new partial transcript is ready (call GetText).
func (st *Stream) Feed(pcm []float32) (int, error) {
	if len(pcm) == 0 {
		return 0, nil
	}
	rc := C.crispasr_stream_feed(st.handle, (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)))
	if rc < 0 {
		return 0, errors.New("crispasr_stream_feed failed")
	}
	return int(rc), nil
}

// GetText returns the latest committed transcript + absolute audio-time bounds.
func (st *Stream) GetText() (StreamingUpdate, error) {
	var buf [8192]C.char
	var t0, t1 C.double
	var counter C.longlong
	rc := C.crispasr_stream_get_text(st.handle, &buf[0], C.int(len(buf)), &t0, &t1, &counter)
	if rc < 0 {
		return StreamingUpdate{}, errors.New("crispasr_stream_get_text failed")
	}
	return StreamingUpdate{
		Text:    C.GoString(&buf[0]),
		T0:      float64(t0),
		T1:      float64(t1),
		Counter: int64(counter),
	}, nil
}

// Flush finalises any remaining buffered audio.
func (st *Stream) Flush() error {
	if rc := C.crispasr_stream_flush(st.handle); rc < 0 {
		return errors.New("crispasr_stream_flush failed")
	}
	return nil
}

// Close releases the streaming handle.
func (st *Stream) Close() {
	if st != nil && st.handle != nil {
		C.crispasr_stream_close(st.handle)
		st.handle = nil
	}
}

// DetectLanguage auto-detects the spoken language on raw 16 kHz mono PCM.
// method: 0=Whisper, 1=Silero (default), 2=Firered, 3=Ecapa.
// Returns the ISO 639-1 code and the model's confidence in [0, 1].
func (s *CrispasrSession) DetectLanguage(pcm []float32, lidModelPath string, method int) (string, float32, error) {
	cpath := C.CString(lidModelPath)
	defer C.free(unsafe.Pointer(cpath))
	var outLang [16]C.char
	var outProb C.float
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	rc := C.crispasr_session_detect_language(s.handle, pcmPtr, C.int(len(pcm)), cpath, C.int(method),
		&outLang[0], C.int(len(outLang)), &outProb)
	if rc != 0 {
		return "", 0, errors.New("crispasr_session_detect_language failed")
	}
	return C.GoString(&outLang[0]), float32(outProb), nil
}

// SetCodecPath loads a separate codec GGUF.
// Required for qwen3-tts (12 Hz tokenizer) and orpheus (SNAC codec);
// no-op for other backends.
func (s *CrispasrSession) SetCodecPath(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	rc := C.crispasr_session_set_codec_path(s.handle, cpath)
	if rc != 0 {
		return errors.New("crispasr_session_set_codec_path failed")
	}
	return nil
}

// SetVoice loads a voice prompt: a baked GGUF voice pack OR a *.wav reference.
// `refText` is required for qwen3-tts when `path` is a WAV; pass an empty
// string otherwise.
//
// For orpheus voice selection is BY NAME — use SetSpeakerName instead.
func (s *CrispasrSession) SetVoice(path, refText string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var rtPtr *C.char
	if refText != "" {
		crt := C.CString(refText)
		defer C.free(unsafe.Pointer(crt))
		rtPtr = crt
	}
	rc := C.crispasr_session_set_voice(s.handle, cpath, rtPtr)
	if rc != 0 {
		return errors.New("crispasr_session_set_voice failed")
	}
	return nil
}

// SetSpeakerName selects a fixed/preset speaker by NAME for backends
// that bake speaker names into the GGUF (orpheus today). Names are
// e.g. "tara"/"leo" for the canopylabs English finetune; "Anton"/"Sophie"
// for the Kartoffel_Orpheus DE finetunes. Use Speakers() to enumerate.
func (s *CrispasrSession) SetSpeakerName(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rc := C.crispasr_session_set_speaker_name(s.handle, cname)
	switch rc {
	case 0:
		return nil
	case -2:
		return fmt.Errorf("unknown speaker %q; call Speakers() to enumerate", name)
	case -3:
		return errors.New("backend has no preset speakers; use SetVoice instead")
	default:
		return fmt.Errorf("crispasr_session_set_speaker_name failed (rc=%d)", int(rc))
	}
}

// SetInstruct sets the natural-language voice description for
// instruct-tuned TTS backends (qwen3-tts VoiceDesign today).
// Required before Synthesize when the loaded backend is VoiceDesign.
// Detect via IsVoiceDesign().
func (s *CrispasrSession) SetInstruct(instruct string) error {
	cins := C.CString(instruct)
	defer C.free(unsafe.Pointer(cins))
	rc := C.crispasr_session_set_instruct(s.handle, cins)
	switch rc {
	case 0:
		return nil
	case -3:
		return errors.New("backend is not a VoiceDesign variant; SetInstruct only applies to qwen3-tts VoiceDesign")
	default:
		return fmt.Errorf("crispasr_session_set_instruct failed (rc=%d)", int(rc))
	}
}

// IsCustomVoice reports whether the loaded model is a qwen3-tts
// CustomVoice variant (use SetSpeakerName for it).
func (s *CrispasrSession) IsCustomVoice() bool {
	return C.crispasr_session_is_custom_voice(s.handle) != 0
}

// IsVoiceDesign reports whether the loaded model is a qwen3-tts
// VoiceDesign variant (use SetInstruct for it).
func (s *CrispasrSession) IsVoiceDesign() bool {
	return C.crispasr_session_is_voice_design(s.handle) != 0
}

// Speakers returns the list of preset speaker names for the active
// backend. Empty if the backend has no preset-speaker contract.
func (s *CrispasrSession) Speakers() []string {
	n := int(C.crispasr_session_n_speakers(s.handle))
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		ptr := C.crispasr_session_get_speaker_name(s.handle, C.int(i))
		if ptr != nil {
			out = append(out, C.GoString(ptr))
		}
	}
	return out
}

// Synthesize converts `text` to 24 kHz mono PCM. Requires a TTS-capable
// backend (kokoro / vibevoice / qwen3-tts / orpheus).
func (s *CrispasrSession) Synthesize(text string) ([]float32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	var n C.int
	ptr := C.crispasr_session_synthesize(s.handle, ctext, &n)
	if ptr == nil || n <= 0 {
		return nil, errors.New("crispasr_session_synthesize: no audio produced")
	}
	defer C.crispasr_pcm_free(ptr)
	samples := make([]float32, int(n))
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(n))
	copy(samples, src)
	return samples, nil
}

// KokoroResolved is the result of KokoroResolveForLang. Mirrors the
// Python wrapper's KokoroResolved dataclass and the Rust crate's
// kokoro_resolve_for_lang() return type.
type KokoroResolved struct {
	ModelPath       string // path to actually load (may differ from input)
	VoicePath       string // fallback voice path; empty if not applicable
	VoiceName       string // basename of the picked voice (e.g. "df_victoria")
	BackboneSwapped bool   // true iff the model path was rewritten
}

// KokoroResolveForLang returns the kokoro model + fallback voice that
// CrispASR's CLI would pick for `lang`. Mirrors PLAN #56 opt 2b. Wrappers
// should call this *before* SessionOpen so the routing kicks in even
// outside the CLI entry point.
func KokoroResolveForLang(modelPath, lang string) (KokoroResolved, error) {
	cmodel := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cmodel))
	clang := C.CString(lang)
	defer C.free(unsafe.Pointer(clang))

	outModel := (*C.char)(C.malloc(1024))
	defer C.free(unsafe.Pointer(outModel))
	outVoice := (*C.char)(C.malloc(1024))
	defer C.free(unsafe.Pointer(outVoice))
	outPicked := (*C.char)(C.malloc(64))
	defer C.free(unsafe.Pointer(outPicked))

	swapped := false
	rc := C.crispasr_kokoro_resolve_model_for_lang_abi(cmodel, clang, outModel, 1024)
	if rc < 0 {
		return KokoroResolved{}, errors.New("kokoro_resolve_model_for_lang: buffer too small")
	}
	if rc == 0 {
		swapped = true
	}
	resolvedModel := C.GoString(outModel)
	if resolvedModel == "" {
		resolvedModel = modelPath
	}

	rc = C.crispasr_kokoro_resolve_fallback_voice_abi(cmodel, clang, outVoice, 1024, outPicked, 64)
	if rc < 0 {
		return KokoroResolved{}, errors.New("kokoro_resolve_fallback_voice: buffer too small")
	}
	out := KokoroResolved{ModelPath: resolvedModel, BackboneSwapped: swapped}
	if rc == 0 {
		out.VoicePath = C.GoString(outVoice)
		out.VoiceName = C.GoString(outPicked)
	}
	return out, nil
}
