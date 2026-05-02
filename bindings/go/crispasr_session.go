package whisper

// Minimal TTS surface for the Go binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, orpheus) plus the kokoro per-language model + voice
// resolver (PLAN #56 opt 2b).

/*
#cgo LDFLAGS: -lcrispasr
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
