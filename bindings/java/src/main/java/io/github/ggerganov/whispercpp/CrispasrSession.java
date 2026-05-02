package io.github.ggerganov.whispercpp;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;

/**
 * Minimal TTS surface for the Java binding. Exposes the unified
 * CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
 * qwen3-tts, orpheus) plus the kokoro per-language model + voice
 * resolver (PLAN #56 opt 2b).
 *
 * <p>Usage:
 * <pre>{@code
 * CrispasrSession.Resolved r = CrispasrSession.kokoroResolveForLang(
 *     "/models/kokoro-82m-q8_0.gguf", "de");
 * try (CrispasrSession s = CrispasrSession.open(r.modelPath, 4)) {
 *     if (r.voicePath != null) s.setVoice(r.voicePath, null);
 *     float[] pcm = s.synthesize("Guten Tag");
 *     // ... write WAV ...
 * }
 * }</pre>
 */
public final class CrispasrSession implements AutoCloseable {

    public interface Lib extends Library {
        Lib INSTANCE = Native.load("crispasr", Lib.class);

        Pointer crispasr_session_open(String modelPath, int nThreads);
        void    crispasr_session_close(Pointer session);
        int     crispasr_session_set_codec_path(Pointer session, String path);
        int     crispasr_session_set_voice(Pointer session, String path, String refTextOrNull);
        int     crispasr_session_set_speaker_name(Pointer session, String name);
        int     crispasr_session_n_speakers(Pointer session);
        String  crispasr_session_get_speaker_name(Pointer session, int i);
        int     crispasr_session_set_instruct(Pointer session, String instruct);
        int     crispasr_session_is_custom_voice(Pointer session);
        int     crispasr_session_is_voice_design(Pointer session);
        Pointer crispasr_session_synthesize(Pointer session, String text, IntByReference outNSamples);
        void    crispasr_pcm_free(Pointer pcm);
        int     crispasr_session_kokoro_clear_phoneme_cache(Pointer session);

        int crispasr_kokoro_resolve_model_for_lang_abi(
                String modelPath, String lang, byte[] outPath, int outPathLen);
        int crispasr_kokoro_resolve_fallback_voice_abi(
                String modelPath, String lang,
                byte[] outPath, int outPathLen,
                byte[] outPicked, int outPickedLen);
    }

    private Pointer handle;

    private CrispasrSession(Pointer handle) {
        this.handle = handle;
    }

    /**
     * Open a backend session for the given model file. The backend is
     * detected automatically from the GGUF metadata.
     *
     * @throws IllegalStateException if the model can't be loaded.
     */
    public static CrispasrSession open(String modelPath, int nThreads) {
        Pointer p = Lib.INSTANCE.crispasr_session_open(modelPath, nThreads);
        if (p == null) {
            throw new IllegalStateException("crispasr_session_open: failed to open " + modelPath);
        }
        return new CrispasrSession(p);
    }

    /**
     * Drop the kokoro per-session phoneme cache. No-op for non-kokoro
     * backends. Useful for long-running daemons that resynthesize across
     * many speakers and want bounded memory. (PLAN #56 #5)
     */
    public void clearPhonemeCache() {
        int rc = Lib.INSTANCE.crispasr_session_kokoro_clear_phoneme_cache(handle);
        if (rc != 0) throw new IllegalStateException("clear_phoneme_cache failed (rc=" + rc + ")");
    }

    /**
     * Load a separate codec GGUF. Required for qwen3-tts (12 Hz tokenizer)
     * and orpheus (SNAC codec); no-op for other backends.
     */
    public void setCodecPath(String path) {
        int rc = Lib.INSTANCE.crispasr_session_set_codec_path(handle, path);
        if (rc != 0) throw new IllegalStateException("set_codec_path failed (rc=" + rc + ")");
    }

    /**
     * Load a voice prompt: a baked GGUF voice pack OR a *.wav reference.
     * {@code refText} is required for qwen3-tts when {@code path} is a WAV;
     * pass {@code null} otherwise.
     *
     * <p>For orpheus voice selection is BY NAME — use
     * {@link #setSpeakerName(String)} instead.
     */
    public void setVoice(String path, String refText) {
        int rc = Lib.INSTANCE.crispasr_session_set_voice(handle, path, refText);
        if (rc != 0) throw new IllegalStateException("set_voice failed (rc=" + rc + ")");
    }

    /**
     * Select a fixed/preset speaker by name (orpheus). Names are e.g.
     * {@code "tara"}, {@code "leo"}, {@code "leah"} for canopylabs;
     * {@code "Anton"}, {@code "Sophie"} for the Kartoffel_Orpheus DE
     * finetunes. Use {@link #speakers()} to enumerate.
     *
     * @throws IllegalArgumentException if {@code name} is not in the GGUF metadata
     * @throws IllegalStateException if the active backend has no preset-speaker contract
     */
    public void setSpeakerName(String name) {
        int rc = Lib.INSTANCE.crispasr_session_set_speaker_name(handle, name);
        if (rc == -2) throw new IllegalArgumentException("unknown speaker: " + name + "; call speakers() to enumerate");
        if (rc == -3) throw new IllegalStateException("backend has no preset speakers; use setVoice() instead");
        if (rc != 0) throw new IllegalStateException("set_speaker_name failed (rc=" + rc + ")");
    }

    /**
     * Return the list of preset speaker names for the active backend.
     * Empty if the backend has no preset-speaker contract.
     */
    public String[] speakers() {
        int n = Lib.INSTANCE.crispasr_session_n_speakers(handle);
        String[] out = new String[n];
        for (int i = 0; i < n; i++) {
            String s = Lib.INSTANCE.crispasr_session_get_speaker_name(handle, i);
            out[i] = (s == null) ? "" : s;
        }
        return out;
    }

    /**
     * Set the natural-language voice description for instruct-tuned TTS
     * backends (qwen3-tts VoiceDesign today). Required before
     * {@link #synthesize(String)} when the loaded backend is VoiceDesign.
     * Detect via {@link #isVoiceDesign()}.
     *
     * @throws IllegalStateException if the active backend isn't a VoiceDesign variant
     */
    public void setInstruct(String instruct) {
        int rc = Lib.INSTANCE.crispasr_session_set_instruct(handle, instruct);
        if (rc == -3) throw new IllegalStateException(
                "backend is not a VoiceDesign variant; setInstruct only applies to qwen3-tts VoiceDesign models");
        if (rc != 0) throw new IllegalStateException("set_instruct failed (rc=" + rc + ")");
    }

    /**
     * Whether the loaded model is a qwen3-tts CustomVoice variant
     * (use {@link #setSpeakerName(String)} for it).
     */
    public boolean isCustomVoice() {
        return Lib.INSTANCE.crispasr_session_is_custom_voice(handle) != 0;
    }

    /**
     * Whether the loaded model is a qwen3-tts VoiceDesign variant
     * (use {@link #setInstruct(String)} for it).
     */
    public boolean isVoiceDesign() {
        return Lib.INSTANCE.crispasr_session_is_voice_design(handle) != 0;
    }

    /**
     * Synthesise {@code text} to 24 kHz mono PCM. Requires a TTS-capable
     * backend (kokoro / vibevoice / qwen3-tts / orpheus).
     */
    public float[] synthesize(String text) {
        IntByReference n = new IntByReference(0);
        Pointer pcm = Lib.INSTANCE.crispasr_session_synthesize(handle, text, n);
        if (pcm == null || n.getValue() <= 0) {
            throw new IllegalStateException("synthesize returned no audio");
        }
        try {
            return pcm.getFloatArray(0, n.getValue());
        } finally {
            Lib.INSTANCE.crispasr_pcm_free(pcm);
        }
    }

    @Override
    public void close() {
        if (handle != null) {
            Lib.INSTANCE.crispasr_session_close(handle);
            handle = null;
        }
    }

    // -----------------------------------------------------------------
    // Kokoro per-language routing (PLAN #56 opt 2b)
    // -----------------------------------------------------------------

    /** Result of {@link #kokoroResolveForLang(String, String)}. */
    public static final class Resolved {
        /** Path to actually load (may differ from input). */
        public final String modelPath;
        /** Fallback voice path; {@code null} if not applicable. */
        public final String voicePath;
        /** Basename of the picked voice (e.g. "df_victoria"); {@code null} otherwise. */
        public final String voiceName;
        /** True iff the model path was rewritten to the German backbone. */
        public final boolean backboneSwapped;

        Resolved(String modelPath, String voicePath, String voiceName, boolean backboneSwapped) {
            this.modelPath = modelPath;
            this.voicePath = voicePath;
            this.voiceName = voiceName;
            this.backboneSwapped = backboneSwapped;
        }
    }

    /**
     * Resolve the kokoro model + fallback voice for {@code lang}. Mirrors
     * what the CrispASR CLI does for {@code --backend kokoro -l <lang>}
     * (PLAN #56 opt 2b). Wrappers should call this <em>before</em>
     * {@link #open(String, int)} so the routing kicks in even outside
     * the CLI entry point.
     */
    public static Resolved kokoroResolveForLang(String modelPath, String lang) {
        byte[] outModel = new byte[1024];
        byte[] outVoice = new byte[1024];
        byte[] outPicked = new byte[64];

        int rc = Lib.INSTANCE.crispasr_kokoro_resolve_model_for_lang_abi(
                modelPath, lang == null ? "" : lang, outModel, outModel.length);
        if (rc < 0) throw new IllegalStateException("kokoro_resolve_model_for_lang: buffer too small");
        boolean swapped = (rc == 0);
        String resolvedModel = nullTerminated(outModel);
        if (resolvedModel.isEmpty()) resolvedModel = modelPath;

        rc = Lib.INSTANCE.crispasr_kokoro_resolve_fallback_voice_abi(
                modelPath, lang == null ? "" : lang,
                outVoice, outVoice.length, outPicked, outPicked.length);
        if (rc < 0) throw new IllegalStateException("kokoro_resolve_fallback_voice: buffer too small");
        if (rc == 0) {
            return new Resolved(resolvedModel, nullTerminated(outVoice), nullTerminated(outPicked), swapped);
        }
        return new Resolved(resolvedModel, null, null, swapped);
    }

    private static String nullTerminated(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, java.nio.charset.StandardCharsets.UTF_8);
    }
}
