# Architecture

CrispASR is structured around three layers on top of whisper.cpp.
The split between `src/` (library) and `examples/cli/` (presentation)
is deliberate: **every algorithm** — VAD, diarization, LID, CTC
alignment, HF download/cache, model registry — lives in `src/` behind
a stable C-ABI (`src/crispasr_c_api.cpp`), and every consumer (CLI,
Dart, Python, Rust, Go, Java, Ruby) reaches it through the same
symbols. The CLI keeps only presentation + UX policy.

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses CLI args, dispatches to backend when --backend           │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/crispasr_*_cli.{h,cpp}                               │
│   Thin CLI shims for policy only — auto-download, TTY prompts,    │
│   sherpa-ONNX subprocess fallbacks. Delegate the algorithmic      │
│   work to the shared library below.                                │
├───────────────────────────────────────────────────────────────────┤
│ src/crispasr_c_api.cpp — C-ABI (shared with Dart / Python / Rust) │
│   crispasr_vad.{h,cpp}           Silero VAD + whisper-style       │
│                                  stitching, timestamp remap       │
│   crispasr_diarize.{h,cpp}       energy / xcorr / vad-turns /     │
│                                  native pyannote diarization      │
│   crispasr_lid.{h,cpp}           whisper-tiny + silero-native LID │
│   crispasr_aligner.{h,cpp}       canary-CTC + qwen3-forced-aligner│
│   crispasr_cache.{h,cpp}         HF download + ~/.cache/crispasr  │
│   crispasr_model_registry.{h,cpp} backend → canonical GGUF URL    │
├───────────────────────────────────────────────────────────────────┤
│ src/{whisper,parakeet,canary,canary_ctc,cohere,qwen3_asr,         │
│      voxtral,voxtral4b,granite_speech,silero_lid,pyannote_seg}.cpp│
│   Per-model runtimes (public C APIs)                              │
├───────────────────────────────────────────────────────────────────┤
│ src/core/      — shared model primitives (crispasr-core)          │
│   mel.{h,cpp}          log-mel spectrogram (NeMo + HF clusters)   │
│   ffn.h                SwiGLU + SiLU FFN helpers                  │
│   attention.h          Llama-style self-attention + flash-attn    │
│   gguf_loader.{h,cpp}  Unified GGUF open / weight mmap / lookup   │
├───────────────────────────────────────────────────────────────────┤
│ ggml                                                               │
└───────────────────────────────────────────────────────────────────┘
```

## `src/` — shared library surface

Every algorithm listed below is exposed as `extern "C"` functions
with a `crispasr_` prefix. The CLI, Python, Rust, and Dart bindings
all consume the same symbols.

| File | Role |
|---|---|
| `crispasr_c_api.cpp` | The C-ABI. Exports session open/close/transcribe, VAD, diarize, LID, alignment, cache, registry — everything a wrapper needs. |
| `crispasr_vad.{h,cpp}` | Silero VAD slicing + whisper-style stitching with timestamp remapping. Used by `crispasr_session_transcribe_vad`. |
| `crispasr_diarize.{h,cpp}` | Four diarizers: energy (stereo), xcorr (stereo, TDOA), vad-turns (mono, timing), pyannote (mono, GGUF). |
| `crispasr_lid.{h,cpp}` | whisper-tiny + silero-native language ID with process-wide whisper-context cache. |
| `crispasr_aligner.{h,cpp}` | canary-CTC + Qwen3-ForcedAligner forced alignment behind one entry point; filename-based dispatch. |
| `crispasr_cache.{h,cpp}` | WinHTTP / curl / wget download into `~/.cache/crispasr/`; zombie-file detection. |
| `crispasr_model_registry.{h,cpp}` | Backend → canonical GGUF URL table; fuzzy filename lookup for "did you mean …?" hints. |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended). |

## `examples/cli/` — presentation + policy

| File | Role |
|---|---|
| `cli.cpp` | crispasr entry point, extended with `--backend` dispatch branch. |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect. |
| `crispasr_backend_{parakeet,canary,cohere,granite,granite_nle,voxtral,voxtral4b,qwen3,fastconformer_ctc,wav2vec2,glm_asr,kyutai_stt,firered_asr,moonshine,moonshine_streaming,omniasr,gemma4_e2b,mimo_asr,vibevoice,qwen3_tts,orpheus,kokoro,chatterbox,m2m100,t5}.cpp` | Per-backend thin wrapper over each model's C API. ASR backends emit `crispasr_segment`s; TTS backends (`vibevoice`, `qwen3_tts`, `orpheus`, `kokoro`, `chatterbox`) implement `synthesize(text)` instead and write 24 kHz mono WAV via `--tts-output`; the translation backends (`m2m100` for facebook m2m100 + WMT21, `t5` for MADLAD-400 / future T5 translation) implement `translate_text(text, src, tgt)` and write UTF-8 to stdout. |
| `crispasr_output.{h,cpp}` | TXT / SRT / VTT / CSV / JSON / LRC writers on `crispasr_segment`. |
| `crispasr_vad_cli.{h,cpp}` | Delegates to `src/crispasr_vad`; adds auto-download for the Silero GGUF. |
| `crispasr_lid_cli.{h,cpp}` | Delegates to `src/crispasr_lid`; adds auto-download + sherpa-ONNX subprocess fallback. |
| `crispasr_diarize_cli.{h,cpp}` | Delegates to `src/crispasr_diarize`; adds sherpa subprocess fallback + pyannote GGUF auto-download. |
| `crispasr_model_mgr_cli.{h,cpp}` | Delegates to `src/crispasr_model_registry`; adds "Download now? [Y/n]" prompt on TTY. |
| `crispasr_aligner_cli.{h,cpp}` | Adapter converting `CrispasrAlignedWord` → the CLI's `crispasr_word` shape. |
| `crispasr_server.cpp` | HTTP server for the persistent-model mode + OpenAI-compatible endpoints. |
| `crispasr_llm_pipeline.h` | Templated audio-LLM pipeline (mel → encoder → prompt → KV decode). |
| `crispasr_run.cpp` | Top-level pipeline dispatch: resolve → detect → load → slice → transcribe → write. |

## `src/core/` — the shared model primitives

Duplicated scaffolding is bundled in a single static library,
`crispasr-core`, linked into every non-whisper model target.

| Header | Replaces | Consumers |
|---|---|---|
| `core/mel.{h,cpp}` | 7× copy-pasted STFT + mel filterbank + log + norm | parakeet, canary, canary_ctc, cohere, voxtral, voxtral4b, qwen3 |
| `core/ffn.h` | 4× inline SwiGLU blocks | qwen3, voxtral, voxtral4b, granite |
| `core/attention.h` | Llama-style self-attention with NEOX RoPE + GQA + flash-attn | voxtral, granite (via `core_granite_llm`) |
| `core/gguf_loader.{h,cpp}` | 8× identical two-pass GGUF load + mmap + tensor-map build | all non-whisper models |
| `core/fft.h` | Radix-2 Cooley-Tukey FFT (4× duplicated) | granite_speech, granite_nle (kokoro/mimo can adopt) |
| `core/cpu_ops.h` | CPU LayerNorm + matmul fallbacks (when no GPU sched is available) | granite_speech, granite_nle |
| `core/ctc.h` | `posterior_weighted_pool` + `greedy_decode_with_blank` | granite_nle (any aux-head/CTC variant can adopt) |
| `core/fastconformer.h` | NeMo-style FastConformer block (conv subsampling + MHA RPE) | parakeet, canary, canary_ctc |
| `core/conformer_ibm.h` | IBM Macaron Conformer block (FFN + Shaw RPE attn + conv module + FFN + Shaw lookup) — **sibling of `fastconformer.h`, intentionally not merged** | granite_speech, granite_nle |
| `core/granite_llm.h` | Granite-1B 40-block backbone (RMSNorm + GQA(16/4) flash-attn + RoPE + SwiGLU + µP residual scale); `is_causal` flag picks KV-cached prefill+decode (`core_attn::kv_self_attn`) vs non-causal flash (whole-sequence editing) | granite_speech, granite_nle |
| `core/qformer.h` | Windowed simplified Q-Former: pass A (LayerNorm + concat + linear + GELU) and per-window cross-attn + MLP cgraph builder | granite_nle (NAR-only — granite_speech uses a different full BLIP-2 Q-Former) |
| `core/bpe.h` | GPT-2 byte-level BPE encode + decode | granite_speech, granite_nle, voxtral, qwen3, glm-asr |
| `core/greedy_decode.h` | Autoregressive greedy decode loop with EOS handling | qwen3, voxtral, voxtral4b, granite, glm-asr |

`core_mel::Params` spans both algorithm clusters: the NeMo family
(`ln` + per-mel z-score + `(T, n_mels)` layout) and the HF/Whisper
family (`log10` + global clip normalization + `(n_mels, T)` layout),
with knobs for `LogGuard` (add-epsilon vs max-clip), `MatmulPrecision`
(`Float` vs `Double`), `FbLayout` (`MelsFreqs` vs `FreqsMels`),
`drop_last_frame` / `drop_first_frame_if_odd`, and `pad_to_T`.

`core_gguf::WeightLoad` owns the `ggml_context`, the
`ggml_backend_buffer_t`, and the `std::map<std::string, ggml_tensor*>`
in one struct that models `std::move()` into their own state. The
mmap path has a `pread` / `fseek` fallback for filesystems that don't
support mmap.

## Whisper is the reference implementation

`src/crispasr` is **intentionally not migrated** to `src/core/` (yet)
— it's (for the time being) the battle-tested reference and the
`crispasr -m ggml-base.en.bin …` code path is byte-identical to
upstream `whisper.cpp`. This guarantee is a test gate: every
CrispASR commit that touches the CLI is checked against it.

## Regression discipline

Every `src/core/` migration commit includes a `md5sum`-level
regression test against `samples/jfk.wav`:

- **mel extraction**: bit-identical transcript + SRT on parakeet,
  canary, canary_ctc, voxtral, voxtral4b, qwen3. Cohere transcript
  is bit-identical but a single SRT boundary shifts by 80 ms due to
  the CBLAS → manual-loop matmul accumulator reorder.
- **ffn extraction**: bit-identical on qwen3, voxtral, voxtral4b,
  granite.
- **gguf_loader extraction**: bit-identical on all 8 non-whisper
  models.
- **attention extraction**: bit-identical on voxtral (only consumer
  so far).

## Backend internals

> **Note:** the snapshot below was last hand-edited in early 2026 and
> is not regenerated from the registry — treat it as a sketch, not
> ground truth. The authoritative source for what's compiled in,
> per-backend GPU support, and current capability bits is
> `src/crispasr_model_registry.cpp` and the `capabilities()` returned
> by each adapter in `examples/cli/crispasr_backend_*.cpp`.

| Backend | Arch pattern | ggml graph | Flash attn | KV cache | GPU | Shared core modules |
|---|---|:-:|:-:|:-:|---|---|
| whisper | Enc-dec transformer | ✔ | ✔ | ✔ | CUDA / Metal / Vulkan | (upstream) |
| parakeet | FastConformer + TDT | ✔ | ✔ | partial | CPU | mel, fastconformer |
| canary | FastConformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel, fastconformer |
| cohere | Conformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel |
| granite | Conformer + Q-Former + LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| voxtral | Whisper enc + Mistral LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| voxtral4b | RoPE enc + 3.4 B LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| qwen3 | Whisper enc + Qwen3 LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| fc-ctc | FastConformer + CTC | ✔ | ✔ | — | CPU | mel, fastconformer |
| wav2vec2 | CNN + Transformer + CTC | ✔ | — | — | CUDA / Metal | gguf_loader |
| glm-asr | Whisper enc + Llama LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| kyutai-stt | Mimi codec + causal LM | ✔ | ✔ | ✔ | CPU | gguf_loader |
| firered-asr | Conformer + CTC + beam dec | ✔ | ✔ | ✔ | CPU | mel, gguf_loader |
| moonshine | Conv + 6L enc-dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| moonshine-streaming | Sliding-window enc + dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| omniasr | wav2vec2 enc + CTC / LLM | ✔ | ✔ | CTC: — / LLM: ✔ | CPU | gguf_loader, kv_self_attn, swiglu |
| gemma4-e2b | Conformer enc + Gemma4 LLM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| mimo-asr | wav2vec2 enc + Qwen2 LM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| vibevoice | σ-VAE + Qwen2 (ASR) / TTS LM (synth) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader |
| kokoro | StyleTTS2 BERT + ProsodyPredictor + iSTFTNet | ✔ | — | — | CPU | gguf_loader, fft, ffn |
| qwen3-tts | Qwen3 talker + 12 Hz codec + code-predictor | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| orpheus | Llama-3.2 talker + SNAC RVQ codec | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| chatterbox | T3 (Llama / GPT-2) + S3Gen (Conformer + UNet1D CFM + HiFTGen) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu, fft |
| m2m100 | facebook/m2m100 12L+12L transformer (text-to-text translation; WMT21 4.7B variant via `--backend m2m100-wmt21`) | ✔ | — | ✔ (cross-attn) | CUDA / Metal | gguf_loader, kv_self_attn |
| madlad / t5 | T5 encoder-decoder (MADLAD-400 12L+12L, gated-GELU, RMSNorm, bucketed rel-pos bias). Tokens match Python SP bit-by-bit; translation outputs match the HF reference. | ✔ | — | ✔ (cross-attn) | CUDA / Metal | gguf_loader, ffn |

### Architecture families

- **Feedforward CTC** (wav2vec2, omniasr-CTC, fc-ctc, firered-asr):
  No decoder, no KV cache. Fastest. No native punctuation.
- **Encoder-decoder** (whisper, canary, cohere, moonshine,
  moonshine-streaming): cross-attention KV cache, autoregressive
  text decoder.
- **Audio-LLM** (granite, voxtral, voxtral4b, qwen3, glm-asr,
  omniasr-LLM, gemma4-e2b, mimo-asr, vibevoice): audio features
  injected into LLM embedding space, KV-cached autoregressive
  decoding.
- **Transducer** (parakeet): LSTM predictor + joint network,
  frame-synchronous TDT decoding.
- **Codec + LM** (kyutai-stt): neural audio codec (RVQ) →
  token-based LM.
- **TTS — codec / vocoder pipeline**:
  - **Discrete-token codec + vocoder** (qwen3-tts, orpheus): talker
    LM emits codec tokens; a separate decoder GGUF (12 Hz codec /
    SNAC RVQ) renders the audio. Two-GGUF runtime.
  - **Flow-matching mel + iSTFT vocoder** (chatterbox / chatterbox-
    turbo / kartoffelbox-turbo / lahgtna-chatterbox): T3 emits speech
    tokens; S3Gen runs an UpsampleConformerEncoder + UNet1D CFM
    (10-step Euler for base / 2-step meanflow for turbo) producing a
    mel-spectrogram, then HiFTGenerator (conv chains + Snake +
    iSTFT) renders 24 kHz audio. Two-GGUF runtime.
  - **Realtime σ-VAE** (vibevoice in TTS mode): 4L base LM + 20L TTS
    LM + DPM-Solver++ + σ-VAE decoder.
  - **StyleTTS2 / iSTFTNet** (kokoro): BERT + ProsodyPredictor
    + iSTFTNet decoder, single-shot (no AR).
- **Text-to-text translation**:
  - **m2m100** (also runs WMT21 dense-24-wide-en-x via the same
    runtime — see `--backend m2m100-wmt21`): SentencePiece BPE
    + transformer encoder + transformer decoder (with cross-attn
    KV cache) + greedy decode. Source/target language codes prefix
    the encoder/decoder input streams.
  - **t5_translate / madlad** (MADLAD-400 3B-mt and any future
    T5-family translation model): T5 encoder-decoder with gated-GELU
    FFN, RMSNorm, bucketed relative-position bias, SentencePiece
    256K Viterbi-unigram tokenizer. Target language as `<2xx>` input
    prefix on MADLAD; encoder is otherwise language-agnostic. Tokens
    match Python SP bit-by-bit; translation outputs match the HF
    reference (validated end-to-end on flan-t5-small + MADLAD-3b).

  Both are driven by `--text "..." -sl <src> -tl <tgt>`.

### Optimization opportunities

- **Beam search** for all encoder-decoder and Audio-LLM backends —
  PLAN #63 added it for several LLM backends; whisper + firered-asr
  always had it.
- **Fused QKV** (single matmul for Q / K / V projections) — used in
  CrispEmbed, applicable to all attention layers; landed for
  qwen3-tts talker (Q8_0/Q4_K-skipped) under env flag
  `QWEN3_TTS_FUSED_QKV`.
- **Temperature sampling** for the few backends that don't have it
  (glm-asr, kyutai-stt, firered-asr, moonshine, omniasr-LLM) via
  `core_greedy_decode`.
- **GPU offload** for the still-CPU-only backends — needs
  `ggml_backend_sched` with GPU primary.

---

## Per-backend architecture details

Detailed architecture notes for backends whose design warrants more than
a one-line summary. The [README backend table](../README.md#asr-backends)
links here for each entry.

### granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar

**granite** (`granite-speech-{3.2-8b, 3.3-2b, 3.3-8b}`, `granite-4.0-1b-speech`):
Conformer encoder + BLIP-2 Q-Former + Granite LLM (μP scaling).

**granite-4.1** (`granite-speech-4.1-2b`): Same architecture as 4.0
(16-layer Conformer + Q-Former + Granite LLM); "2B" = full system.
Encoder runs as a single ggml graph by default with per-layer Shaw RPE
in attention (PLAN #16) — bit-near-identical to the per-op CPU loop,
~2.1× faster end-to-end on M1+Q4_K. `GRANITE_DISABLE_ENCODER_GRAPH=1`
falls back to the CPU loop.

**granite-4.1-plus** (`granite-speech-4.1-2b-plus`): 4.1 + 2-layer
encoder hidden-state concatenation (1024+1024=2048 projector input);
emits punctuated / capitalised transcripts by default. `cat_hidden_layers`
post-norm tensors are captured inline in the graph and `ggml_concat`-ed
with the final encoder output, so PLUS rides the GPU path too (~2.5×
end-to-end on M1+Q4_K).

**granite-4.1-nar** (`granite-speech-4.1-2b-nar`): 4.1 with
non-autoregressive decoder — single LLM forward over [audio, text+slots]
+ slot argmax decode (`is_causal=False` everywhere); 4-layer encoder
hidden-state concatenation + posterior-pooled BPE auxiliary CTC head;
bit-exact end-to-end on JFK via `crispasr-diff granite-nle`. Wired into
the main CLI as `--backend granite-4.1-nar` (alias `granite-nar`).
Encoder also runs as a single ggml graph (sibling builder with self-cond
residual + snapshot concat + final CTC logits), ~3× faster end-to-end on
M1+Q4_K.

### kokoro

StyleTTS2 / iSTFTNet (BERT + ProsodyPredictor + iSTFTNet decoder, 82M
params); per-voice GGUF; in-process libespeak-ng phonemizer with LRU
cache; auto-routing for `-l de` swaps in the German-trained backbone +
cascading voice fallback.

Models: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
+ [`dida-80b/kokoro-german-hui-multispeaker-base`](https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base)
(German backbone) + [`kikiri-tts/kikiri-german-{victoria,martin}`](https://huggingface.co/kikiri-tts)
(German voicepacks).

### orpheus

Llama-3.2-3B-Instruct talker (28L, 3072 d) + SNAC RVQ codec (3
codebooks × 4096 @ 24 kHz); 8 baked English speakers
(`tara`/`leah`/`leo`/...). Pick the speaker with `--voice <name>` and
pass `--temperature 0.6` (engine_class.py default — greedy loops).

Drop-in DE checkpoint variants:
- `--backend kartoffel-orpheus-de-natural` — [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF), 19 speakers, ASR-roundtrip word-exact via parakeet-v3 -l de
- `--backend kartoffel-orpheus-de-synthetic` — [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF), 4 speakers + 12 emotions + 5 outbursts via `{Speaker} - {Emotion}: {text}` syntax
- `--backend lex-au-orpheus-de` — `lex-au/Orpheus-3b-German-FT-Q8_0.gguf`

### chatterbox / chatterbox-turbo / kartoffelbox-turbo / lahgtna-chatterbox

Two-GGUF runtime: T3 AR text→speech-tokens + S3Gen flow-matching
speech-tokens→24 kHz waveform.

**T3 (Text-to-Tokens)**: Llama-30L for base/lahgtna, GPT-2-24L for
turbo/kartoffelbox-turbo.

**S3Gen (Tokens-to-Speech)**: UpsampleConformerEncoder + UNet1D CFM +
HiFTGenerator vocoder. Turbo uses 2-step meanflow CFM (vs 10-step cosine
for base). Default voice baked into T3 (`conds.*`); voice cloning
goes through `models/bake-chatterbox-voice-from-wav.py`, which runs
upstream `prepare_conditionals(wav)` (VoiceEncoder LSTM →
256-d speaker emb, CAMPPlus TDNN → 192-d x-vector, S3Tokenizer →
prompt tokens, 24 kHz mel extractor → prompt mel) and writes a small
voice GGUF (~150-200 KB) using the same tensor names the runtime
already accepts for the built-in voice. `--voice <voice.gguf>` then
loads it via `chatterbox_load_voice_gguf` into a separate
`voice_ctx_w` / `voice_buf_w` and rebinds `ctx->conds.*` pointers,
leaving the original baked-in default-voice tensors allocated but
unreferenced. In-process WAV → cond extraction is fully ported across four
modules: VE (`src/chatterbox_ve.cpp`), S3Tokenizer V2
(`src/chatterbox_s3tok.cpp`), CAMPPlus + Kaldi fbank
(`src/chatterbox_campplus.cpp` + `src/core/kaldi_fbank.{h,cpp}`),
and 24 kHz Matcha mel (in `chatterbox_campplus.cpp`). All are
verified bit- or fp32-rounding-tight against PyTorch via
`crispasr-diff chatterbox`. A polyphase Kaiser-windowed sinc
resampler (`src/core/audio_resample.{h,cpp}`) handles the 16 ↔
24 kHz conversion. The runtime forks on the input rate when
`--voice` is a `.wav` path: 24 kHz input triggers atomic cloning
(all five conds derived from one source); 16 kHz input keeps the
T3-side-only partial path (the S3Gen-side `gen.*` triple stays at
default to avoid the inconsistent-conditioning silence trap).
Output may drift from the python baker due to the resampler
differing slightly from librosa kaiser_fast; for perfect parity
the python baker workflow remains recommended.
S3Gen GGUF is auto-discovered next to T3 or passed via `--codec-model`.
See [`docs/tts.md`](tts.md#voice-cloning) for the workflow.

Variants:
- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — base, English
- [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) — 350M distilled, meanflow
- [`cstr/kartoffelbox-turbo-GGUF`](https://huggingface.co/cstr/kartoffelbox-turbo-GGUF) — German fine-tune of turbo
- [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) — Arabic fine-tune of base

Conformer rel-pos parity gap closed in §80 — encoder_out now bit-exact
to Python reference.

### omniasr (CTC + LLM + Unlimited)

wav2vec2-style CNN frontend (7 layers, stride 5+2×6=320) + 24–48L
transformer encoder + either CTC head or 12L LLaMA decoder (SwiGLU,
RoPE, d=4096, 8 heads).

**CTC variant**: greedy argmax with CTC blank collapse.

**LLM variant** (`omniasr-llm-300m-v2`): Encoder projection (1024→4096)
+ language conditioning (1694 FLORES-200 codes) + autoregressive decode.
Best quality for the 1600+ language family.

**Unlimited variant** (`omniasr-llm-unlimited-300m-v2`): Same architecture
but trained with a streaming segment-token protocol. Audio is split into
15-second segments; each segment is decoded independently with a segment
marker token that signals whether more audio follows. Three special tokens
above vocab_size in tok_emb: `streaming_lang` (lid marker),
`last_segment`, `regular_segment`. Auto-detected at load time from
tok_emb shape (vocab_size + 3). Supports arbitrarily long audio input.

### vibevoice

σ-VAE ConvNeXt encoders + Qwen2.5-7B decoder. Dual-mode: ASR (with
timestamps, diarization, hotwords) and TTS (DPM-Solver++ flow matching).

### mimo-asr

6L input_local_transformer (1024d) + 36L Qwen2 LM (4096d, 32Q/8KV);
8-channel RVQ codes from separate MiMo-Audio-Tokenizer GGUF
(`--codec-model`). Mandarin (Wu/Cantonese/Hokkien/Sichuanese dialects)
+ English + code-switching.

### qwen3-tts

Qwen3 talker LM + 12 Hz RVQ speech tokenizer. Three variants:
- `qwen3-tts-0.6b-base` — 0.6B talker, baked voice pack or WAV + `--ref-text`
- `qwen3-tts-1.7b-base` — 1.7B talker, higher quality
- `qwen3-tts-1.7b-voicedesign` — natural-language voice description via `--instruct`

### m2m100 / wmt21

12L encoder + 12L decoder transformer (d=1024, 16 heads, FFN=4096, ReLU,
pre-norm) + SentencePiece BPE (128K vocab, 100 language codes) +
sinusoidal positional embeddings + cross-attention KV cache + greedy
decode. en→de exact match to the Python reference; Q8_0 (~502 MB)
preserves quality.

**WMT21** (`wmt21-dense-24-wide-en-x`): Same architecture scaled to 4.7B
parameters (24L encoder, wider FFN). Won the WMT21 News competition.
Routes through the same m2m100 runtime.

### madlad

T5 encoder-decoder (12L+12L, d=2048, gated-GELU FFN, RMSNorm, bucketed
relative-position bias) + SentencePiece (256K vocab). Target language
specified as `<2xx>` input prefix. Tokens match Python SentencePiece
bit-by-bit; output matches HF reference.
