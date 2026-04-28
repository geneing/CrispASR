# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

**Current state (April 2026, v0.5.3):** 21 ASR backends + TTS, unified CLI,
OpenAI-compatible server, shared `src/core/` library, FireRedPunc
post-processor, C-ABI + Python/Rust/Dart wrappers, CI on 6 platforms.
All backends support `-m auto --auto-download`. Three new ggml ops
(`conv_1d_cf`, `conv_1d_dw_cf`, `conv_1d_group`). ggml bumped to 0.10.0.

---

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **DONE** | [#54 NeMo-cluster encoder cos](#54-nemo-cluster-encoder-cosine-divergence-parakeet-post-mel-fix) | Medium-Large | bias-load fix: cos_mean 0.79 → 0.996 |
| **MEDIUM** | [#5 Reference backends](#5-reference-backends-for-parakeetcanarycohere) | Medium | parakeet/cohere DONE; canary remaining |
| **LOW** | #41 Moonshine IPA / phoneme | High | Deferred |
| **LOW** | ~~#40b Moonshine streaming~~ | ~~High~~ | **DONE** (3 sizes) |
| **LOW** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | High | |
| **LOW** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | |
| **LOW** | [#11 WebSocket server](#11-websocket-streaming-server) | High | |
| **LOW** | [#16 Shaw RPE](#16-shaw-rpe-for-granite-graph) | Medium | |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | Needs ≥16 GB RAM |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | License unclear |

---





## 40. More Moonshine model variants

Convert + upload to HuggingFace:
- ~~`moonshine-base` (61.5M, better WER)~~ **DONE** (cstr/moonshine-base-GGUF)
- `moonshine-streaming-tiny/small/medium` — different architecture, needs new runtime
- ~~`moonshine-tiny-{ja,ar,ko,zh,vi,uk}` (multilingual)~~ **DONE** (12 repos on HF)
- ~~`moonshine-base-{ja,uk,vi,zh,ar,ko}` (multilingual)~~ **DONE** (12 repos on HF)

Converter fix: 1D tensors (norms, biases) forced to F32; conv_1d_f32 mul_mat
argument order fixed for F16 kernels.

---

## 41. Moonshine phoneme / IPA output

moonshine-ai/moonshine has a `GraphemeToPhonemizer` — G2P (text→IPA),
NOT audio→phoneme. Runs on transcription output.

**Options:**
1. Port G2P tables to C++ (~500 LOC, needs pronunciation dicts)
2. Post-processing module with `--output-ipa` flag
3. External-only (document piping through Python G2P)

**Recommendation:** Option 3 for now. IPA is niche; ROI of porting is low.

---

## 5. Reference backends for parakeet/canary/cohere

Write `tools/reference_backends/{parakeet,canary,cohere}.py` for
`crispasr-diff` reference activation comparison.

**Effort:** ~100-150 LOC per backend.

**Status (April 2026):** `parakeet` reference backend done — used to
diagnose the JA xscaling bug. Cohere already has one
(`reference_backends/cohere.py`). Canary is the only one still missing.

---

## 7. Native voxtral4b streaming

Expose voxtral4b's native 240ms-2.4s latency streaming via pre_hook
audio frame injection. Needs threading (encoder thread + decoder thread).

**Effort:** ~200-300 LOC. High complexity.

---

## 9. Parakeet TDT decoder GPU

Port LSTM predictor + joint head from CPU loops to ggml graphs. LSTM
is sequential → per-step kernel launches. Encoder already 85%+ of time.

**Effort:** ~150 LOC. Small gain.

---

## 11. WebSocket streaming server

Add `/ws` endpoint for real-time streaming over HTTP. httplib doesn't
support WebSocket — need custom protocol or library.

**Effort:** ~200-300 LOC.

---


## 16. Shaw RPE for granite graph

Add query-dependent Shaw RPE to granite ggml encoder graph (currently
uses flash_attn_ext without RPE). Manual attention: QK^T + RPE bias +
softmax + V matmul.

**Effort:** ~80 LOC.

---


## 42. VibeVoice-ASR 7B

**BLOCKED:** Needs ≥16 GB RAM for conversion. Converter OOMs on 8 GB due
to Qwen2.5-7B embedding (152064 × 3584 = 2.1 GB F32).

**Fix:** Use `safe_open` per-tensor conversion. Then Q4_K → ~4 GB.

Full architecture analysis in HISTORY.md #34. C++ runtime partially
implemented (`src/vibevoice.cpp`). F16 im2col precision issue in
depthwise conv needs fixing.

---

## 43. Fun-ASR-Nano

**BLOCKED:** License unclear. Issue filed at `FunAudioLLM/Fun-ASR#99`.
No response. HF model card has no license field.

---

## 50. Gemma-4-E2B runtime refactor — **DONE** (April 2026)

Working end-to-end. Q4_K transcribes JFK perfectly: *"And so my
fellow Americans ask not what your country can do for you, ask
what you can do for your country."*

Per-stage cos vs HF reference:

```
mel_spectrogram          1.0000  bit-exact
audio_subsample_output   1.0000
audio_layer_0..7        >0.998
audio_layer_11           0.969
audio_tower_output       0.962
encoder_output           0.966
```

Bugs fixed (full list in HISTORY.md and `LEARNINGS.md`):

1. Attention scale = 1.0 (q_norm replaces 1/√d, not the other way around).
2. v_norm RMSNorm-without-weight on V before flash-attn.
3. layer_scalar applied ONCE at end of layer.
4. PLE block at end of layer + per_layer_inputs prep stage.
5. KV-share donor map (LAST 20 layers reuse, not first 20 — converter
   metadata + runtime both honour it now).
6. LLM MLP is GeGLU not SwiGLU (`gelu_pytorch_tanh`).
7. Audio attn Q/K scaling (`q_scale·softplus(per_dim_scale)`,
   `k_scale=log(1+e)/log(2)`).
8. Audio subsample LayerNorm + ReLU (was RMSNorm + SiLU).
9. Audio lconv1d SiLU between conv_norm and out_proj.
10. Audio chunked-local attention with relative position bias
    (block-wise manual attention; flash_attn_ext can't express the
    softcap-before-mask order HF needs).
11. Audio→LLM adapter pre-projection RMSNorm.
12. HF-faithful mel FE (`fft_length=512`, `frame_length=320`,
    semicausal pad, `log(mel + mel_floor=0.001)`, no Slaney norm).
13. Subsample axis order: ne=(n_mels, T_mel) input, (M, T, C)→(C, M, T)
    flatten with C-fast for HF's per-frame feature ordering.
14. **ClippableLinear QAT scalars**: 480 trained scalars per audio
    tower applied via `clamp(input)→matmul→clamp(output)` per HF's
    `Gemma4ClippableLinear.forward`. Stop-skipping in converter +
    runtime support; this was the dominant remaining bug.

Open follow-ups (not blockers): see TODO under #50 — further per-token
decode optimisation (now 1.4× realtime after the `end_of_turn` eos
fix; ~220 ms/tok dominated by 35-layer + double-wide-MLP + PLE),
audio hparams to GGUF for multi-flavour support, CrispAudio
shared-lib extraction.

---

## 51. MiMo-V2.5-ASR runtime

Converter done (`models/convert-mimo-asr-to-gguf.py` plus the
audio-tokenizer side `models/convert-mimo-tokenizer-to-gguf.py`).
Runtime not yet written.

- **Architecture:** 6-layer input_local_transformer + 36-layer Qwen2
  LLM (4096d, 32Q/8KV, SiLU, RoPE).
- **Audio path:** 8-channel RVQ tokens from `cstr/mimo-tokenizer-GGUF`
  (separate model) get fed into the input_local_transformer, which
  produces embeddings the Qwen2 LLM consumes.
- **Reuse:** Qwen2 LLM core can lean on `core_attn::kv_self_attn` +
  `core_ffn::swiglu` — both already in production via qwen3-asr.

**Effort:** Large (~1000 LOC), but most of it is the audio-tokenizer
runtime (RVQ encoder/decoder over conv stacks). The Qwen2 part
mostly slots into the existing core helpers.

---

## 52. Qwen3-TTS

User-requested follow-on to the VibeVoice TTS work. Apache-2.0
collection: [Qwen/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
[HF collection](https://huggingface.co/collections/Qwen/qwen3-tts).

- **Six repos in the collection** (all BF16 safetensors, Apache 2.0):
  - `Qwen/Qwen3-TTS-Tokenizer-12Hz` — RVQ codec, 16 codebooks × 2048,
    12.5 FPS at 24 kHz. Non-DiT lightweight architecture (8L
    encoder + 8L decoder).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` — base talker LM with
    voice clone (3s reference audio).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` — fine-tuned,
    fixed speakers.
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — instruction-tuned
    (voice description → speech).
- **Architecture:** "Discrete Multi-Codebook LM" — Qwen3 backbone
  with a 16-codebook output head. No DiT; direct AR generation of
  RVQ codes. ~97ms end-to-end latency, 10 languages incl.
  en/de/zh/ja/ko/it.
- **Status (April 2026):** **talker + ICL prefill + code_predictor live; intelligible synthesis verified (codec via Python)**.
  - Converter (`models/convert-qwen3-tts-to-gguf.py`) maps the
    actual HF tensor namespace (`talker.model.codec_embedding` →
    `talker.token_embd`, `talker.codec_head` → `talker.output`,
    `talker.text_projection.linear_fc{1,2}` → `talker.text_proj.fc{1,2}`,
    speaker_encoder + 15-CB code-predictor likewise). Writes BPE
    merges so subword tokenisation works. **Verified** end-to-end:
    478 tensors, 0 unmapped, `qwen3-tts-0.6b-base.gguf` loads.
  - Runtime: 28L Qwen3 talker forward via `core_attn::kv_self_attn`
    + `core_ffn::swiglu` (single-axis NEOX RoPE — text-only collapse,
    swap to `ggml_rope_multi` once codec splice lands). F16 KV cache.
    `qwen3_tts_synthesize_codes(text)` AR-decodes codebook-0 from
    `codec_head`. Empirically produces 923 codes for "Hello, this
    is a test." on Metal.
  - Diff harness wired: `tools/reference_backends/qwen3_tts.py`
    drives the qwen-tts pip package on CPU and dumps stage tensors
    to a GGUF archive. `crispasr-diff qwen3-tts` reads `text_input_ids`
    from the archive and runs `qwen3_tts_run_text_proj` against
    `text_proj_out` → **PASS at cos_min=1.000000** for the
    text_embedding + resize-MLP path on the "Hello world." prompt.
  - Debug knobs: `QWEN3_TTS_{BENCH,DEBUG,DUMP_DIR}` env vars in the
    style of `GEMMA4_E2B_BENCH` / `OMNIASR_DUMP_DIR`.
  - **Verified milestones (in landed order):**
    1. ✓ Talker forward (28L Qwen3 + Q/K-norm + flash-attn + F16 KV
       cache). `talker_logits` PASS at cos_min=1.000000 against
       PyTorch when fed the same prefill (commit `2b85b78`).
    2. ✓ ICL prefill builder. Independent C++ implementation of
       `Qwen3TTSForConditionalGeneration.generate_icl_prompt` —
       chat-template prompt + codec sentinels + speaker_embed (from
       voice pack) + per-frame summed 16-codebook ref_code embeddings
       + tts_pad/bos/eos splice. `talker_logits_via_icl` PASS at
       cos_min=1.000000 (commit `b939d4f`).
    3. ✓ Code predictor (5L Qwen3, 15 separate codec_embedding +
       lm_head pairs, top-k=50 + temperature=0.9 sampling). Greedy
       was a silent-output trap; sampling matches the reference
       `subtalker_dosample=True` default (commit `9608202`,
       `69c135c`).
    4. ✓ Roundtrip: TTS-out → ASR-in. Our pipeline synthesises
       "The quick brown fox jumps over the lazy dog." → 55 frames
       × 16 codebooks → Python codec decode → 4.4s audio →
       parakeet ASR transcribes back verbatim.
  - **Open** (in priority order):
    1. ✓ **Codec decoder** (Tokenizer-12Hz, commits `d1f47b1`, `48c6c1a`).
       Converter rewritten (0 unmapped, 253 tensors, 0.25 GB F16 GGUF).
       C++ decoder: SplitRVQ → pre_conv → 8L XFMR(512d, sliding-window=72)
       → 2× ConvNeXt upsample → 4× SnakeBeta+tconv DecoderBlock → PCM.
       Diff harness: 8/8 stages PASS (cos_min≥0.999). CPU verified.
       Metal: GPU scheduler conflict on M1 — use_gpu=false for codec;
       investigate separately as PLAN #52 open item.
    2. **Runtime ECAPA speaker_encoder forward.** Removes the
       `bake-qwen3-tts-voice-pack.py` dependency for new voices —
       end users pass any ref WAV and we compute spk_embedding in
       C++. ~250 LOC: mel(24kHz) → TDNN → 3 SE-Res2Net → MFA →
       ASP → Conv1×1 → 1024-d. Reference: `Qwen3TTSSpeakerEncoder`
       in `ref/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py`.
    3. **Runtime codec encoder forward.** Mimi-based encoder (used
       in voice cloning to extract `ref_code` from the reference
       WAV). Closes the loop on the bake script — pure C++ pipeline.
       Larger effort (Mimi is its own architecture).
    4. **Performance pass.** Current AR step is ~137 ms/frame on
       M1 Metal — 1.7× slower than real-time at 12.5 fps. Bottleneck
       is the 15 sequential code_predictor graph builds per frame.
       Fuse them into a single graph with all 15 lm_heads and KV
       writes baked in. Also: fused QKV in talker (qwen3_asr has
       this; qwen3_tts could too), Q4_K weights (1.83 GB → 480 MB).
- **Reuse:** the talker is essentially Qwen3-0.6B/1.7B with a
  multi-codebook output head — `core_attn::kv_self_attn` +
  `core_ffn::swiglu` again. The codec needs new code for RVQ
  decoding; that work is shared with MiMo (#51) and overlaps in
  shape with the VibeVoice σ-VAE decoder, so a `core_audio_decoder`
  helper is worth landing alongside the runtime (see #53).

**Effort:** Large. ~1500 LOC across runtime + codec + reference
backend. The two TTS targets (Qwen3-TTS and any future expansion)
share enough that landing one substantially de-risks the other.

---

## 54. NeMo-cluster encoder cosine divergence — **DONE** (April 2026)

Resolved by loading the per-layer bias tensors that the parakeet
loader was previously skipping. After the fix:

| sample | mel cos | enc cos before | enc cos after |
|---|---|---|---|
| reazon_meal_11s | 0.999 | 0.792 | **0.996** |
| jsut 3.19s | 0.998 | 0.806 | (similar) |

Per-layer diff confirmed the divergence started at `encoder_layer_0`
(immediately after a bit-exact `pre_encode_output`), localising the
bug inside the conformer block. The GGUF stored `attn.{q,k,v,out}.bias`
+ `{ff1,ff2}.linear{1,2}.bias` + `conv.{pw1,pw2}.bias` (10 biases per
layer × 24 layers = 240 tensors), but `parakeet_load_weights` only
loaded the weights — the bias slots stayed nullptr, and `mm_bias`
silently skipped the bias add. The fix was 10 lines:
`e.X_b = try_("…bias")` for each missing slot in `parakeet_load_weights`.

`try_get()` rather than `require()` keeps the loader compatible with
v3 (which has `use_bias=False` and stores no biases at all). v3 EN
on JFK still passes regression. Documented in LEARNINGS.md.

Residual issue (open follow-up, lower priority): on layers 17-22 the
cos_mean stays high (~0.99) but cos_min crashes to negative values on
specific frames — `encoder_layer_22` cos_min = -0.67. Indicates a
small number of frames are catastrophically wrong while the bulk
match. Suspects: rel_shift edge cases, specific position-encoding
positions where numerical instability surfaces, or a buffer-aliasing
issue in the dump path that masks an analogous issue in production.
Not blocking the bug-report fix.

The diagnostic infrastructure landed (per-layer captures in
`reference_backends/parakeet.py`, `parakeet_run_encoder_dump` C-API,
`encoder_output_ref_mel` + `encoder_layer_K` stages in the diff
harness) is reusable for canary, canary_ctc, and any future
NeMo-cluster runtime debug.

---

## 54-historical. NeMo-cluster encoder cosine divergence (original analysis)

After the preemph + Bessel-corrected PerFeatureZ fix
(`mel_spectrogram cos_mean = 0.999451` on reazon_meal_11s, up from
0.990, with the major-deletion symptom from issue #37 gone), the
24-layer FastConformer encoder still diverges from NeMo:

| sample              | mel cos_mean | enc cos_mean | enc cos_min |
|---------------------|--------------|--------------|-------------|
| reazon_meal_11s.wav | 0.999451     | 0.791530     | 0.476437    |
| jsut 3.19s          | 0.998316     | 0.805847     | 0.258203    |

**Why this matters:** transcripts on conversational JA still have
small hallucinations (`本当` prefix on the meal sample) and partial
syllables (`うん` → `どう`). The cos drop is too large to be pure
mel propagation through residuals — likely a bug in one of the
conformer pieces.

**Where to look (highest-yield first):**

1. **Add per-layer encoder captures to the diff harness.** The Python
   reference (`tools/reference_backends/parakeet.py`) currently only
   captures `encoder_output`. Add `encoder_layer_0..23` via
   `register_forward_hook` on each `model.encoder.layers[i]`, plus
   `pre_encode_output` after the dw_striding subsampling. Then add
   matching extraction points in `examples/cli/crispasr_diff_main.cpp`
   (or a new `parakeet-test-encoder` example that returns
   intermediates by index). Cos drop layer-by-layer pinpoints whether
   the bug is in subsampling, layer 0, or compounding through layers.

2. **Pre-encode (subsampling) is the most likely culprit.** The
   `core_conformer::build_pre_encode` ends with a permute+reshape
   from `(W3, H3, C, 1)` to `(W3*C, H3)`. NeMo's `Subsampling`
   module flattens differently — the order in which freq and channel
   dimensions are zipped before the linear layer determines whether
   `out_w` sees the same input as PyTorch. Triple-check vs
   `nemo.collections.asr.modules.subsampling.ConvSubsampling`.

3. **rel_shift's stride math.** The `(2T-1, T, H) → (T, T, H)` view in
   `core_conformer::rel_shift` is correct in principle but easy to get
   subtly off-by-one. The view uses
   `nb1 = a->nb[1] - a->nb[0]` and `offset = (T-1) * a->nb[0]`. If
   the offset or stride is wrong by one element it stays "almost
   right" — cosine ~0.9 not catastrophic.

4. **xscaling placement.** Current code applies `sqrt(d_model)` to
   the pre-encode output BEFORE the rel-pos sinusoidal table is
   added. NeMo's `RelPositionalEncoding` scales the input then adds
   pos to keys/values, but the `pos_enc` we materialise is fed into
   the BD branch of attention only. Verify that we're not
   double-scaling or missing a scale somewhere in the rel-pos
   contribution.

5. **As a sanity check**, run the diff harness with `mel_spectrogram`
   from the reference *substituted* for our C++ mel (requires a small
   harness branch). If `encoder_output` cos jumps to ~1.0 with the
   reference mel as input, the residual encoder error is pure mel
   propagation through residuals and we should chase further mel
   bit-exactness instead. If it stays at ~0.8, there's a real
   encoder bug.

**Why it wasn't caught earlier:** parakeet-tdt-0.6b-v3 (the English
model) is robust to this divergence and produces correct transcripts.
parakeet-tdt-0.6b-ja amplifies it on conversational audio because the
Japanese subword vocabulary is denser per second, so each frame
deletion from the encoder costs more transcript content.

**Effort:** Medium-large. Step 1 (per-layer hooks + matching C++
extraction) is ~150 LOC; the actual fix depends on what the diff
shows. Likely 1-3 days end-to-end.

**Files:**
- `tools/reference_backends/parakeet.py` — add per-layer hooks
- `examples/cli/crispasr_diff_main.cpp` — add `encoder_layer_K`
  comparison stages for the parakeet branch
- `src/parakeet.cpp` / `src/parakeet.h` — add a per-layer
  `parakeet_run_encoder_to_layer(K)` entry point
- `src/core/fastconformer.h` — likely fix once diff localises it

---

## 53. `core/audio_decoder.h` — DRY across TTS / codec backends

VibeVoice TTS (σ-VAE), Qwen3-TTS (RVQ codec) and MiMo
(MiMo-Audio-Tokenizer) all share the "discrete codes → 24 kHz
waveform" shape: a stack of 1D conv up-sampling blocks driven by a
small transformer or directly by quantised codes. Pull the recurring
patterns into `src/core/audio_decoder.h`:

- RVQ codebook lookup (multi-stage residual) — same in MiMo and
  Qwen3-TTS, plus their semantic-quantiser variant.
- 1D-conv up-sampling stack (`ggml_conv_1d_cf` already there from
  vibevoice work — reuse).
- Optional small transformer on the latent stream (Qwen3-TTS has 8L
  decoder transformer).

Same layering pattern as `core_conformer` / `core_attn` /
`core_ffn`. ~200-300 LOC in headers; payoff is two new TTS backends
plus future ones (e.g. Mimi/Encodec) sharing the same decode path.

**Effort:** Medium. Easier *after* one TTS runtime exists — extract
upward rather than design forward.

---


## Ecosystem expansion (lower priority)

### New backends from PazaBench assessment (see HISTORY.md #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |
| **XiaomiMiMo/MiMo-V2.5-ASR** | TBD (check) | LLM-style multimodal speech (similar to Qwen3-ASR pattern) | Medium — user-requested in #35 |
| **google/gemma-4-E2B** | Gemma terms | Conformer + Gemma 4 decoder; matches "Gemma 4 Audio" entry below | Medium — user-requested in #35 |

### From llama.cpp (MIT)

| Model | Architecture | Notes |
|---|---|---|
| Ultravox | Whisper encoder + Llama 3.2 1B/8B | Speech understanding |
| Gemma 4 Audio | Conformer, chunked attention | Streaming, multimodal |
| LFM2-Audio | Conformer variant | Position embeddings |

### Post-processing

| Model | License | Type | Priority |
|---|---|---|---|
| FireRedPunc | Apache-2.0 | BERT punct (zh+en) | **DONE** |
| fullstop-multilingual | MIT | XLM-R punct (en/de/fr/it) | Medium |
| bert-restore-punctuation | MIT | BERT punct+truecase (en) | Medium |
| xashru/punctuation | Apache-2.0 | XLM-R+BiLSTM-CRF (40+ langs) | Low |

### Optimizations (cross-cutting, from survey + CrispEmbed comparison)

| # | Optimization | Applies to | Expected gain | Status |
|---|---|---|---|---|
| O1 | `ggml_soft_max_ext` fusion | wav2vec2, canary, fastconformer | -10% wav2vec2 | **DONE** |
| O11 | wav2vec2 CNN → ggml | wav2vec2 family | **10.8x** | **DONE** |
| O9/#44 | FireRed ggml Q4_K decoder | firered-asr | **6.3x** | **DONE** |
| O10 | Sliding window attention | voxtral4b | Already implemented | **DONE** |
| O2 | Fused QKV pre-merge | LLM decoders | ~10-15% attn (GPU) | API ready in core/attention.h; CPU gain <1%, defer to GPU |
| O3 | Temperature sampling | glm-asr, kyutai-stt | Feature parity | **DONE** |
| O5 | Pipelined mel+encode | LLM backends, CPU | ~15-20% | TODO |
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | TODO |
| O6 | Batched encoder (GPU) | All + GPU | 3-5x | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | TODO |
| O12 | `ggml_conv_1d_cf` channels-first conv | vibevoice VAE | **-29% VAE, -15% total** | **DONE** |
| O13 | `ggml_conv_1d_group` + CNN cleanup | wav2vec2 family | **-12% total** (pos -12%, CNN -22%) | **DONE** |
| O14 | `--tts-steps` configurable DPM steps | vibevoice TTS | **-31% diffusion** | **DONE** |
| O15 | Remove redundant neg base LM | vibevoice TTS | Eliminated 60 LOC of wasted compute | **DONE** |

**From COMPARISON.md (llama.cpp patterns):**
- `ggml_soft_max_ext` with baked scale (O1) — already in llama.cpp, saves one `ggml_scale` op per attention layer
- Chunked window attention (O10) — llama.cpp uses for Gemma4A Conformer
- Conv2d subsampling via ggml ops — llama.cpp does this for Qwen3-ASR encoder

**From CrispEmbed (shared core patterns):**
- Fused QKV (O2) — CrispEmbed pre-merges Q/K/V weights at init, one matmul instead of 3
- SentencePiece Viterbi DP tokenizer — CrispEmbed has proper optimal tokenization
- Lazy graph allocation (`no_alloc=true` + scheduler) — reduces memory churn

**From LEARNINGS.md (FireRed decoder triage):**
- Small per-step ggml graphs are SLOWER than CPU loops (scheduling overhead)
- BUT: native Q4_K matmuls via ggml are 9.3x faster than F32 OpenMP (lesson: never dequant)

### Audio format support

- `.m4a`, `.mp4`, `.webm` crash with upstream ffmpeg integration — needs fix or robust fallback
- `.aiff`, `.wma`, raw PCM not supported without pre-conversion
- Consider bundling a lightweight M4A/AAC decoder or improving the ffmpeg path
- Only move LARGE, REUSED matmuls onto ggml/GPU
- Persistent subgraphs per decode step > one-off graphs

### Other

- **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches
- ~~**TTS module** — VibeVoice-Realtime-0.5B text-to-speech~~ **DONE** — perfect ASR round-trip on all test cases. 17 bugs found via stage-by-stage diff. Uses DPM-Solver++, dual KV CFG, voice prompts, EOS classifier, text/speech interleaving.
- ~~**ggml_conv_1d_dw F16 im2col fix**~~ **DONE** — solved via `ggml_conv_1d_dw_cf` (direct F32, no im2col)

---

## Publish language wrappers to package registries

Today the Rust, Dart, and Python wrappers all live in this repo and (for
Python) require a `pip install -e .` from a clone. Move all three onto
their language-native registries so users can install with one command.

**Status (2026-04-25):** All three wrappers now have publishable
metadata + dry-runs pass. The CI workflow `release-wrappers.yml` is
wired up but cannot run until the **one-time registry setup** below
is complete.

| Wrapper | Pre-flight | Blocker |
|---|---|---|
| Python `crispasr` 0.4.9 | sdist + wheel build clean | PyPI trusted-publisher must be configured |
| Dart `crispasr` 0.4.9 | `dart pub publish --dry-run` passes (warnings only) | pub.dev automated publishing must be configured |
| Rust `crispasr-sys` 0.1.7 | `cargo publish --dry-run` clean (5.9 KiB) | needs `CARGO_REGISTRY_TOKEN` repo secret |
| Rust `crispasr` 0.1.7 | publish-order dependent on `crispasr-sys` | same |

### One-time registry setup (must happen before first tag)

1. **PyPI** — go to https://pypi.org/manage/account/publishing/ and add
   a "pending publisher": owner `CrispStrobe`, repo `CrispASR`,
   workflow `release-wrappers.yml`, environment `pypi`. Then push any
   `v*` tag.
2. **crates.io** — generate a token at https://crates.io/me, add it
   as the `CARGO_REGISTRY_TOKEN` secret on the GitHub repo.
3. **pub.dev** — go to https://pub.dev/packages/crispasr/admin (after
   first manual publish or claim) → enable automated publishing → set
   tag pattern `v{{version}}`. Alternatively for the first publish,
   run `dart pub publish` locally with the package owner's credentials.

### Pattern (matches whisper.cpp approach)

All three wrappers are thin FFI/ctypes shims over the C ABI in
`src/crispasr_c_api.cpp`. They do **not** bundle the native library — the
user must have `libcrispasr.{so,dylib,dll}` installed (Homebrew, apt, or
built from source). This keeps the wheels/crates/pub packages tiny and
avoids a per-platform build matrix on every release.

| Wrapper | Registry | Effort | Notes |
|---|---|---|---|
| Python | PyPI | Low | Add `python/pyproject.toml`; pure-Python wheel; `_helpers.c` builds at install if a C toolchain is present, else falls back to ctypes-only path |
| Rust   | crates.io | Low | `crispasr-sys` then `crispasr` (two `cargo publish` calls); already has `Cargo.toml` |
| Dart   | pub.dev | Low | `flutter pub publish --dry-run` then `flutter pub publish`; already has `pubspec.yaml` |

### Library discovery (Python)

Update `_find_lib()` in `python/crispasr/_binding.py` to probe, in order:
1. `$CRISPASR_LIB_PATH` env var (explicit override)
2. `sys.prefix/lib/` (system or virtualenv install)
3. Standard Homebrew/Linux paths (`/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`)
4. Existing repo-relative fallbacks (for `pip install -e .` from a clone)

If none found, raise `RuntimeError` with a helpful message linking to
install docs (the same pattern Tesseract / faster-whisper use).

### Release automation

Add a tag-triggered workflow `.github/workflows/release-wrappers.yml`
that, on `v*` tags, runs in parallel:
- `python -m build && twine upload` (PyPI, OIDC trusted-publishing — no API token)
- `cargo publish -p crispasr-sys && cargo publish -p crispasr` (crates.io, `CARGO_REGISTRY_TOKEN` secret)
- `dart pub publish --force` (pub.dev, OIDC publishing)

Trigger only on tag push, not on every commit. Version bumps stay
manual — bump `pyproject.toml` / `Cargo.toml` / `pubspec.yaml` together
in the same commit that creates the tag.

### Future: bundled wheels for Python

After the pure-Python release is out, add a follow-up release pipeline
using `cibuildwheel` to produce manylinux2014 + macOS arm64/x64 +
Windows wheels with `libcrispasr.*` bundled inside via `auditwheel` /
`delocate` / `delvewheel`. Same for Rust if we ever want
`crispasr-sys` to vendor the native build like `tch-rs` /
`onnxruntime-sys` do. Defer until pure-Python wheel is out and stable.
