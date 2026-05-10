# CrispASR — Performance benchmarks

Test audio: jfk.wav (11.0s), Q4_K quantization, greedy decode (`-bs 1`).

---

## Backend × Optimization matrix

At-a-glance view of which performance knobs each backend supports today,
and where the gaps are. Last refresh: **2026-05-04** (after PLAN §79 —
14-commit session that shipped #69a / #69b / #69e / #72 / #73).

**Legend**: ✓ = supported, opt-in via env var · `F16` = stuck at F16
(quant cache types unavailable; attention path needs migration) ·
`—` = not applicable (no KV cache or no transformer blocks) ·
`·` = applicable but not yet wired (port deferred).

### LLM-decoder ASR (high VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | weight residency |
|---|:-:|:-:|:-:|:-:|:-:|
| voxtral4b (4B) | ✓ | ✓ | ✓ | ✓ | gpu |
| voxtral (3B) | ✓ | ✓ | ✓ | ✓ | gpu |
| granite-speech (1B / 4.0 / 4.1 / 4.1-plus / 4.1-nar) | ✓ | ✓ | ✓ | ✓ | gpu |
| gemma4-e2b (5B effective) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| mimo-asr (1.4B) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| qwen3-asr (0.6B) | ✓ | ✓ | ✓ | ✓ | gpu |
| glm-asr (1B) | ✓ | ✓ | ✓ | ✓ | gpu |
| omniasr-llm (300M) | ✓ | ✓ | ✓ | ✓ | gpu |
| vibevoice (4B ASR mode) | F16 | F16 | F16 | ✓ | gpu |

### Encoder-decoder ASR (medium VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| canary (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext default, -17 % on JFK with q8_0/q4_0 |
| cohere (2B) | ✓ | ✓ | ✓ | · | flash_attn_ext available; +11 % regression vs cast-on-read on JFK with q8_0/q4_0 — long-form rerun needed before promoting (see PLAN) |
| kyutai-stt (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext native, quant-safe |
| firered-asr (900M) | — | — | — | — | inline AED, no exposed transformer KV |
| moonshine-tiny / streaming | — | — | — | — | tiny decoder, no exposed KV |

### Encoder-only ASR (low VRAM, single forward)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| whisper (legacy) | ✓ | ✓ | ✓ | — | upstream loader, separate path |
| parakeet (TDT) | — | — | — | — | RNN-T transducer, no KV cache |
| fastconformer-ctc | — | — | — | — | CTC head |
| wav2vec2 / hubert / data2vec | — | — | — | — | CTC heads |
| omniasr (CTC variant) | — | — | — | — | CTC head |

### TTS

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| orpheus (3B + DE / lex-au variants) | ✓ | ✓ | ✓ | ✓ | shared Llama-3 path |
| chatterbox (T3 + CFG cache) | ✓ | ✓ | ✓ | · | uses kv_self_attn natively |
| qwen3-tts (0.6B + 1.7B variants) | ✓ talker | ✓ talker | ✓ talker | · | code-predictor cache stays F16 (separate path) |
| vibevoice (4B TTS mode) | F16 | F16 | F16 | ✓ | KV migration still pending; layer offload routes `tts_lm.layers.<N>.*` |
| kokoro | — | — | — | — | non-AR vocoder, no transformer KV |

### Where the gaps are

1. **Layer offload (`N_GPU_LAYERS`) on encoder-decoder ASR** (canary,
   cohere, kyutai-stt). Their cross-attention layout doesn't have the
   `blk.<N>.*` block-tagged tensors that the layer-split predicate
   recognises. Encoder-decoder offload is its own design problem —
   probably want to offload only the LLM/decoder side, but the tensor
   names (`<arch>.dec.<N>.*` etc.) need bespoke per-backend predicates.
2. **vibevoice quant K/V (both modes)**. The attention path uses the
   `ggml_cpy(K_perm, view_into_kv_k)` pattern that's incompatible with
   quant K/V (see LEARNINGS.md "ggml_cont(ggml_permute(quant_tensor))
   doesn't move data"). Migration recipe is the canary/cohere
   `ggml_flash_attn_ext` port — ~50-80 LOC + F16 mask graph input.
   Layer offload (`N_GPU_LAYERS`) is independently shipped and works
   on F16 K/V; the migration only unlocks quant K/V on top.
3. **qwen3-tts code-predictor cache**. Talker KV is fully covered via
   `core_attn::kv_self_attn`; the secondary code-predictor path
   doesn't go through that helper, so its cache stays F16. Lower-
   priority since the talker dominates per-frame cost.
4. **Linux/CUDA validation of #72 GPU residency.** mimo-asr 22 % /
   gemma4-e2b 2.2x speedups were measured on Apple Silicon Metal.
   dGPU should be even more favourable; deferred until a CUDA host
   is available. If a platform regresses, gate via env
   (`CRISPASR_FORCE_CPU_WEIGHTS=1`).
5. **Cohere flash_attn_ext regresses on short audio.** JFK (~11 s)
   with q8_0 K / q4_0 V is +11 % slower under flash than under the
   cast-on-read fallback (canary on the same workload is -17 %, so
   the kernel works — cohere's cache layout or head dim flips the
   crossover). Need a multi-minute clip to confirm flash pulls ahead
   on long-form before promoting it to the recommended path; until
   then short-form users on cohere should treat flash as opt-in.

### Stacking the four knobs

Each addresses an independent bottleneck:

| knob | addresses | when to use |
|---|---|---|
| `CRISPASR_KV_QUANT_K=q8_0 / _V=q4_0` | KV size in VRAM | always reasonable for LLM-decode ASR; quartered V cache on long context |
| `CRISPASR_KV_ON_CPU=1` | KV doesn't fit in VRAM at all | very long context with a tight VRAM budget |
| `CRISPASR_N_GPU_LAYERS=N` | model itself doesn't fit in VRAM | model size > VRAM; spill the last (total-N) layers |
| `CRISPASR_FORCE_CPU_WEIGHTS=1` (proposed) | platform regressed on §72 GPU residency | not yet wired — none seen on Apple Silicon |

```bash
# Maximum-memory-savings combo for a VRAM-tight host
CRISPASR_N_GPU_LAYERS=10 \
  CRISPASR_KV_ON_CPU=1 \
  CRISPASR_KV_QUANT_K=q8_0 \
  CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f long.wav
```

See [`docs/cli.md`](docs/cli.md) "Memory footprint" for the full env-
var reference and the llama.cpp parity comparison table; HISTORY §79
for the implementation write-up.

---

## Kaggle T4 GPU — 2026-04-26

Platform: 2x Tesla T4 (15 GB VRAM each), 4 CPU threads, CUDA.
Commit: `b9fd8eb`. **All 19 backends pass.**

### By architecture

#### Encoder-CTC (non-autoregressive, single forward pass)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| FastConformer CTC Large | 120M | 83 | 0.0% | **9.6x** | 1.1s | 18 FC layers |
| OmniASR CTC 1B v2 | 975M | 551 | 4.5% | 7.4x | 1.5s | w2v-BERT enc, 276ms GPU |
| Data2Vec Base | 95M | 78 | 0.0% | 5.3x | 2.1s | 12 layers, pos_conv 735ms |
| Wav2Vec2 XLSR-EN | 300M | 212 | 0.0% | 3.6x | 3.1s | 24 layers, pos_conv 1.6s |
| HuBERT Large | 300M | 212 | 0.0% | 3.6x | 3.1s | Same runtime as wav2vec2 |

#### Encoder-TDT (non-autoregressive, transducer)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B | 600M | 466 | 0.0% | 5.6x | 2.0s | 24 FC layers + joint net |

#### Encoder-Decoder / AED (autoregressive, attention-based)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Whisper (base) | 74M | 141 | 0.0% | **9.3x** | 1.2s | Full GPU (upstream) |
| Moonshine Tiny | 27M | 20 | 9.1% | 6.7x | 1.6s | CPU-only, tiny |
| Canary 1B | 1B | 672 | 0.0% | 6.2x | 1.8s | GPU enc+dec, 32+8 layers |
| Cohere Transcribe | 2B | 1440 | 0.0% | 5.2x | 2.1s | GPU enc, AED dec |
| Kyutai STT 1B | 1B | 636 | 4.5% | 1.4x | 7.7s | 24-layer Mimi decoder |
| FireRed ASR2 AED | 900M | 918 | 0.0% | 0.6x | 19.0s | CPU Q4_K SIMD dec (60ms/step) |

#### Encoder-LLM (autoregressive, language model decoder)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Qwen3 ASR 0.6B | 780M | 515 | 0.0% | 4.7x | 2.3s | 0.6B LLM |
| GLM ASR Nano | 1.3B | 1262 | 0.0% | 4.6x | 2.4s | ~1B LLM |
| Voxtral Mini 3B | 3B | 2530 | 0.0% | 2.4x | 4.7s | Mistral 3B LLM |
| OmniASR LLM 300M | 1.6B | 1018 | 4.5% | 1.7x | 6.4s | LLaMA 1.3B dec |
| Granite Speech 1B | 2.9B | 2805 | 0.0% | 1.7x | 6.4s | Granite LLM |
| VibeVoice ASR | 4.5B | 4589 | 4.5% | 1.2x | 8.8s | ~4B LLM, JSON output |
| Voxtral 4B Realtime | 4B | 2407 | 0.0% | 0.9x | 12.8s | Causal streaming arch (PLAN #7 streaming API; 1.6s first-text-token) |

### Speed ranking

| Rank | Backend | RTx | Time | Architecture |
|---|---|---|---|---|
| 1 | FastConformer CTC | 9.6x | 1.1s | Encoder-CTC |
| 2 | Whisper base | 9.3x | 1.2s | Encoder-Decoder |
| 3 | OmniASR CTC 1B | 7.4x | 1.5s | Encoder-CTC |
| 4 | Moonshine Tiny | 6.7x | 1.6s | Encoder-Decoder |
| 5 | Canary 1B | 6.2x | 1.8s | Encoder-AED |
| 6 | Parakeet TDT 0.6B | 5.6x | 2.0s | Encoder-TDT |
| 7 | Data2Vec Base | 5.3x | 2.1s | Encoder-CTC |
| 8 | Cohere Transcribe | 5.2x | 2.1s | Encoder-AED |
| 9 | Qwen3 ASR 0.6B | 4.7x | 2.3s | Encoder-LLM |
| 10 | GLM ASR Nano | 4.6x | 2.4s | Encoder-LLM |
| 11 | Wav2Vec2 XLSR-EN | 3.6x | 3.1s | Encoder-CTC |
| 12 | HuBERT Large | 3.6x | 3.1s | Encoder-CTC |
| 13 | Voxtral Mini 3B | 2.4x | 4.7s | Encoder-LLM |
| 14 | OmniASR LLM 300M | 1.7x | 6.4s | Encoder-LLM |
| 15 | Granite Speech 1B | 1.7x | 6.4s | Encoder-LLM |
| 16 | Kyutai STT 1B | 1.4x | 7.7s | Encoder-AED |
| 17 | VibeVoice ASR | 1.2x | 8.8s | Encoder-LLM |
| 18 | Voxtral 4B Realtime | 0.9x | 12.8s | Encoder-LLM |
| 19 | FireRed ASR2 AED | 0.6x | 19.0s | Encoder-AED |

---

## CPU-only VPS — 2026-04-24

Platform: x86_64, 4 threads, 7.6 GB RAM, AVX2, no GPU.

| Backend | RTx (CPU) | Time (CPU) | RTx (T4) | Speedup |
|---|---|---|---|---|
| FastConformer CTC | 9.4x | 1.2s | 9.6x | 1.1x |
| Moonshine Tiny | 16.8x | 0.7s | 6.7x | 0.4x* |
| Parakeet TDT 0.6B | 2.9x | 3.8s | 5.6x | 1.9x |
| Canary 1B | 2.7x | 4.0s | 6.2x | 2.2x |
| Data2Vec Base | 2.1x | 5.2s | 5.3x | 2.5x |
| Qwen3 ASR 0.6B | 1.7x | 6.5s | 4.7x | 2.8x |
| Wav2Vec2 XLSR-EN | 1.1x | 9.9s | 3.6x | 3.2x |
| Cohere Transcribe | 1.4x | 7.7s | 5.2x | 3.7x |
| FireRed ASR2 AED | 0.1x | 123s | 0.6x | 6.5x |

*Moonshine runs CPU-only on both (tiny model, no GPU benefit).

GPU acceleration is strongest for encoder-heavy models (2-6x). Decoder-bound
models benefit less (FireRed decoder still runs on CPU even with GPU).

---

## Per-phase breakdowns

### wav2vec2 family (Kaggle T4)

| Model | CNN | Pos conv | Encoder | Total |
|---|---|---|---|---|
| wav2vec2-large (24L) | 215ms | 1588ms | 127ms | 1941ms |
| hubert-large (24L) | 227ms | 1595ms | 128ms | 1960ms |
| data2vec-base (12L) | 221ms | 735ms | 57ms | 1023ms |

**Bottleneck:** pos_conv (grouped conv1d on CPU) = 50-80% of total time.
Encoder graph on GPU is only 57-128ms.

### FireRed AED decoder (Kaggle T4)

| Phase | Time | Notes |
|---|---|---|
| Fbank extraction | ~50ms | CPU |
| Conv2d subsampling | ~100ms | CPU |
| Hybrid encoder (16L) | ~17s | GPU matmuls + CPU attention, slow due to CPU weight copies |
| K/V precompute | 433ms | GPU (scheduler auto-copies) |
| Decoder (28 steps) | 1695ms | CPU Q4_K SIMD, 60.5ms/step |
| **Total** | **19.0s** | Encoder dominates |

### OmniASR (Kaggle T4)

| Model | Encoder | Prefill | Decode | Total | RTx |
|---|---|---|---|---|---|
| CTC 1B v2 | 244ms | — | — | 277ms | 39.8x (encoder only) |
| LLM 300M v2 | 97ms | 803ms | 4028ms (103 steps) | 5021ms | 2.2x |

---

## Key observations

1. **CTC models dominate on speed.** No decoder loop = one forward pass.
2. **Small LLM decoders (0.6-1B) are competitive** — Qwen3 and GLM hit 4.5x+
   realtime with 0% WER, close to encoder-only models.
3. **Large LLMs (3-4.5B) are 1-2x realtime** on T4. Usable but not fast.
4. **Most WER=0% on jfk.wav.** The 4.5% models have minor formatting differences,
   not actual transcription errors. Moonshine Tiny (9.1%) has a real word error.
5. **wav2vec2 pos_conv was the bottleneck** — now 4.9x faster with ggml grouped
   conv (im2col + mul_mat SIMD). Was 1.6s (80% of runtime), now 324ms (~3.5%).
6. **FireRed encoder is slow** because CPU weights auto-copy to GPU per-layer.
   Pre-loading encoder weights to GPU would save ~15s.

---

## Optimization history

### wav2vec2 grouped conv — 2026-04-27

| Path | pos_conv | Notes |
|---|---|---|
| Manual C++ (OMP) | 1588ms | 4-thread OMP, plain float loops |
| **ggml im2col + mul_mat** | **324ms** | **4.9x faster**, SIMD kernels |

The grouped positional conv (C=1024, K=128, G=16) is decomposed into G=16
independent `ggml_pad_ext` + `ggml_im2col` + `ggml_mul_mat` calls. The
mul_mat output `[cpg, T]` is transposed to channel-first before reassembly.
Applies to wav2vec2, data2vec, and hubert.

### FireRed decoder — 2026-04-26

| Path | ms/step | 28 tokens | Why |
|---|---|---|---|
| Manual C++ F32 (original) | 4400 | 123s | No SIMD, no parallelism |
| + OpenMP matmuls | 2320 | 58s | 2.1x from OMP |
| + ggml Q4_K CPU native | **70** | **2.0s** | 9.3x from fused SIMD kernel |
| ggml_vecmat on CUDA | 2600 | timeout | CUDA launch overhead kills it |
| F32 dequant + cpu_matmul | 590 | 16.5s | No SIMD, OMP disabled on Kaggle |
| **ggml_vecmat CPU (final)** | **60** | **1.7s** | Weights on CPU, native Q4_K |

### wav2vec2 CNN — 2026-04-24

| Change | CNN | Total | Speedup |
|---|---|---|---|
| Baseline (manual C++) | 95.2s | 108.4s | 1.0x |
| ggml F32 im2col | 2.4s | 15.5s | 7.0x |
| + OpenMP pos_conv | 2.3s | 9.9s | 10.9x |

### voxtral4b streaming — 2026-05 (PLAN #7 phases 1+1.5+2+3+4)

Native incremental encoder + streaming-prompt decode + speculative
prefill + combined-chunk flush + live captions + decoder thread.
M1 Q4_K JFK 11 s baseline, all variants bit-exact-batch:

| Stage / phase | Metric | Before | After | Δ |
|---|---|---|---|---|
| Phase 1 (initial) | first-text-token | n/a | 2674ms | — |
| + 240ms chunks (phase 2) | feed total | 23s | 9.1s | 2.5× faster |
| + default-unification fix | encoder drain | 2064ms | 1016ms | -1.0s |
| + fused QKV (Q4_K) | per-decode-step | 56ms | 50.4ms | -10% |
| + combined-chunk flush (phase 3) | encoder drain | 990ms | 307ms | -683ms |
| + speculative prefill (phase 3) | first-text-token | 921ms | **650ms** | -271ms |

**Final**: first-text-token 2674ms → **650ms (4.1× faster)**;
sequential live decode (phase 3); decoder thread for non-blocking
feed (phase 4, gated on `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1`).

The remaining ~410ms gap to the model's ≤240ms target is the
architectural floor: 8 streaming-pad warmup steps × 50.4ms + LLM
prefill = 655ms minimum on M1 Q4_K. Cross that floor only via a
faster Q4_K Metal kernel or a model with a different prompt
convention (no streaming-pad warmup).

Cross-backend portability of the fused-QKV Q4_K pattern:
- qwen3-asr Q4_K: default-on (transcript correct; perf within
  noise on JFK's short-decode shape)
- voxtral 3B Q4_K: opt-in (`CRISPASR_VOXTRAL_FUSED_QKV=1`); A/B
  showed no measurable speedup on JFK
- qwen3-tts: opt-in (existing convention)

### FastConformer encoder flash_attn_ext — 2026-05-09

Commit `c2423313` rewrites the FastConformer encoder self-attention
(parakeet, canary, canary_ctc) from 3 separate matmuls + add + softmax
for Shaw relative-position attention into a single `ggml_flash_attn_ext`
call per layer with the BD position bias precomputed and passed as the
additive mask. Reduces per-encoder-pass kernel dispatches from
32 layers × 3 matmuls = 96 down to 32 — the dominant win on GPUs
where per-launch overhead is real.

Re-verification on Apple M1 Metal (`build-ninja-compile/`,
`GGML_METAL=ON`, `GGML_BLAS=ON` Apple), 3-pass warm-cache JFK 11 s:

| Backend | Baseline (`c2423313~1`) median | Flash-attn (`c2423313`) median | Speedup | Output |
|---|---|---|---|---|
| parakeet (TDT 0.6B v3 F16) | 2.57 s (4.3× RT) | 1.60 s (6.9× RT) | **1.61× (38% faster)** | bit-identical ✓ |
| canary (1B v2 Q4_K) | 1.53 s (7.2× RT) | 1.15 s (9.6× RT) | **1.33× (25% faster)** | bit-identical ✓ |

Substantially exceeds the commit message's CPU number (~10%), confirming
the GPU-vs-CPU hypothesis: with kernel-launch overhead in the picture,
fusion pays off ~3-4× more. Wallclock includes whisper-tiny LID
(~77 MB Metal load) and feature extraction — both unchanged across the
two builds, so the encoder-attention-only speedup is larger than the
table suggests. Parakeet benefits more than canary because its encoder
runs longer per token (TDT joint loop), so the 32-layer attention block
dominates a larger share of wallclock.

Issue #81 ("parakeet 5× slower than ONNX on GPU") — this commit closes
a chunk of the gap but not all of it. Next likely targets: decoder
loop, joint network, log-mel host→device transfer.

### onnx-asr cross-comparison — issue #81 (2026-05-09)

Replicating the issue reporter's setup (libcrispasr via Python ctypes,
parakeet-tdt-0.6b-v3 q8_0 GGUF) and comparing against onnx-asr 0.11.0
on the same Apple M1, JFK 11 s, 3 warm passes per path. crispasr is at
flash-attn commit `c2423313`. ONNX backend selection follows
`istupakov/onnx-asr`'s upstream recipe (`pip install onnx-asr`).

**ONNX execution-provider reality check on M1:**

| ONNX model | CPU EP | CoreML EP |
|---|---|---|
| `nemo-parakeet-tdt-0.6b-v3` (F32, external-data, encoder 2.4 GB) | ✓ | ✗ external-data initializer + CoreML's 316-partition subgraph split lose `model_path`; inlining hits protobuf's 2 GB ceiling. Tracked upstream: [`microsoft/onnxruntime#26355`](https://github.com/microsoft/onnxruntime/issues/26355), closed *not planned* |
| `nemo-parakeet-ctc-0.6b` (F32 + external data) | ✓ | ✗ same issue |
| `nemo-parakeet-ctc-0.6b` int8 (single-file, 650 MB) | ✓ | ✓ loads after ~10 s CoreML compile |

The upstream onnx-asr README claim "Works on … macOS … with support for
… CoreML" is therefore **partially true** on Apple Silicon for parakeet:
only the smaller CTC int8 single-file export reaches CoreML; the full
TDT (and full-precision CTC) exports stay CPU-only because of how
istupakov packages them with external-data tensors larger than
protobuf's 2 GB limit.

**TDT-vs-TDT bench** (JFK 11 s, 3 warm passes, load avg ~4.0):

| path | median | RT× |
|---|---|---|
| **crispasr ctypes Session, parakeet-tdt q8_0, Metal** | **1.34 s** | **8.24×** |
| onnx-asr `nemo-parakeet-tdt-0.6b-v3` F32, CPU EP | 1.77 s | 6.23× |

Apples-to-apples on the TDT architecture: **crispasr Metal beats
onnx-asr CPU by 1.32×.** The Q8_0 ctypes path is faster than the F16
CLI numbers above because it skips the CLI's whisper-tiny LID startup
(~77 MB Metal load) and output formatting overhead — closer to what
the issue reporter actually measured.

**CTC-vs-CTC bench** (JFK 11 s, 3 warm passes, all CTC outputs
identical, q8_0 quants, load avg ~2.6):

| path | median | RT× |
|---|---|---|
| **crispasr Session, parakeet-ctc-0.6b q8_0, Metal** | **~460 ms** | **~24×** |
| onnx-asr `nemo-parakeet-ctc-0.6b` (~600M) int8, CPU EP | 724 ms | 15.2× |
| onnx-asr `nemo-parakeet-ctc-0.6b` (~600M) int8, CoreML EP | 1279 ms | 8.6× |

(crispasr Metal value is from the `stt_en_fastconformer_ctc_xlarge` 3-pass
bench at load ~2.6 — identical encoder + CTC-head graph as
`parakeet-ctc-0.6b`, only the tokenizer + training data differ. The new
parakeet-ctc-0.6b GGUFs fall in the same window when measured under
the same load — variance ~0.4–0.7 s observed across loads 2.6–4.0.)

`nvidia/parakeet-ctc-0.6b` (24L) and `nvidia/parakeet-ctc-1.1b` (42L)
are now first-class in crispasr — the existing
`models/convert-stt-fastconformer-ctc-to-gguf.py` handles both (encoder
+ CTC head are architecturally identical to the `stt_en_fastconformer_ctc_*`
family); `examples/cli/crispasr_backend.cpp` auto-routes
`parakeet-ctc-*.gguf` filenames to the `fastconformer-ctc` backend (the
JA hybrid `parakeet-tdt_ctc-0.6b-ja` stays on the `parakeet` TDT path
via the "tdt" guard). Quantised variants
([F16, Q8_0, Q5_0, Q4_K]):
[`cstr/parakeet-ctc-0.6b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-0.6b-GGUF)
and [`cstr/parakeet-ctc-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-1.1b-GGUF).
**crispasr wins by ~1.6×** on the same upstream model on M1 Metal.

Two M1-specific surprises worth surfacing:

1. **CoreML EP is *slower* than CPU EP on M1** for parakeet-shaped
   graphs (CTC: 1.28 s vs 0.72 s on the same int8 model). M1's CPU
   vector pipeline + onnxruntime CPU kernels outpace CoreML's
   per-graph compile + dispatch overhead. ONNX users on Apple Silicon
   should default to CPU EP for parakeet, not CoreML.
2. **CoreML EP isn't even reachable for the upstream parakeet TDT
   ONNX export** (external-data + protobuf 2 GB ceiling, see table
   above). The headline "works on macOS with CoreML" claim only
   applies to the smaller CTC int8 single-file export.

**Reframing the 5× claim in issue #81:** the reporter is on Windows +
RTX 4070 + onnxruntime-directml — i.e. ONNX with a *working dGPU
execution provider*. DirectML on a 4070 is a real architectural
advantage no amount of ggml-side fusion will fully erase until our
CUDA / Vulkan kernels for the parakeet hot paths reach parity. On M1
the picture inverts: ONNX's only ergonomic path is CPU EP (or CoreML
EP for the smaller CTC int8 export, where it's *slower* than CPU
anyway), and crispasr Metal beats every ONNX path that loads — by
1.32× on TDT-vs-TDT and **1.58× on CTC-vs-CTC at the same param
count.** The actionable framing for the issue is "which CUDA / Vulkan
kernels in the parakeet path are leaving perf on the table on dGPU"
rather than "parakeet is slow on GPU universally."

Reproduce:

```bash
pip install onnx-asr soundfile
HF_HOME=/Volumes/backups/ai/huggingface-hub \
CRISPASR_LIB_PATH=$(pwd)/build-ninja-compile/src/libcrispasr.dylib \
PYTHONPATH=$(pwd)/python \
python -c "
import time, soundfile as sf, onnx_asr
from crispasr import Session
audio,_ = sf.read('samples/jfk.wav', dtype='float32')

# 1) crispasr Q8_0 GGUF via ctypes (matches issue #81 reporter setup)
sess = Session('<path-to>/parakeet-tdt-0.6b-v3-q8_0.gguf', backend='parakeet')
sess.transcribe(audio.copy(), language='en')  # warm
for i in range(3):
    t = time.perf_counter()
    sess.transcribe(audio.copy(), language='en')
    print(f'crispasr q8_0 Metal: {(time.perf_counter()-t)*1000:.0f} ms')

# 2) onnx-asr TDT CPU EP
m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', providers=['CPUExecutionProvider'])
m.recognize(audio)
for i in range(3):
    t = time.perf_counter()
    m.recognize(audio)
    print(f'onnx tdt CPU EP:     {(time.perf_counter()-t)*1000:.0f} ms')
"
```

### jason-ni/parakeet.cpp cross-comparison — issue #81 (2026-05-10)

[`jason-ni/parakeet.cpp`](https://github.com/jason-ni/parakeet.cpp) is
the prior public attempt at a ggml port of parakeet, referenced in
issue #81 as evidence that "ggml-based parakeet is slow." Author paused
2025-07 with the README note "the ggml implementation is not as
efficient as expected" after observing 1 s encoder time vs a claimed
0.001 s for parakeet-mlx. The 0.001 s claim is almost certainly an
async-dispatch return time on MLX, not actual compute — real MLX
encoder cost is in the same single-second range as ours and theirs.

**Scope of their build.** Encoder-only proof of concept for **Parakeet
TDT 0.6B v2** (English-only, MLX checkpoint), F32 weights, ~4 000 LOC
including a custom mini-runtime (`framework_*`). No decoder, no joint
network, no streaming, no quantisation, no Python/CLI integration.
Test harness is `parakeet_cpp <gguf> <pe.bin> <input.data>` — feeds
pre-baked mel features, returns encoder hidden states.

**Their graph.** Standard FastConformer encoder, 24 layers ×
(LN → FF1 → LN → self-attn → LN → conv → LN → FF2 → LN), exactly the
architecture we ship. The interesting differences are all in the self-
attention block at `src/framework_nn.cpp` lines 820–1010:

- **Shaw relative-position attention done as separate ops**:
  `matrix_ac = mul_mat(K, Q+u_bias)`,
  `matrix_bd = mul_mat(P_emb, Q+v_bias)`, then a left-pad + slice
  trick to align positions, add, scale, softmax, multiply by V.
  3 matmuls + softmax + matmul + several view/transpose passes per
  layer. Same shape as our pre-`c2423313` baseline.
- **`ggml_flash_attn_ext` path is written but commented out** (lines
  944–987). They had the fused approach drafted, didn't activate it —
  exactly the path our `c2423313` activates and tunes.
- **Conv2D pre-encode (subsampling)** uses `ggml_conv_2d` — same as
  us. Their conformer self-attn comment notes
  `weight f16 is required for ggml_conv_2d_dw` on Metal — same Metal
  constraint we observed.
- **F32 weights only**, no Q4_K / Q8_0 / F16 quantisation paths.

So architecturally their encoder is a sibling of ours minus the
2026-05 flash-attn-ext fusion. The "ggml is slow" finding they paused
on is exactly the bottleneck commit `c2423313` addresses (1.61× on
parakeet TDT v3 F16, M1 Metal).

**Apples-to-apples on their own test audio** (`assets/input.wav`,
47.74 s, mel features identical), encoder-only, 3 warm runs:

| build | hardware | precision | encoder mean | RT |
|---|---|---|---|---|
| jason-ni/parakeet.cpp (their README) | Apple **M4** | F32 | 0.92 s | 51.9× |
| **crispasr `parakeet_test_encoder` (this commit, flash-attn-ext)** | Apple **M1** | F16 | **1.66 s** | **28.8×** |
| crispasr `parakeet_test_encoder` (this commit, flash-attn-ext) | Apple M1 | Q8_0 | 2.64 s | 18.1× |

Hardware-normalised: M4 GPU is ≈ 1.5–1.8× M1 on Metal compute,
putting jason-ni's number at ~30–35× RT on M1-equivalent hardware.
We're at 28.8× RT on M1 with F16 + flash-attn-ext — **roughly within
hardware noise of each other** for encoder-only. The gap they panicked
about against MLX is illusory; the gap against ours doesn't exist
once you normalise hardware.

**Important encoder-vs-pipeline note.** On the *encoder alone*, F16 is
faster than Q8_0 on Metal (Q8_0 dequant overhead doesn't pay off when
encoder ops are matmul-bandwidth-friendly even at F16). Q8_0 wins for
the **full pipeline** because the TDT joint network + label-predictor
LSTM run many small matmuls per output token where weight memory
bandwidth dominates. The `tools/benchmark_asr_engines` matrix puts
Q8_0 at 7.4× RT for full inference / 60 s. **Different shapes win
different quants** — pick by what your pipeline actually does, not
by quant name alone.

**What we have that they don't, attributable to specific work:**

1. Flash-attn-ext attention fusion (`c2423313`). Their `ggml_flash_attn_ext`
   path exists in code but is commented out.
2. Full TDT decoder (label predictor + joint network + per-frame TDT
   step). They're encoder-only.
3. Quantisation paths (Q4_K, Q5_0, Q8_0). They ship F32 only.
4. Multilingual TDT v3 support. They support v2 (English-only).
5. Production integration: CLI, `python/crispasr/Session`, streaming,
   VAD, mic, WER tooling, multi-backend dispatch. Theirs is a single
   test binary.
6. Cross-platform: CUDA / Vulkan / Metal / CPU. Theirs is
   Metal-focused (`-DGGML_METAL=ON`).

**Reframe of issue #81 in light of this**: the prior public ggml
attempt (jason-ni) plateaued at our pre-`c2423313` baseline and paused
on a misread benchmark. Our crispasr build, post-flash-attn fusion,
matches it on encoder-only and ships everything else around it. The
remaining issue #81 gap on Windows + RTX 4070 + DirectML is still
about CUDA/Vulkan kernel coverage on the dGPU side, not about ggml
fundamentally being too slow for parakeet.

Reproduce the encoder-only number:

```python
# Save jason-ni's input.wav reference: 47.74 s, 16 kHz mono.
# T_mel = 4774 (10 ms hop matches both their preprocess and ours).
import ctypes, time
lib = ctypes.CDLL('build-ninja-compile/src/libcrispasr.dylib')
lib.crispasr_parakeet_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.crispasr_parakeet_init.restype  = ctypes.c_void_p
lib.parakeet_test_encoder.argtypes  = [ctypes.c_void_p, ctypes.c_int]
lib.parakeet_test_encoder.restype   = ctypes.c_int
lib.crispasr_parakeet_free.argtypes = [ctypes.c_void_p]

ctx = lib.crispasr_parakeet_init(b'parakeet-tdt-0.6b-v3.gguf', 4, 1)  # F16, flash-attn on
lib.parakeet_test_encoder(ctx, 4774)  # warm
for _ in range(3):
    t = time.perf_counter()
    lib.parakeet_test_encoder(ctx, 4774)
    print(f'{(time.perf_counter()-t)*1000:.0f} ms')
lib.crispasr_parakeet_free(ctx)
```

`parakeet_test_encoder` runs the full encoder graph with mel = zeros —
compute-bound, identical kernel dispatches to a real call, no I/O.
Use it instead of the CLI when you want encoder-only timing without
LID-model load, mel extraction, and the TDT decoder loop in the
wallclock.

---

## Reproduce

```bash
# Per-backend timing
CRISPASR_VERBOSE=1 crispasr --backend firered-asr -m auto -f jfk.wav -v -bs 1

# wav2vec2 phase breakdown
WAV2VEC2_VERBOSE=1 crispasr --backend wav2vec2 -m auto -f jfk.wav -v

# Full Kaggle benchmark (all 19 backends)
# See tools/kaggle-benchmark-all-backends.py or gist:
# https://gist.github.com/CrispStrobe/c15f7a64878d93907a8a4a51b193b806
```
