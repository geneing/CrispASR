# Canary 1B v2 — ggml port plan

Goal: a `canary-main` CLI that runs `nvidia/canary-1b-v2` on CPU via ggml,
with explicit `-sl SRC` / `-tl TGT` language flags for both ASR and speech
translation, plus native word-level timestamps from the model's auxiliary
CTC head.

## Why

- **Fixes the parakeet language-ID problem.** Canary takes `source_lang`
  as an explicit task token in the decoder prompt — no auto-detect, no
  ambiguity. Tested on the German clips that broke parakeet.
- **Adds speech translation** (24 languages → English, English → 24
  languages). The only translation runtime in this repo.
- **978M params**, between parakeet (600M) and cohere (2B). Will land
  at ~600 MB Q4_K.
- **CC-BY-4.0** licence.
- **Native word + segment timestamps** built in (the model card lists
  them as supported).

## Implementation effort

The work is much smaller than parakeet because **we already have both
halves of Canary's architecture**:

| Canary component | Already in this repo |
| --- | --- |
| FastConformer encoder (32 layers) | `parakeet.cpp` (24-layer version) |
| Conv2d dw_striding subsampling | `parakeet.cpp` and `cohere.cpp` |
| Rel-pos attention with Transformer-XL biases | `parakeet.cpp` and `cohere.cpp` |
| Mel preprocessor (128 mel, 16 kHz) | `parakeet.cpp` (identical config) |
| Transformer decoder (8 layers) | `cohere.cpp` |
| Cross-attention KV cache | `cohere.cpp` |
| 16384-token SentencePiece | `cohere.cpp` |
| Task token prompt prefix | `cohere.cpp` |

Canary is essentially **parakeet's encoder + cohere's decoder + a
different prompt prefix**.

## Architecture summary (from model card)

```
audio
  ↓ NeMo mel preprocessor (128 mels, 16 kHz, n_fft=512, hop=160, Hann)
  ↓ Conv2d dw_striding subsampling (8× temporal)
  ↓ 32× FastConformer block:
       FFN1 (Macaron, ½ scale)
       MHA  (rel-pos with Transformer-XL untied biases)
       Conv (pw1 + GLU + dw_k=9 + BN + swish + pw2)
       FFN2 (Macaron, ½ scale)
       LN_out
  → encoder_out [T_enc, d_enc]
  → cross-KV pre-computed for all 8 decoder layers (cohere-style)
                                                   ↓
                              ┌── 8× Transformer decoder block ──┐
                              │     SA(causal) → CrossAttn        │
                              │     → FFN → LN                    │
                              └────────────┬──────────────────────┘
                                           ↓
                                   linear → 16384-class logits

Decoder prompt: <|startoftranscript|><|src_lang|><|target_lang|><|task|>...
Task tokens drive ASR vs AST and source/target language selection.
```

## Tasks

### 1. Inspect .nemo (1 day)

- [ ] Download `nvidia/canary-1b-v2`
- [ ] Unpack tarball, dump `model_config.yaml` and the keys of
      `model_weights.ckpt`
- [ ] Confirm encoder is 32-layer FastConformer with the same hyperparam
      shape as parakeet (d_enc, n_heads, head_dim, FFN, conv_k)
- [ ] Confirm decoder is 8-layer Transformer (d_dec, n_heads, head_dim,
      FFN, max_ctx)
- [ ] Document tensor naming conventions (NeMo uses
      `encoder.encoder.0.norm_self_att.weight` etc.)
- [ ] Identify the task tokens: `<|en|>`, `<|de|>`, `<|transcribe|>`,
      `<|translate|>` (or whatever Canary uses) and their token ids

### 2. .nemo → GGUF conversion script (2-3 days)

- [ ] `models/convert-canary-to-gguf.py` adapted from
      `convert-parakeet-to-gguf.py`
- [ ] Map encoder tensors with the same naming we used for parakeet
      (`encoder.layers.{i}.*`, `encoder.pre.*`)
- [ ] Map decoder tensors using cohere's naming
      (`decoder.layers.{i}.*` with self-attn / cross-attn / FFN slots)
- [ ] Bake mel filterbank + Hann window from `preprocessor.featurizer.*`
- [ ] Embed SentencePiece vocab as `tokenizer.ggml.tokens`
- [ ] Add a synthetic zero `conv.dw.bias` per encoder layer (BN fold target)
- [ ] Write all `canary.*` hparams (32 enc layers, 8 dec layers, etc.)
- [ ] Test loading: open the GGUF, verify all tensors bind

### 3. src/canary.{h,cpp} loader (2 days)

- [ ] Public C API mirroring cohere.h:
      `canary_init_from_file`, `canary_free`,
      `canary_transcribe_ex(samples, n, source_lang, target_lang, t_offset_cs)`,
      `canary_result` with words + tokens
- [ ] Loader: open GGUF, read hparams, mmap tensors, bind into per-layer
      structs (`canary_pre_encode`, `canary_enc_layer`, `canary_dec_layer`)
- [ ] BatchNorm folding (steal `parakeet_fold_batchnorm`)
- [ ] Backend selection (Metal / CUDA / CPU like parakeet + cohere)

### 4. Encoder forward pass (1 day)

- [ ] Mostly copy `parakeet_build_graph_encoder` — same Conformer block,
      same dw_striding subsampling, same rel-pos shift trick
- [ ] Adjust for 32 layers vs 24
- [ ] Compute cross-attention K/V for all 8 decoder layers up front
      (cohere pattern — avoids re-encoding per decoder step)
- [ ] Smoke test: zero mel of 100 frames → expected T_enc, finite values

### 5. Decoder forward pass (3 days)

- [ ] Steal cohere's decoder graph builder. The decoder block is:
        SA(causal) + KV cache → CrossAttn → FFN → LN
- [ ] Per-step graph build (one token at a time, autoregressive)
- [ ] Self-attention KV cache stored on backend buffer
- [ ] Cross-attention reads pre-computed encoder K/V
- [ ] Output logits over the 16384-class vocab

### 6. Task token prompt + greedy decode loop (2 days)

- [ ] Build the decoder prompt prefix from `(source_lang, target_lang)`:
      e.g. `<|startoftranscript|><|de|><|en|><|translate|><|notimestamp|>`
      (exact tokens depend on what Canary uses — confirmed from `tokenizer.json`)
- [ ] Greedy decode loop: feed prompt, then sample tokens one at a time
      until `<|endoftext|>` or max_tokens
- [ ] Handle EOS, sentence breaks, punctuation
- [ ] Detokenise via SentencePiece (`▁` → space)

### 7. Word and segment timestamps (2 days)

- [ ] Approach A: cohere-style cross-attention DTW (we know this hits
      ~360 ms MAE, but it's free since we already have the helper)
- [ ] Approach B: if Canary's `.nemo` ships an auxiliary CTC head
      (the model card mentions a `timestamps_asr_model` add-on), wire
      that in and run forced alignment exactly like `cohere-align`.
      That would give 30-50 ms accuracy.
- [ ] Group sub-word tokens into words at SentencePiece boundaries
      (steal parakeet's grouper)

### 8. CLI tool `canary-main` (2 days)

- [ ] `examples/canary-main/main.cpp` mirroring `parakeet-main`
- [ ] New flags: `-sl LANG` (source), `-tl LANG` (target). When
      `sl == tl`, it's ASR; when they differ, it's translation.
- [ ] All the existing chunking / VAD / SRT / VTT / TXT plumbing
- [ ] List supported languages on `--help`

### 9. CMake wiring (1 day)

- [ ] `add_library(canary ...)` in `src/CMakeLists.txt`
- [ ] Same Metal / CUDA conditional setup as cohere/parakeet
- [ ] `examples/canary-main/CMakeLists.txt`
- [ ] Register in `examples/CMakeLists.txt`

### 10. Testing (2-3 days)

- [ ] `samples/jfk.wav` (English ASR sanity)
- [ ] The German clips that broke parakeet (`jazeschann`, `sarma`) —
      must produce coherent German with `-sl de -tl de`
- [ ] Translation: same German clips with `-sl de -tl en`
- [ ] Quantise to Q4_K, Q5_0, Q8_0 with `cohere-quantize`
- [ ] Benchmark vs cohere and parakeet on the voxpopuli demo clip

### 11. Documentation + HF release (2 days)

- [ ] Update README.md with a Canary section + comparison table
- [ ] HF model card at `cstr/canary-1b-v2-GGUF`
- [ ] Upload F16, Q8_0, Q5_0, Q4_K
- [ ] Update `benchmark_cohere.md` with canary rows

## Total estimate: ~2 weeks

For a working CPU runtime with explicit language control, ASR + AST, and
word/segment timestamps. Roughly **half** the parakeet effort because
~80 % of the code is already written.

## Status update (after v1 ship)

All v1 tasks above are complete and shipping. End-to-end working:
  - JFK English ASR → "And so, my fellow Americans, ask not what your country..."
  - Sarma German ASR → clean German
  - Sarma DE→EN translation → fluent English at 2× realtime CPU
  - DTW word timestamps via cross-attention (cohere-style)
  - VAD + chunking + SRT/VTT/TXT
  - Q4_K/Q5_0/Q8_0 + F16 on HuggingFace at cstr/canary-1b-v2-GGUF

## v1.1 — auxiliary CTC forced alignment for ~30-50 ms word stamps

The Canary `.nemo` ships an extra 600M-parameter Parakeet CTC model
(`timestamps_asr_model_weights.ckpt` + `timestamps_asr_model_config.yaml`)
that NeMo Forced Aligner uses for word-level timestamps. Per the Canary
paper, this is the *intended* timestamp path — cross-attention DTW (which
we currently use) tops out around 360 ms MAE, while CTC forced alignment
gives 30-50 ms.

The good news: it's a parakeet-style 24-layer FastConformer with the
same vocab as canary's main model, and we already have all the pieces:

  - `models/convert-parakeet-to-gguf.py` for the encoder (same shape as
    parakeet-tdt-0.6b-v3, identical 725 tensor count)
  - `src/parakeet.cpp` encoder forward pass (works as-is)
  - `src/align.cpp` Viterbi forced alignment (already used by cohere-align
    for the same kind of task on wav2vec2)

The work needed:

- [ ] `models/convert-canary-timestamps-to-gguf.py` — extract the
      auxiliary CTC model from inside the canary `.nemo` tarball, repackage
      as a parakeet-style GGUF but write the CTC head weights instead of
      the TDT decoder + joint
- [ ] Add a `parakeet_compute_logits()` C API entry that returns raw
      `[T_enc, vocab_size]` CTC logits (currently parakeet's encoder forward
      is internal-only)
- [ ] `canary-main` flag `-ts-model FNAME` that loads the aux CTC GGUF,
      runs it on the same audio, and uses `ctc_forced_align()` (from
      `src/align.cpp`) to map canary's transcript text to encoder frames
- [ ] Patch the canary_result word timings with the CTC stamps

Estimated effort: ~600 LOC, mostly plumbing. Most of the heavy lifting
is already done in cohere-align.

## v1.2 — per-language WER benchmark on FLEURS

- [ ] Small eval harness that downloads FLEURS, runs canary on each
      language, computes WER, and reports a per-language table.
- [ ] Compare against the per-language WER table in the Canary paper
      (Appendix B) to validate the C++ port matches the Python reference.

## Out of scope for v1

- Streaming / chunked inference (matches parakeet's v1 stance)
- Beam search (greedy is fine for canary like it is for cohere)
- Speaker diarisation (no model support)
- The forced-alignment CTC head (defer until approach A is in and we
  measure how bad cross-attn DTW is on canary's specific decoder)

## Reference implementations to crib from

- **`istupakov/onnx-asr`** — Python ONNX inference for canary, including
  the task token prompt format and the greedy loop. Best reference for
  the prompt prefix exact tokens.
- **NeMo `nemo/collections/asr/models/canary_models.py`** — ground truth
  for the `transcribe()` API and how `source_lang` / `target_lang`
  feed into the decoder.
- **`parakeet.cpp`** in this repo — encoder + mel + Conformer block.
- **`cohere.cpp`** in this repo — decoder + cross-KV pre-compute +
  task-token prompt prefix.
