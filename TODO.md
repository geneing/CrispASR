# CrispASR — TODO

Live tracker of pending work across the unified `crispasr` binary and the
shared `src/core/` infrastructure. Items marked **[next]** are the current
session's immediate targets; **[later]** are queued; **[upstream]** are
blocked on external fixes (tracked in detail in `UPSTREAM.md`).

Historical milestones and the per-model port plans are in `HISTORY.md`.
Technical deep-dives (optimisation notes, RoPE lessons, benchmark tables)
are in `LEARNINGS.md`.

---

## Near-term — `src/core/` Phase 0

Extraction is ~90% done for mel + ffn + gguf_loader, and attention has a
minimal pilot. The remaining pieces are documented below.

- **[next]** **`src/core/attention.h` — persistent-KV-cache variant.**
  The current `core_attn::llama_self_attn` fits voxtral 3B, which rebuilds
  Q/K/V for the full context on every forward pass. qwen3, voxtral4b, and
  granite LLM blocks use a persistent backend-buffer KV cache: K/V are
  written to a pre-allocated view at `n_past` and read back through a
  contiguous view for attention. Needs a sibling helper, e.g.
  `llama_self_attn_kv(…, kv_k, kv_v, n_past, …)`.
  Pilot on qwen3 LLM, then voxtral4b, then granite LLM. ~100 LOC helper +
  3 migrations, ~30-60 LOC saved per block.

- **[next]** **`src/core/attention.h` — Q/K norm variant for qwen3.**
  Qwen3 applies a post-projection RMSNorm to Q and K before RoPE. Add
  `q_norm_w`, `k_norm_w` optional pointers to the helper (or a separate
  variant) so qwen3's audio encoder + LLM can both use it.

- **[later]** **`src/core/attention.h` — voxtral audio encoder.**
  Different flavour: Q/V biases, **no** K bias (Whisper quirk), no RoPE.
  ~30-line sibling helper.

- **[later]** **`src/core/attention.h` — sliding-window attention.**
  voxtral4b audio encoder uses 750-token SWA. Needs a `sliding_window`
  knob and the mask has to be pre-built by the caller (or constructed
  by the helper from `sliding_window`).

- **[later]** **`src/core/attention.h` — µP scale tricks.**
  Granite uses `attention_multiplier` (0.0078125 = 1/128) as the attention
  scale instead of `1/sqrt(d)` and `residual_multiplier` (0.22) on the
  residual add. Parameterise via `Config::attn_scale` and
  `Config::residual_scale` in the helper.

- **[next]** **`src/core/greedy_decode.h` — unified LLM decode loop.**
  The voxtral/voxtral4b/qwen3/granite backends all duplicate the same
  `greedy decode from last-token logits → embed → forward → argmax → append`
  loop with minor variations in EOS handling and max-tokens. Should
  replace the CLI-level `examples/cli/crispasr_llm_pipeline.h` template
  so the decode loop lives next to the model code. ~80 LOC helper + 4
  migrations, ~150 LOC saved across the models.

- **[next]** **`src/core/mel::Params::stacked_frames`.**
  Granite's mel output is stacked `(160, T/2)` = 2 × 80 mels per frame.
  Add a `stacked_frames` knob to `Params` (default 1; when 2, post-
  process the TimeMels output by zipping consecutive frames together
  and dropping any trailing odd frame). Last holdout on `core_mel::`
  migration.

- **[later]** **`cli.cpp` output writer refactor (task #4).**
  `output_json` (282 lines) and `output_wts` (120 lines) in cli.cpp
  still iterate a `whisper_context *` directly. Refactor to consume
  `const std::vector<crispasr_segment> &` so the whisper backend can
  also go through the unified writers. Unblocks the next item.

- **[later]** **`backend-whisper.cpp` wrapper (task #15).**
  Gated on the output writer refactor. When both land, the whisper code
  path in cli.cpp can dispatch through the same backend factory as
  everything else, and the `#if 0`-guarded old `whisper_params` block
  can come out.

---

## CLI + examples cleanup

- **[next]** Delete the per-model `examples/*-main/` directories once
  `crispasr --backend X` has shipped and regression-tested in CI.
  Candidates: `parakeet-main`, `canary-main`, `cohere-main`,
  `qwen3-asr-main`, `voxtral-main`, `voxtral4b-main`, `granite-main`,
  `nfa-align`, `cohere-align`. Update `examples/CMakeLists.txt`.
  Add deprecation-warning stubs under `examples/deprecation-warning/`
  for one release cycle before full removal.

- **[later]** `tests/CMakeLists.txt` uses `whisper-cli` as the test
  target. Keep that target name (we already preserve it) but move the
  tests over to `$<TARGET_FILE:crispasr>` once the rename has propagated.

---

## Feature parity gaps (non-whisper backends vs whisper)

The whisper backend in CrispASR is the most feature-complete. The
capability matrix in the README shows which features are missing on
each backend. High-value gaps to close:

- **[later]** **Temperature / beam search** — no non-whisper backend
  currently exposes sampling controls. `voxtral`, `voxtral4b`, `qwen3`,
  `granite` all run pure greedy decode. Hook the sampler into the
  shared `core/greedy_decode.h` helper when it lands.

- **[later]** **VAD integration in LLM backends.** qwen3 and voxtral
  currently don't chunk long audio; the dispatch layer does VAD slicing
  but the LLM models themselves pad to a fixed 30s window. Variable-
  length mel would let them handle >30s natively.

- **[later]** **Streaming transcription for voxtral4b.** The model is
  designed for realtime streaming with configurable 240ms-2.4s delay.
  Currently we run it in batch mode like the others. Exposing a
  streaming mode through the CLI is a bigger design question.

- **[later]** **Audio understanding mode for voxtral 3B.** The model
  supports Q&A over audio content, not just transcription. Needs a
  prompt template flag and a chat-style turn loop. Separate feature,
  not a strict regression.

---

## Per-model follow-ups

### parakeet
- **[later]** Port the TDT decoder (LSTM predictor + joint head) to
  ggml graphs so it can run on GPU. Currently pure CPU float* loops.
  Risk: per-token LSTM stepping is sequential, so GPU speedup may be
  small. Encoder is already the dominant cost.

### canary
- **[later]** Speech translation quality validation at scale.
  Currently regression-tested on German only.

### cohere
- **[later]** F32→F16 self-attention KV cache upgrade. Currently uses
  F32 where other models use F16, wasting 2× GPU memory bandwidth.
  ~30 LOC, low risk.

### qwen3 / voxtral
- **[later]** Stop recreating `ggml_backend_sched` on every compute
  call (encoder, prefill, each decode step). Create once at init with
  worst-case node budget; use `ggml_backend_sched_reset()` between
  calls. ~80 LOC per runtime.

### voxtral4b
- **[later]** Reduce right padding from 17 → 10 tokens to match the
  reference `voxtral.c` implementation.
- **[later]** SRT/VTT subtitle output (currently only plain transcript;
  CTC alignment already works via `-am`).

### granite
- **[later]** HF release of quantised GGUFs (`cstr/granite-speech-4.0-1b-GGUF`
  is still pending). Need `cohere-quantize granite-speech-1b.gguf …`
  then upload.
- **[later]** Performance tuning (encoder Conformer is slow per-layer CPU).
  Consider porting to a single ggml graph like canary did.
- **[later]** Remove dead ggml graph encoder `granite_build_encoder`.
- **[later]** Migrate mel to `core_mel::compute` once `stacked_frames`
  lands.

### canary_ctc (aligner)
- **[later]** Fix single-backend scheduler — currently no CPU fallback
  if the primary backend rejects an op. Match the 2-backend pattern
  from canary.cpp / cohere.cpp. ~20 LOC.

---

## Markdown cleanup (this session)

Consolidating ~15 historical notes into three live docs:

- `TODO.md` — this file (replaces all `*-todo.md`)
- `LEARNINGS.md` — technical insights, benchmarks, comparisons
- `HISTORY.md` — condensed chronology of the ports

Remove after consolidation: `canary-todo.md`, `parakeet-todo.md`,
`granite-todo.md`, `voxtral-todo.md`, `voxtral-4b-todo.md`,
`qwen3-asr-todo.md`, `TODO_COHERE_OPTIMIZATION.md`,
`benchmark_cohere.md`, `qwen3-asr-benchmark.md`, `ggml_plans.md`,
`voxtral-comparison.md`, `test_german.md`, `PERFORMANCE.md`.

Keep: `README.md`, `TODO.md`, `LEARNINGS.md`, `HISTORY.md`, `UPSTREAM.md`,
`README_sycl.md`, `ci/README.md`, `models/README.md`, `samples/README.md`,
`hf_readmes/*.md`.

---

## Upstream dependencies

Full tracking is in `UPSTREAM.md`. Short summary:

- **[upstream]** whisper.cpp `examples/ffmpeg-transcode.cpp` mp4-family
  container crash. Workaround: pre-convert with ffmpeg one-liner.
- **[upstream]** ggml x86 AVX-VNNI / AVX512-VNNI dispatch for Q8_0 dot
  products. Closes the 5-second gap to ONNX INT8 on x86 servers.
- **[upstream]** NeMo Forced Aligner auxiliary CTC model standalone
  release. Not blocking — our converter extracts it from the `.nemo` tarball.

---

## HF releases

| Repo | Status |
| --- | --- |
| `cstr/parakeet-tdt-0.6b-v3-GGUF` | ✅ shipped |
| `cstr/parakeet_de_med-GGUF` | ✅ shipped |
| `cstr/canary-1b-v2-GGUF` | ✅ shipped |
| `cstr/canary-ctc-aligner-GGUF` | ✅ shipped |
| `cstr/cohere-transcribe-03-2026-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ✅ shipped (Q4_K + Q8_0) |
| `cstr/granite-speech-4.0-1b-GGUF` | ❌ pending quantize + upload |
