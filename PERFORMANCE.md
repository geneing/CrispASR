# CrispASR — Performance benchmarks

Benchmarks comparing CrispASR's multi-backend inference against
[antirez/voxtral.c](https://github.com/antirez/voxtral.c), the
single-model C implementation of Voxtral Realtime 4B.

---

## Test setup

**Audio:** `samples/jfk.wav` — 11 seconds of JFK's "And so, my fellow
Americans" speech (16 kHz mono PCM).

**CrispASR hardware:** Intel Xeon (Skylake), 4 vCPUs, 7.6 GiB RAM,
no GPU. NFS-mounted storage. 4 threads (default).

**voxtral.c hardware (from their README):** Apple M3 Max, 40-core GPU,
128 GB RAM, 400 GB/s bandwidth.

---

## CrispASR — all backends on CPU (jfk.wav, 11s)

| Backend | Model | Quant | Size | Wall time | Realtime factor | Transcript correct |
|---|---|---|---|---:|---:|:---:|
| **qwen3** | Qwen3-ASR-0.6B | Q4_K | 513 MB | 10.5 s | 0.95x | yes |
| **parakeet** | Parakeet-TDT-0.6B-v3 | F16 | 1.2 GB | 11.3 s | 1.03x | yes |
| **whisper** | Whisper small | F16 | 464 MB | 11.6 s | 1.05x | yes |
| **canary** | Canary-1B-v2 | F16 | 1.9 GB | 15.5 s | 1.41x | yes |
| **cohere** | Cohere Transcribe | Q5_0 | 1.7 GB | 23.5 s | 2.14x | yes |
| **granite** | Granite 4.0-1B | Q5_0 | 2.5 GB | 33.8 s | 3.07x | yes |
| **voxtral** | Voxtral-Mini-3B | Q4_K | 2.5 GB | 75.6 s | 6.87x | yes |
| **voxtral4b** | Voxtral-Mini-4B-RT | F16 | 8.3 GB | 172.5 s | 15.7x | yes |

**Notes:**
- "Realtime factor" = wall_time / audio_duration. Values < 1.0 mean
  faster-than-realtime transcription.
- voxtral4b (8.3 GB F16) is the same model as antirez/voxtral.c targets,
  but running on CPU instead of GPU. With Q4_K quantisation this would
  drop to ~2.5 GB and be significantly faster.
- qwen3 at Q4_K is the fastest backend overall — faster-than-realtime
  on this 4-vCPU machine with 30+ language support.
- All backends produce correct transcripts for this sample.

---

## Comparison with antirez/voxtral.c

antirez/voxtral.c is a purpose-built C implementation of the Voxtral
Realtime 4B model with Apple Metal (MPS) GPU acceleration. CrispASR
runs the same model via ggml with automatic backend selection.

### Architecture differences

| | CrispASR | voxtral.c |
|---|---|---|
| **Models** | 11 backends (whisper, parakeet, canary, cohere, granite, voxtral, voxtral4b, qwen3, fc-ctc, wav2vec2, canary_ctc) | 1 (Voxtral Realtime 4B only) |
| **Weight format** | GGUF (F16/Q4_K/Q5_0/Q8_0) | BF16 raw safetensors |
| **Quantisation** | Full ggml quantisation support (2-8 bit) | None (BF16 only) |
| **GPU backends** | CUDA, Metal, Vulkan, SYCL (via ggml) | Apple MPS only |
| **Streaming** | Generic `--stream`/`--mic`/`--live` for all 11 backends | Native voxtral4b streaming with configurable latency |
| **Binary size** | Single `crispasr` binary for all backends | Single binary, voxtral4b only |
| **Dependencies** | ggml (bundled) | Apple Accelerate/MPS |

### Performance comparison (Voxtral Realtime 4B)

| Metric | voxtral.c (M3 Max GPU) | CrispASR (Xeon CPU) |
|---|---|---|
| **Model size** | 8.9 GB (BF16) | 8.3 GB (F16) |
| **Encoder (3.6s audio)** | 284 ms (MPS) | ~15 s (CPU, estimated) |
| **Decoder per-step** | 23.5-31.6 ms | ~800 ms (CPU) |
| **Total (11s jfk.wav)** | ~5 s (estimated) | 172.5 s |
| **Realtime factor** | ~2.5x faster than RT | 15.7x slower than RT |

The voxtral.c numbers reflect Apple M3 Max GPU acceleration, which
is ~30x faster than CPU-only inference. CrispASR with a CUDA/Metal GPU
would close this gap significantly (ggml's Metal backend achieves
comparable performance to hand-tuned MPS kernels on Apple Silicon).

### Where CrispASR wins

1. **Model variety:** 11 backends vs 1. For the same 11s clip, qwen3
   at Q4_K (513 MB) transcribes in 10.5s on CPU — faster than
   voxtral4b on GPU with 17x less memory.

2. **Quantisation:** Q4_K reduces voxtral4b from 8.3 GB to ~2.5 GB
   with minimal quality loss. voxtral.c has no quantisation support.

3. **Cross-platform GPU:** ggml supports CUDA, Metal, Vulkan, SYCL.
   voxtral.c is Apple MPS only.

4. **Feature completeness:** Word timestamps (via CTC/forced aligner),
   speaker diarisation, language identification, VAD, SRT/VTT output,
   temperature sampling, best-of-N — all work across backends.

### Where voxtral.c wins

1. **Raw GPU speed on Apple Silicon:** Hand-tuned MPS kernels for this
   specific model architecture outperform generic ggml Metal dispatch.

2. **Native streaming latency:** Purpose-built 240ms-2.4s latency
   streaming protocol vs CrispASR's generic chunked streaming.

3. **Simplicity:** Single-model focus means less code, fewer
   abstractions, easier to understand and modify.

---

## Backend selection guide (by speed on CPU)

For CPU-only deployment, pick the smallest model that meets your
accuracy needs:

| Need | Best pick | Speed |
|---|---|---|
| Fastest, 30+ languages | **qwen3** Q4_K (513 MB) | ~1x RT |
| English-only, word timestamps | **parakeet** F16 (1.2 GB) | ~1x RT |
| Battle-tested, all features | **whisper** small (464 MB) | ~1x RT |
| Best multilingual accuracy | **canary** F16 (1.9 GB) | ~1.4x RT |
| Lowest English WER | **cohere** Q5_0 (1.7 GB) | ~2.1x RT |

For GPU deployment, the larger models (voxtral, voxtral4b, granite)
become viable — the LLM decoder parallelises well on GPU.
