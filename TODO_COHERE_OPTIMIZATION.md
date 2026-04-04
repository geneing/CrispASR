# Cohere Transcribe Optimization Plan

This document tracks the progress of porting `cohere-whisper.cpp` to a full `ggml` compute graph (Priority 2B).

## Progress Tracker

- [x] **Phase 0: Infrastructure Refactoring**
    - [x] Update `cohere_model` to manage `ggml_backend_buffer` and `ggml_context` for weights properly.
    - [x] Update `cohere_context` to include `ggml_backend`, `ggml_backend_sched`, and memory allocators.
    - [x] Implement robust GGUF tensor loading into the backend buffers.
- [x] **Phase 1: Decoder Graph Port**
    - [x] Implement `cohere_build_graph_decoder`.
    - [x] Support KV caching within the graph.
    - [x] Resolve `GGML_ASSERT(ggml_can_repeat(b, a))` broadcasting issues.
    - [x] Fix `ggml_mul_mat` dimension mismatches and transpose assertions.
    - [x] Verify numerical consistency and successful iterative decoding.
- [ ] **Phase 2: Encoder Graph Port**
    - [ ] Implement `cohere_build_graph_encoder` (48 Conformer layers).
    - [ ] Port Conv2D subsampling to `ggml_conv_2d`.
    - [ ] Implement Conformer relative-position attention (relative shift) in `ggml`.
    - [ ] Implement Conformer convolution module.
- [ ] **Phase 3: Integration & Cleanup**
    - [ ] Remove `cblas_sgemm` and `ct_linear` dependencies.
    - [ ] Remove manual F32 weight caching.
    - [ ] Enable F16 weight support natively in `ggml_mul_mat`.
- [ ] **Phase 4: Advanced Features**
    - [ ] Quantization tool (Q8_0, Q4_K).
    - [ ] GPU Backend support (CUDA/Metal).

## Current Status
- Decoder: **Graph implementation functional and verified**.
- Encoder: **Imperative implementation functional (26x speedup)**. Porting to graph in progress.
