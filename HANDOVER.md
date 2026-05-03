# CrispASR — Handover prompt for next session

**Date:** 2026-05-03
**Last session:** 2026-05-02/03 (2-day session, ~50 commits)
**Repo:** https://github.com/CrispStrobe/CrispASR
**Branch:** `main`
**Version:** v0.5.5 (released, 8 binary packages)

## What was done

This session closed 5 PLAN items (#11, #41, #60, #62, #63), made major
progress on #59 (binding parity), fixed 4 GitHub issues (#48, #49, #51,
#31), shipped 6 CI cross-platform fixes, added beam search to 2 backends,
flash attention to 6, auto-punc for CTC, 17 new unit tests, and a
WebSocket streaming server.

Full write-up: HISTORY.md §72-§76.

## What's open

### Highest priority — #57 Phase 3: Chatterbox CFM solver

**The single biggest unfinished item.** Builds a flow-matching ODE solver
in ggml, which unlocks Chatterbox (MIT) + Kartoffelbox_Turbo (CC-BY-4.0,
German) + lahgtna-chatterbox (MIT, Arabic).

**Architecture to implement:**

ResembleAI/chatterbox pipeline:
1. **BPE tokenizer** — `tokenizer.json`, already have `core/bpe.h`
2. **T3 (0.5B Llama AR)** — `t3_cfg.safetensors`, reuse orpheus Llama AR decode
3. **S3Gen (CFM denoiser)** ← THE NEW PIECE — `s3gen.safetensors`, CosyVoice-style
   conditional flow matching. ~12 Euler/midpoint ODE steps. Each step is one
   forward pass of the denoising network `v_θ(x_t, t, conditioning)`.
4. **HiFT-GAN vocoder** — mel → waveform. Similar to existing codec decoders.
5. **Voice encoder** — `ve.safetensors`, clone conditioning. Similar to ECAPA.

**What to do:**
1. Read Chatterbox Python source: https://github.com/ResembleAI/chatterbox
   Focus on `chatterbox/models/s3gen/` for the denoising network arch.
2. Inspect `s3gen.safetensors` weight shapes to map layers.
3. Write `tools/convert-chatterbox-to-gguf.py` — convert T3 + S3Gen + vocoder + VE.
4. Implement `src/chatterbox.{h,cpp}`:
   - T3 forward (Llama AR, reuse `core_attn::kv_self_attn` + `core_ffn::swiglu`)
   - S3Gen forward (denoising network graph) — likely a UNet or DiT variant
   - ODE loop: `x_{t+dt} = x_t + dt * v_θ(x_t, t, cond)` for t in [0, 1]
   - HiFT-GAN vocoder (ConvTranspose1D chain)
   - VE (speaker encoder for cloning)
5. Wire into CLI backend + Session API.
6. Test: TTS → ASR roundtrip on parakeet.

**Effort:** ~1500 LOC, 1-2 full sessions.

### Other open items (by priority)

| # | Item | Effort | Notes |
|---|---|---|---|
| #52 | Qwen3-TTS perf pass | Medium | ~16 ms/frame over budget. Needs quiet machine + qwen3-tts model. Gated env flags ready: FUSED_QKV, LK_BUCKET, CP_STEP0_CACHE. |
| #56 | Kokoro phonemizer | Small | Mandarin tone numbers + JA kanji→kana. Needs `pypinyin` and `mecab`/`kakasi` integration. |
| #58 | MOSS-Audio-4B | Large | New audio-understanding backend. DeepStack cross-layer feature injection — entirely new arch. |
| #59 | Binding parity | Small | Go has full surface. Java needs: diarize, punc wrappers. Ruby needs: diarize, LID, punc, registry/cache. JS needs WebAssembly rethink. |
| #51c | MiMo F16 step decode | Small | BLOCKED — needs ≥32 GB RAM. |

### Blocked / Parked

- #42 VibeVoice-ASR 7B — needs ≥16 GB RAM
- #43 Fun-ASR-Nano — license unclear
- #9 Parakeet TDT GPU — parked (encoder 85%+ of time, <0.7s savings)

## Key files to know

| Path | What |
|---|---|
| `PLAN.md` | All roadmap items with status |
| `HISTORY.md` | Completed work write-ups (§1-§76) |
| `LEARNINGS.md` | Technical lessons learned |
| `tools/test-all-backends.py` | 19-backend regression suite |
| `src/core/beam_decode.h` | Shared beam search (replay + branched KV) |
| `src/core/greedy_decode.h` | Shared greedy decode |
| `src/crispasr_c_api.cpp` | C API (~127 exports, 3400 lines) |
| `examples/cli/crispasr_backend_*.cpp` | Per-backend CLI adapters |
| `examples/server/ws_stream.cpp` | WebSocket streaming server |

## Build & test

```bash
cd /mnt/akademie_storage/whisper.cpp
git pull
cmake -B build && cmake --build build -j$(nproc)

# Smoke test all backends
python3 tools/test-all-backends.py --skip-missing --cache-mode keep

# Feature test (longer, tests beam/temp/punc/vad/lid)
python3 tools/test-all-backends.py --profile feature --skip-missing

# Unit tests
./build/bin/test-core-decode
./build/bin/test-registry
./build/bin/test-crispasr-cache
./build/bin/test-params
```

## User preferences (from memory)

- No Co-Authored-By lines in commits
- Never close GitHub issues (user handles that)
- Never use /tmp — all models/data go to /mnt/storage subfolders
- Models are cached at `/home/claudeuser/.cache/crispasr/` (~23 GB, 27 files)
- VAD uses actual VAD, not fixed chunks
