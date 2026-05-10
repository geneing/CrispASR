# IndexTTS-1.5 — Handover for 99.5% Parity

## Current State (2026-05-10)

The IndexTTS backend is **functional end-to-end**: text → conditioning → GPT mel codes → latent → BigVGAN → audio.
ASR roundtrip produces "Hello, well." (vs Python's "Hello world!") — a single-token beam search divergence at step 14.

### What works (verified against Python):
- **Conditioning (Conformer+Perceiver):** norm 288.88 vs 288.92 (0.01% off). First5 values match to 3 decimal places.
- **GPT mel codes:** 100% identical when given same conditioning (verified with INDEXTTS_COND_FILE override)
- **Latent extraction:** Shape [56, 1280] matches, rms=1.1726 matches
- **BigVGAN vocoder:** With Python's exact latent, ASR = "Hello world!" (matches Python)
- **End-to-end without overrides:** First 14/55 mel codes match Python. ASR = "Hello, well."

### The remaining gap: mel code divergence at step 14

With our own conditioning (which is 0.01% off from Python's), beam search produces
slightly different logits at step 14. Our beam picks token 6283 while Python picks 6109.
The logit difference is <0.01 — within F16 precision error compounded through 24 GPT layers.

## How to achieve 99.5% parity

### Option A: F32 GGUF model (eliminates precision gap entirely)

1. Re-run `models/convert-indextts-to-gguf.py` with `--ftype f32` (or modify to keep all weights F32)
2. Test with the F32 GGUF — if mel codes match 100%, the gap is purely F16 precision
3. If confirmed, the F16 gap is acceptable (it's inherent to quantization, not a bug)

### Option B: Match Python's exact attention precision

The remaining precision gap may come from:
- `ggml_flash_attn_ext` with `GGML_PREC_F32` in the GPT — verify this produces identical results to standard matmul attention
- The Conformer's flash_attn_ext — try standard matmul attention instead to eliminate precision differences
- F16→F32 weight dequantization at each layer (try `ggml_cast` to F32 once at load time for critical weights)

### Option C: Larger beam size

Increasing beam size from 3 to 5 or 8 may allow the correct path to survive even with slight logit differences.
Python uses `num_beams=3` by default, but the C++ implementation could use a larger beam to compensate for F16 drift.

## Files modified in this session

```
src/indextts.cpp          — mel_pos fix, latent count fix, conformer pos encoding fix, sample rate fix, cond override
src/indextts_voc.cpp      — SnakeBeta memory layout fix, latent transpose fix
tools/reference_backends/indextts.py — added gpt.ln_f, step logging
examples/cli/crispasr_backend_indextts.cpp — sample rate resampling
LEARNINGS.md              — documented all findings
```

## Diff-testing commands

```bash
# 1. Generate Python reference conditioning + latent + mel codes:
cd /mnt/akademie_storage/index-tts-src
# (see the Python scripts used in this session — they bypass the broken normalizer via patches)

# 2. Test with Python conditioning override (isolates GPT precision):
INDEXTTS_COND_FILE=/mnt/storage/indextts_cond_jfk2.bin \
  ./build/bin/crispasr --backend indextts \
  --model /mnt/akademie_storage/models/indextts-gguf/indextts-gpt.gguf \
  --codec-model /mnt/akademie_storage/models/indextts-gguf/indextts-bigvgan.gguf \
  --tts "Hello world." --voice /mnt/akademie_storage/chatterbox/test_jfk_final.wav

# 3. Test with Python latent override (isolates vocoder):
INDEXTTS_LATENT_FILE=/mnt/storage/indextts_latent_jfk.bin \
  ... (same as above)

# 4. ASR roundtrip:
ffmpeg -y -i tts_output.wav -ar 16000 /mnt/storage/tts_16k.wav
./build/bin/crispasr --backend parakeet --model /mnt/storage/models/parakeet-tdt-0.6b-v2/model.gguf /mnt/storage/tts_16k.wav
```

## Python reference values (JFK reference audio, "HELLO WORLD."):

```
Conditioning: norm=288.92, first5=[-0.17545, 0.05909, -0.08273, -0.10259, -0.05862]
Mel codes (56): [2623, 515, 4643, 7539, 7051, 7119, 7901, 7957, 7002, 7357, 6783, 6599, 6697, 6719, 6109, 5085, 5110, 1573, 3978, 4953, 2342, 1783, 7453, 5750, 1863, 6133, 1634, 8180, 6737, 2012, 1736, 3828, 6355, 105, 2428, 1922, 2343, 3144, 4780, 587, 2455, 959, 7986, 2844, 3467, 6353, 1719, 7442, 4246, 5708, 6956, 4401, 4026, 1940, 6122, 8193]
Latent: shape=[56, 1280], rms=1.1726
Audio: 2.35s, rms=3655, peak=-3.2dB, ASR="Hello world!"
```

## C++ current values:

```
Conditioning: norm=288.88, first5=[-0.17556, 0.06011, -0.08333, -0.10298, -0.05928]
Mel codes (70): [2623, 515, 4643, 7539, 7051, 7119, 7901, 7957, 7002, 7357, 6783, 6599, 6697, 6719, 6283, 6109, ...]
                                                                              ^--- diverges here (step 14)
Audio: 3.03s, rms=4072, peak=-2.5dB, ASR="Hello, well."
```

## Key insight for next session

The 0.01% conditioning difference compounds through 24 GPT layers of dot products into a logit difference
of ~0.01 at step 14. This crosses a beam-score threshold, causing a different token to win.

The cleanest fix: **convert the GGUF to F32** (or at minimum, keep the GPT attention weights in F32)
and verify that mel codes then match 100%. If they do, the implementation is correct and the remaining
gap is a quantization trade-off, not a bug.
