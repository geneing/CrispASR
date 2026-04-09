# Voxtral-Mini-4B-Realtime-2602 port plan

Substantially different architecture from the 3B — not just a dim swap.

## Key differences from Voxtral-Mini-3B

| | 3B (ported) | **4B Realtime** |
| --- | --- | --- |
| Encoder attention | Full (abs pos embed) | **RoPE θ=1e6 + sliding window (750)** |
| Encoder heads | 20 MHA | **32 MHA** |
| LLM layers | 30 | **26** |
| LLM FFN | 8192 | **9216** |
| LLM RoPE θ | 1e8 | **1e6** |
| LLM attention | Full | **Sliding window (8192)** |
| Embeddings | Separate lm_head | **Tied (token_embd = lm_head)** |
| Focus | Quality + understanding | **Realtime streaming** |
| Audio tokens/frame | 1 per 4 Whisper frames | **audio_length_per_tok=8** |
| Class | `VoxtralForConditionalGeneration` | `VoxtralRealtimeForConditionalGeneration` |

## New pieces needed

1. **RoPE encoder** — the 4B's audio encoder uses RoPE position embedding
   instead of learned absolute. This is a non-trivial change from the
   Whisper-style encoder we used for the 3B. Needs `ggml_rope_ext` in the
   encoder block (currently only used in the LLM).
2. **Sliding window attention (encoder)** — window size 750 in the encoder
   means each position only attends to positions ± 375. Needs a windowed
   mask or a sliding-window-aware attention kernel.
3. **Sliding window attention (LLM)** — window size 8192. Same mechanism
   as the encoder SWA but in the decoder. Affects the KV cache: only the
   last 8192 positions' K/V need to be kept.
4. **Tied embeddings** — `lm_head.weight = token_embd.weight` transposed.
   The GGUF won't have a separate `output.weight`; just use `token_embd.weight`
   for both lookup and projection.
5. **Different projector** — `downsample_factor=4` still, but
   `audio_length_per_tok=8` suggests a different frame-stacking pattern.
6. **Realtime streaming** — `default_num_delay_tokens=6` suggests a
   streaming architecture where the model predicts with a 6-token delay.
   For offline (non-streaming) use we can ignore this and just run the
   full forward.

## Effort estimate

~3-4 days. The encoder SWA + RoPE and LLM SWA are the biggest new pieces.
The rest (converter, loader, dim changes) is mechanical.

## Also in the Voxtral family

- `Voxtral-Small-24B-2507`: 24B full version (same arch as 3B but much
  larger LLM). Too big for CPU inference but interesting as a quality reference.
- `Voxtral-4B-TTS-2603`: text-to-speech, completely different modality.
  Same LLM dims as the 4B Realtime (26 layers, 3072 hidden, 9216 FFN)
  but outputs speech audio via a codec vocoder rather than text tokens.
  Out of scope for CrispASR's ASR focus.
