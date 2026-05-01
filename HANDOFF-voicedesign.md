# Handoff: Qwen3-TTS-12Hz-1.7B-VoiceDesign runtime port

## What VoiceDesign is

A Qwen3-TTS variant that generates speech in a voice **described by a
natural-language instruction**, no reference audio and no preset
speaker required:

```python
# upstream pyusage
model.generate_voice_design(
    text="Hello, I'm an excited engineer.",
    instruct="A young female voice with a slight British accent, energetic, slightly fast paced",
    language="English",
)
```

The model is `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (Apache-2.0,
3.83 GB safetensors). Same talker architecture as the 1.7B-Base /
1.7B-CustomVoice we already support — only the prefill conversation
differs. Note: there is **no 0.6B-VoiceDesign**; the upstream
`generate_custom_voice` even disables `instruct` for 0b6 entirely
(see `qwen_tts/inference/qwen3_tts_model.py` ~L799). VoiceDesign is
1.7B only.

## What we already have

| Piece | Status |
|---|---|
| Converter (`models/convert-qwen3-tts-to-gguf.py`) | Already accepts `tts_model_type=voice_design`. The empty `spk_id` dict means it skips the `qwen3tts.spk_names` array — that's fine. **Needs a smoke test** to confirm GGUF round-trips cleanly. |
| Runtime hparam (`src/qwen3_tts.cpp`) | `tts_model_type` is parsed and stored in `ctx->hp.tts_model_type` (string). Comment at L337 already lists `"base" \| "custom_voice" \| "voice_design"`. |
| 1.7B talker forward (Q/KV/FFN/RMSNorm + mrope) | Validated for 1.7B-Base — VoiceDesign uses the same talker. |
| Code predictor / codec / 12Hz tokenizer | All shared with Base + CustomVoice. |
| C-ABI for instruct text | **NONE.** No `qwen3_tts_set_instruct` exists. |
| CLI flag for instruct | **NONE.** `--voice` is overloaded for ICL WAV path / preset speaker name; we'll need a new flag (suggested: `--instruct "..."`) for VoiceDesign. |

## What's missing in our runtime

The upstream `Qwen3TTSForConditionalGeneration.generate(...)` builds
a per-utterance prefill embedding tensor that depends on
`(speaker_embed, instruct_ids, voice_clone_prompt)`. Reading the
relevant code paths in `qwen_tts/core/models/modeling_qwen3_tts.py`
~L2022–L2200, the differences boil down to:

| Component | Base (ICL) | CustomVoice | VoiceDesign |
|---|---|---|---|
| Reference WAV → spk_embed | ECAPA + mean-pool of ref codes | n/a | n/a |
| spk_id lookup → spk_embed | n/a | `talker.token_embd[spk_id]` | n/a (`speaker_embed = None`) |
| `instruct_ids` (description) | None | optional | **required** |
| Codec bridge layout | `[think, think_bos, lang, think_eos, spk, codec_pad, codec_bos]` (or 6-element no-lang version) | same as Base but `spk` is from talker.token_embd | **same as Base/CV but `spk` row is OMITTED** — bridge becomes `[think_*, codec_pad, codec_bos]` (4 or 5 frames instead of 6 or 7) |
| Conversation prefix | `<\|im_start\|>assistant\nREF<\|im_end\|>\n<\|im_start\|>assistant\nTEXT<\|im_end\|>\n<\|im_start\|>assistant\n` | `<\|im_start\|>assistant\nTEXT<\|im_end\|>\n<\|im_start\|>assistant\n` | **`<\|im_start\|>user\nINSTRUCT<\|im_end\|>\n` PREPENDED to the CustomVoice prefix** |
| Embedding path for the instruct block | n/a | `text_proj(text_embd(instruct_ids))` if non-empty | **same — required** |

Concretely, `talker_input_embeds[i]` in `generate()` is the
concatenation of (in order):
1. `text_proj(text_embd(instruct_ids[i]))` — only if instruct is set
2. `_talker_input_embed_role` — `text_proj(text_embd("<|im_start|>assistant\n"))`
3. `_talker_input_embed` — `tts_pad×(L_codec−2) + tts_bos + codec_input_embedding[:-1]`

For VoiceDesign you set `instruct_ids[i] = tokenize("<|im_start|>user\n{description}<|im_end|>\n")`,
`speaker = None` (so `speaker_embed = None` and the codec bridge skips
the spk frame).

## Concrete plan (estimated 250–350 LOC)

### 1. Converter (~10 LOC + smoke test)

Already ~works. Run:

```bash
HF_HOME=/Volumes/backups/ai/huggingface-hub \
HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python models/convert-qwen3-tts-to-gguf.py \
    --input Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --output /Volumes/backups/ai/crispasr-models/qwen3-tts-12hz-1.7b-voicedesign-f16.gguf \
    --outtype f16
```

Then a Q8_0 + Q4_K. Confirm `qwen3tts.tts_model_type == "voice_design"`
in the GGUF metadata. If converter blows up on the empty `spk_id`
dict, guard the `if spk_id_map:` branch (it should already short-circuit).

### 2. C++ runtime — new prefill builder (~150 LOC)

Add `build_voicedesign_prefill_embeds(c, instruct_text, syn_text,
&prefill_embeds, &T_pre, &trailing, &M_trail)` next to
`build_customvoice_prefill_embeds` in `src/qwen3_tts.cpp` (~L2544).
Mirror the CustomVoice version but:

* Tokenise instruct via `<|im_start|>user\n{instruct}<|im_end|>\n`
  (helper exists conceptually — `_build_instruct_text` in the Python
  ref is `f"<|im_start|>user\n{instruct}<|im_end|>\n"`). We already
  push special tokens in `tokenise_assistant_text`; factor out a
  `tokenise_user_instruct(c, text)` helper or extend the existing
  one.
* Run those instruct ids through `text_embd` (talker
  `text_token_embd_w` lookup) → `text_projection` (talker
  `text_proj_w`). The Python call is
  `talker.text_projection(talker.get_text_embeddings()(instruct_id))`.
  Both ops are already exposed in our talker forward — see
  `run_text_proj` / `run_embed_text` in `qwen3_tts.cpp`.
* Build the codec bridge **without the speaker frame**:
  `L_codec = codec_prefill.size() + 2` (just pad+bos, no spk).
* Concatenate `[instruct_proj | role | bridge_with_text_block]`
  (instruct first, role next, bridge last) along time axis to form
  `prefill_embeds`.
* `trailing_text_hidden` and `M_trail` are derived the same way as
  CustomVoice/Base — the trailing text block is the same shape; only
  the leading bridge changes.

### 3. C-ABI + dispatcher (~30 LOC)

Add to `src/qwen3_tts.h`:

```c
// VoiceDesign: set the natural-language voice description used as
// the instruct prompt. Required before qwen3_tts_synthesize_streaming
// when the loaded model is VoiceDesign. Re-callable; latest call wins.
// Returns 0 on success, -1 if the loaded model is not VoiceDesign.
int qwen3_tts_set_instruct(struct qwen3_tts_context* ctx, const char* instruct);

// Returns true if the loaded model is a VoiceDesign variant.
int qwen3_tts_is_voice_design(struct qwen3_tts_context* ctx);
```

In the streaming entry point (~L5076 of `qwen3_tts.cpp`), branch:

```cpp
if (ctx->hp.tts_model_type == "voice_design") {
    if (ctx->runtime_instruct.empty()) {
        fprintf(stderr, "qwen3_tts: VoiceDesign needs --instruct\n");
        return nullptr;
    }
    if (!build_voicedesign_prefill_embeds(ctx, ctx->runtime_instruct, text, ...))
        return nullptr;
}
```

Store `runtime_instruct` (`std::string`) on the context.

### 4. CLI wiring (~20 LOC)

In `examples/cli/crispasr_backend_qwen3_tts.cpp`:

* Add `--instruct "..."` flag, fed to `qwen3_tts_set_instruct`.
* Detect VoiceDesign via `qwen3_tts_is_voice_design` after model load.
  If yes and `--voice` was passed, error with a helpful message
  ("VoiceDesign uses --instruct, not --voice").
* If VoiceDesign and `--instruct` is empty, error before generation.

### 5. Registry entry (~10 LOC)

In `src/crispasr_model_registry.cpp` next to the existing `qwen3-tts*`
rows:

```cpp
{"qwen3-tts-1.7b-voicedesign", "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf",
 "https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF/resolve/main/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf",
 ...},
```

(Upload the GGUFs to `cstr/qwen3-tts-1.7b-voicedesign-GGUF` first.)

### 6. Validation (~30 LOC)

* **JFK-equivalent end-to-end**: synthesize a sample text with a
  fixed instruct, ASR-roundtrip via parakeet-v3-en. Peak/RMS gates
  are necessary but insufficient (per existing `feedback_tts_validation`
  memory) — must transcribe back to something close to the input
  text.
* **Stage diff** (optional but worth it): extend
  `tools/reference_backends/qwen3_tts.py` with a `voice_design` mode
  that records `instruct_emb`, `prefill_emb`, frame-0 codec logits.
  Then `crispasr-diff qwen3-tts <gguf> <ref.gguf> <wav>` should clear
  cos ≥ 0.99 at every stage.

### 7. Docs

* Add a `### VoiceDesign — describe the voice in text` subsection to
  the README's Qwen3-TTS section, with an example invocation.
* Add a row to the Qwen3-TTS supported-backends sub-table.
* Update HISTORY.md with a `### 56.` entry once shipped.

## Reference code (read these first)

* `qwen_tts/core/models/modeling_qwen3_tts.py` ~L2022–L2200 — the
  `generate()` method that branches on speaker_embed / instruct_ids /
  voice_clone_prompt. Local copy: `/tmp/modeling_qwen3_tts.py` if
  still cached, else fetch from
  `https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models/modeling_qwen3_tts.py`.
* `qwen_tts/inference/qwen3_tts_model.py` ~L637–L728 — the
  `generate_voice_design` wrapper. Same fetch path.
* `src/qwen3_tts.cpp` ~L2299–L2600 — our existing
  `build_icl_prefill_embeds` (Base) and `build_customvoice_prefill_embeds`
  (CustomVoice). The new VoiceDesign builder lives here.
* `Qwen3-TTS-12Hz-1.7B-VoiceDesign/config.json` (HuggingFace) —
  confirms `tts_model_type=voice_design`, empty `spk_id`, otherwise
  identical talker_config to 1.7B-CustomVoice.

## Bonus: 1.7B-CustomVoice (same session, much smaller)

While you're in this area, the 1.7B-CustomVoice variant is also
unsupported but trivial — the C++ CustomVoice runtime is already
shape-agnostic between 0.6B and 1.7B. Just:

```bash
# Convert (HF cache already has the weights at
# /Volumes/backups/ai/huggingface-hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/)
HF_HOME=… python models/convert-qwen3-tts-to-gguf.py \
    --input Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --output … --outtype f16
```

Plus an HF upload + one registry entry. ~30 LOC of repo changes.
Worth bundling into the same PR as VoiceDesign.

## Validation gate

* JFK-equivalent ASR roundtrip transcribes back close to the input
  text (≥ 90% word-level recall).
* `crispasr-diff qwen3-tts ... voicedesign` cos ≥ 0.99 on
  `instruct_emb`, `prefill_emb`, frame-0 codec logits.
* Peak ≥ 8000 + RMS ≥ 1000 on the produced WAV (existing TTS
  validation gate).
* `--backend qwen3-tts -m qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf
  --instruct "young female with British accent" --tts "Hello world"
  --tts-output out.wav` must produce audible English speech in a
  voice that matches the description.

## Suggested PLAN.md entry

`#59 Qwen3-TTS-VoiceDesign + 1.7B-CustomVoice` — Medium effort, High
ROI for users who can't or won't ship reference WAVs. Phased: Phase
A = VoiceDesign runtime + GGUFs + CLI; Phase B = 1.7B-CustomVoice
GGUFs + registry; Phase C = stage-diff validation.
