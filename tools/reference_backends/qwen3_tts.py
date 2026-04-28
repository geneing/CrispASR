"""Qwen3-TTS-12Hz-{0.6B,1.7B}-Base reference dump backend.

Captures stage-by-stage activations from the official `qwen_tts` package
(`pip install -U qwen-tts` + clone of QwenLM/Qwen3-TTS) so we can diff
the CrispASR talker against the bit-true PyTorch path. The test prompt
is fixed and embedded in the dump so both sides see exactly the same
inputs.

Stages dumped (subset selectable via tools/dump_reference.py --stages):

  text_input_ids        — the synth text, tokenised by the official processor
  ref_input_ids         — the ref text (voice clone prompt) tokenised
  text_proj_out         — text_embedding(input_ids) → text_projection
                          ⇒ (T, hidden_size). Pure-text path that doesn't
                          depend on speaker_embed / codec splice, and
                          therefore the easiest first-line numerical
                          check on the CrispASR side.
  talker_layer_0_out    — output of decoder layer 0 on the prefill mix
  talker_layer_27_out   — output of the last decoder layer
  talker_output_norm    — final RMSNorm output
  talker_logits         — codec_head(last hidden state) for the prefill
                          tail position
  generated_codes       — first N greedy codebook-0 codes from
                          generate_voice_clone(do_sample=False)

The "audio" arg in tools/dump_reference.py is repurposed for TTS: pass
a 16 kHz mono WAV that's BOTH the reference audio (for voice cloning)
AND a placeholder so the existing dispatcher's audio-loading path
doesn't break. The synth text and ref text are env-configurable
(QWEN3_TTS_REF_TEXT / QWEN3_TTS_SYN_TEXT) with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_input_ids",
    "ref_input_ids",
    "text_proj_out",
    "talker_layer_0_out",
    "talker_layer_27_out",
    "talker_output_norm",
    "talker_logits",
    "generated_codes",
]

# Defaults match the official examples/test_model_12hz_base.py smoke
# test so the diff is reproducible without arguments.
_DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
_DEFAULT_SYN_TEXT = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
_DEFAULT_LANG = "English"


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Qwen3-TTS forward + greedy decode, return captured stage tensors.

    `audio` is the reference audio (16 kHz mono, float32). `model_dir`
    is the HF repo id or local snapshot path. The synth text and ref
    text come from QWEN3_TTS_SYN_TEXT / QWEN3_TTS_REF_TEXT env vars,
    falling back to the official sample defaults.
    """
    import torch

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise SystemExit(
            "qwen_tts package not found. Install with: pip install -U qwen-tts\n"
            f"(original import error: {e})")

    syn_text = os.environ.get("QWEN3_TTS_SYN_TEXT", _DEFAULT_SYN_TEXT)
    ref_text = os.environ.get("QWEN3_TTS_REF_TEXT", _DEFAULT_REF_TEXT)
    language = os.environ.get("QWEN3_TTS_LANG", _DEFAULT_LANG)

    print(f"  loading Qwen3-TTS Base from {model_dir} (CPU, fp32, eager attn)")
    tts = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = tts.model
    model.eval()
    talker = model.talker
    processor = tts.processor

    out: Dict[str, np.ndarray] = {}

    # ---- Tokenise the chat-template prompt + ref text ----
    syn_chat = tts._build_assistant_text(syn_text)  # noqa: SLF001
    ref_chat = tts._build_ref_text(ref_text)        # noqa: SLF001
    syn_ids = processor(text=syn_chat, return_tensors="pt", padding=True)["input_ids"]
    ref_ids = processor(text=ref_chat, return_tensors="pt", padding=True)["input_ids"]
    if syn_ids.dim() == 1:
        syn_ids = syn_ids.unsqueeze(0)
    if ref_ids.dim() == 1:
        ref_ids = ref_ids.unsqueeze(0)
    if "text_input_ids" in stages:
        out["text_input_ids"] = syn_ids[0].detach().cpu().numpy().astype(np.int32).astype(np.float32)
    if "ref_input_ids" in stages:
        out["ref_input_ids"] = ref_ids[0].detach().cpu().numpy().astype(np.int32).astype(np.float32)

    # ---- Pure-text stage: text_embedding + text_projection on the syn prompt ----
    # This isolates the resize-MLP path from the codec splice, so a
    # mismatch here implicates only the text_proj fc1/fc2 + the
    # text_embedding lookup. Done WITHOUT inference_mode so we can
    # also call layer_0 hooks below — but we don't need grads here, so
    # wrap in no_grad to save a few cycles.
    if "text_proj_out" in stages:
        with torch.no_grad():
            text_embeds = talker.get_text_embeddings()(syn_ids)
            text_proj_out = talker.text_projection(text_embeds)
        out["text_proj_out"] = text_proj_out[0].detach().cpu().numpy().astype(np.float32)

    # ---- Per-layer + lm_head dump via forward hooks on a real prefill ----
    captures: Dict[str, np.ndarray] = {}

    def cap(name: str, take_first: bool = True):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    layer_hook_map = {
        "talker_layer_0_out":   talker.model.layers[0],
        "talker_layer_27_out":  talker.model.layers[-1],
        "talker_output_norm":   talker.model.norm,
        "talker_logits":        talker.codec_head,
    }
    for stage_name, mod in layer_hook_map.items():
        if stage_name in stages:
            handles.append(mod.register_forward_hook(cap(stage_name)))

    # ---- Run generate (greedy) for deterministic codes + activations ----
    # do_sample=False => greedy; max_new_tokens=64 keeps this fast for the
    # diff harness even on CPU. Captures fire on the prefill forward.
    if any(s in stages for s in (*layer_hook_map.keys(), "generated_codes")):
        # Need at least 1 ref item; build the standard ICL prompt.
        # `audio` is the reference WAV (already 16 kHz mono float32).
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=(audio.astype(np.float32), 16000),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        with torch.no_grad():
            wavs, _sr = tts.generate_voice_clone(
                text=syn_text,
                language=language,
                voice_clone_prompt=prompt_items,
                max_new_tokens=int(max(64, max_new_tokens or 0)),
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                top_k=1,
            )
        # generate_voice_clone returns wavs; we don't use them here, but
        # the talker codes were captured implicitly via the lm_head hook.
        # For an explicit code stream we'd need a tap inside .generate
        # — skipped for v0; the logits hook is enough for diff testing.

    for h in handles:
        h.remove()
    out.update(captures)

    return out
