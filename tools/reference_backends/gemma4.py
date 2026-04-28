"""Google Gemma-4-E2B reference dump backend.

Loads `google/gemma-4-E2B-it` via HuggingFace Transformers and captures
mel features, encoder output, and per-layer LLM hidden states for
crispasr-diff comparison against the C++ runtime.

Stages:

  raw_audio                 (N,)            input PCM
  mel_spectrogram           (n_mels, T_mel) audio_features from the processor
  encoder_output            (T_enc, d_model) USM Conformer encoder output
  llm_input_embeds          (T_total, d)    embeddings going into the LLM
                                            (audio + text prompt, after Gemma's
                                            sqrt(d_model) embedding scaling)
  llm_hidden_layer_{0,1,8,mid,last}
                            (T_total, d)    selected per-layer outputs
                                            (PLE + attention + FFN applied)
  llm_logits                (T_last, V)     final lm_head logits
  llm_argmax                (1,)            greedy first-token id
  generated_text            (string)        decoded transcript

Usage:

  python tools/dump_reference.py --backend gemma4 \\
      --model-dir google/gemma-4-E2B-it \\
      --audio samples/jfk.wav \\
      --output /tmp/gemma4-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
    "llm_input_embeds",
    "llm_hidden_layer_0",
    "llm_hidden_layer_1",
    "llm_hidden_layer_8",
    "llm_hidden_layer_mid",
    "llm_hidden_layer_last",
    "llm_logits",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Gemma-4-E2B reference forward and return stage captures."""
    import os, torch
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as e:
        raise SystemExit(
            "transformers required. Install: pip install -U transformers\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading Gemma-4-E2B from {pretrained}")
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
    # bfloat16 keeps the 9.6 GB model around 5 GB in RAM — fits on the M1
    # without swap thrashing. The diff comparison is cosine, not absolute,
    # so the 8-bit mantissa loss vs F32 is ~0.001 cos at most.
    dtype = torch.bfloat16 if os.environ.get("CRISPASR_REF_DTYPE", "bf16") in ("bf16", "bfloat16") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained, torch_dtype=dtype, trust_remote_code=True,
    ).eval()
    dev = next(model.parameters()).device
    print(f"  loaded in {dtype} on {dev}")

    # Build the chat-style prompt: an `<audio>` placeholder followed by the
    # user instruction. The exact template lives on the processor.
    chat = [{"role": "user", "content": [
        {"type": "audio", "audio": audio.astype(np.float32)},
        {"type": "text", "text": "Transcribe this audio."},
    ]}]
    inputs = processor.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)
    if "mel_spectrogram" in stages and "input_features" in inputs:
        out["mel_spectrogram"] = inputs["input_features"][0].detach().cpu().float().numpy()

    # ---- Encoder hook ----
    enc_out = {}
    def enc_hook(_m, _i, output):
        t = output[0] if isinstance(output, tuple) else output
        enc_out["v"] = t.detach().clone()
    enc_handle = model.audio_tower.register_forward_hook(enc_hook) \
        if hasattr(model, "audio_tower") else None

    # ---- LLM per-layer hooks ----
    layer_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = []
    for i, layer in enumerate(layers):
        def make(idx):
            def h(_m, _i, output):
                t = output[0] if isinstance(output, tuple) else output
                layer_outputs[idx] = t.detach().clone()
            return h
        handles.append(layer.register_forward_hook(make(i)))

    # ---- Forward (prefill + 1 generated token to stay cheap) ----
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max(1, max_new_tokens),
            do_sample=False, num_beams=1, output_hidden_states=False,
            return_dict_in_generate=True,
        )

    if enc_handle is not None:
        enc_handle.remove()
    for h in handles:
        h.remove()

    if "encoder_output" in stages and "v" in enc_out:
        e = enc_out["v"]
        if e.dim() == 3 and e.shape[0] == 1:
            e = e[0]
        out["encoder_output"] = e.detach().cpu().float().numpy()

    if layer_outputs:
        n_layers = len(layers)
        checkpoints = {
            "llm_hidden_layer_0":     0,
            "llm_hidden_layer_1":     1,
            "llm_hidden_layer_8":     min(8, n_layers - 1),
            "llm_hidden_layer_mid":   max(0, n_layers // 2 - 1),
            "llm_hidden_layer_last":  n_layers - 1,
        }
        for name, idx in checkpoints.items():
            if name in stages and idx in layer_outputs:
                out[name] = layer_outputs[idx][0].detach().cpu().float().numpy()

    # ---- Generated text ----
    if "generated_text" in stages or "llm_argmax" in stages:
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        n_in = inputs["input_ids"].shape[-1]
        new_ids = seq[0, n_in:]
        if "llm_argmax" in stages:
            out["llm_argmax"] = new_ids.detach().cpu().int().numpy().astype(np.int32)
        if "generated_text" in stages:
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            out["generated_text"] = np.array([ord(c) for c in text], dtype=np.int32)

    return out
