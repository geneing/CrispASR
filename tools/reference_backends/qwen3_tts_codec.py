"""Qwen3-TTS-Tokenizer-12Hz codec decoder reference dump backend.

Captures stage-by-stage activations from the official PyTorch
Qwen3TTSTokenizerV2Decoder to diff against the CrispASR C++ codec decoder.

Input: deterministic codes (T=10 frames × 16 codebooks, all zeros by default).
This avoids the voice-pack + talker dependency — only the codec tokenizer model
directory is needed.

Stages dumped:
  codec_input_codes    — (T_codec, n_q) int32 as float32
  codec_rvq_out        — after SplitRVQ decode: (512, T_codec) channels-first
  codec_pre_conv_out   — after pre_conv CausalConvNet: (1024, T_codec)
  codec_xfmr_out       — after transformer + output_proj: (1024, T_codec)
  codec_up0_out        — after first ConvNeXt upsample: (1024, 2*T_codec)
  codec_up1_out        — after second ConvNeXt upsample: (1024, 4*T_codec)
  codec_in_conv_out    — after in_conv: (1536, 4*T_codec)
  codec_blk0_out       — after DecoderBlock 0 (stride=8): (768, 32*T_codec)
  codec_pcm            — final clamp output: (T_pcm,) = (1920*T_codec,)

The "audio" arg is unused (required by the dispatcher but ignored here).
Set QWEN3_TTS_CODEC_T=N to use N codec frames (default 10).
Set QWEN3_TTS_CODEC_CODE=K to use K as the constant code value (default 0).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "codec_input_codes",
    "codec_rvq_out",
    "codec_pre_conv_out",
    "codec_xfmr_out",
    "codec_up0_out",
    "codec_up1_out",
    "codec_in_conv_out",
    "codec_blk0_out",
    "codec_pcm",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run the codec decoder forward and return captured stage tensors."""
    import torch

    # Prefer the local ref tree (same commit as the C++ code) over the pip package.
    ref_path = Path(__file__).resolve().parents[2] / "ref" / "Qwen3-TTS"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    T = int(os.environ.get("QWEN3_TTS_CODEC_T", "10"))
    code_val = int(os.environ.get("QWEN3_TTS_CODEC_CODE", "0"))
    n_q = 16

    print(f"  loading Qwen3-TTS-Tokenizer-12Hz from {model_dir} (CPU, fp32)")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    decoder = model.decoder

    # Build deterministic input codes: [1, n_q, T] int64 for PyTorch
    codes_np = np.full((n_q, T), code_val, dtype=np.int32)  # [n_q, T]
    codes_pt = torch.tensor(codes_np, dtype=torch.long).unsqueeze(0)  # [1, n_q, T]

    out: Dict[str, np.ndarray] = {}

    if "codec_input_codes" in stages:
        # codes_np is [n_q, T]. Store as [T, n_q] (time-first) → GGUF ne[0]=n_q.
        # The C++ graph stores codes in [T, n_q] layout (ne[0]=T innermost? no,
        # actually the input tensor has ne[0]=T ne[1]=n_q). This stage is
        # informational only (not compared element-by-element in the diff).
        out["codec_input_codes"] = codes_np.T.astype(np.float32)  # [T, n_q]

    # ── Hooks to capture intermediate activations ──────────────────────────
    captures: Dict[str, Any] = {}

    def make_hook(name: str):
        def hook(_mod, _inp, out_tensor):
            if name not in captures:
                t = out_tensor[0] if isinstance(out_tensor, tuple) else out_tensor
                captures[name] = t.detach().cpu().float().squeeze(0).numpy()
        return hook

    handles = []

    # RVQ output: the Decoder calls self.quantizer.decode() directly (not __call__),
    # so a forward_hook on the quantizer module wouldn't fire. Instead, hook
    # pre_conv's pre-hook to capture the quantizer output which is pre_conv's input.
    if "codec_rvq_out" in stages:
        def cap_rvq(_mod, args):
            if "codec_rvq_out" not in captures and args:
                captures["codec_rvq_out"] = args[0].detach().cpu().float().squeeze(0).numpy()
        handles.append(decoder.pre_conv.register_forward_pre_hook(cap_rvq))

    # pre_conv output: (B, 1024, T) squeeze → (1024, T)
    if "codec_pre_conv_out" in stages:
        handles.append(decoder.pre_conv.register_forward_hook(make_hook("codec_pre_conv_out")))

    # Transformer output via post-hook on pre_transformer.
    # The transformer emits last_hidden_state (B, T, 1024) via BaseModelOutputWithPast;
    # after .permute(0, 2, 1) in Decoder.forward the shape is (B, 1024, T).
    # Hook on the transformer model itself to get last_hidden_state, then
    # capture the channels-first form after the permute by hooking the
    # upsample[0] PRE-hook (which sees the permuted tensor as first input).
    if "codec_xfmr_out" in stages:
        def cap_xfmr(_mod, args, _kw):
            h = args[0] if args else None
            if h is not None and "codec_xfmr_out" not in captures:
                captures["codec_xfmr_out"] = h.detach().cpu().float().squeeze(0).numpy()  # (1024, T)
        handles.append(decoder.upsample[0][0].register_forward_pre_hook(
            cap_xfmr, with_kwargs=True))

    # ConvNeXt upsample outputs
    if "codec_up0_out" in stages:
        handles.append(decoder.upsample[0][1].register_forward_hook(make_hook("codec_up0_out")))
    if "codec_up1_out" in stages:
        handles.append(decoder.upsample[1][1].register_forward_hook(make_hook("codec_up1_out")))

    # decoder[0] = CausalConvNet(1024→1536) = in_conv
    if "codec_in_conv_out" in stages:
        handles.append(decoder.decoder[0].register_forward_hook(make_hook("codec_in_conv_out")))

    # decoder[1] = DecoderBlock 0 (stride=8, 1536→768)
    if "codec_blk0_out" in stages:
        handles.append(decoder.decoder[1].register_forward_hook(make_hook("codec_blk0_out")))

    # Final PCM — the full forward captures it directly
    with torch.no_grad():
        pcm_pt = decoder(codes_pt)  # (1, 1, T_pcm)

    for h in handles:
        h.remove()

    if "codec_pcm" in stages:
        out["codec_pcm"] = pcm_pt.detach().cpu().float().squeeze().numpy()  # (T_pcm,)

    # Move hook captures into out.
    # ggml stores [C, T] tensors with ne[0]=C (innermost), ne[1]=T.
    # For a numpy (C, T) C-order array, GGUF ne[0]=T ne[1]=C, which
    # doesn't match. Transpose to (T, C) so GGUF ne[0]=C matches ggml.
    for name, arr in captures.items():
        if name in stages:
            out[name] = arr.T if arr.ndim == 2 else arr

    return out
