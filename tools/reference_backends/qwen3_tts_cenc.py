"""Qwen3-TTS-Tokenizer-12Hz codec ENCODER reference dump.

Hooks the PyTorch Qwen3TTSTokenizerV2Encoder forward at three points so
crispasr-diff can verify the C++ encoder numerically against the
reference (the same diff-harness approach that took ECAPA from cos=0.74
to 0.999999 and brought all 8 codec-decoder stages to PASS).

Stages dumped (all stored in (T, C) time-first to match ggml flat layout):
  cenc_input_audio   — fixed deterministic 24kHz PCM (3s of clone.wav)
  cenc_seanet_out    — output of self.encoder (SEANet) [T_enc, 512]
  cenc_xfmr_out      — output of self.encoder_transformer [T_enc, 512]
  cenc_ds_out        — after self.downsample [T_frames, 512]
  cenc_codes         — final RVQ codes [T_frames, 16] as float32

The audio arg is unused — we use a fixed slice of clone.wav so both
sides see identical inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "cenc_input_audio",
    "cenc_se_init",
    "cenc_se_s0",
    "cenc_se_s1",
    "cenc_se_s2",
    "cenc_se_s3",
    "cenc_seanet_out",
    "cenc_xfmr_out",
    "cenc_ds_out",
    "cenc_codes",
]


def _load_clone_audio() -> np.ndarray:
    """Load clone.wav at 24kHz mono float32. Take first 3 seconds for speed."""
    repo_root = Path(__file__).resolve().parents[2]
    wav_path = repo_root / "samples" / "qwen3_tts" / "clone.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"clone.wav not at {wav_path}")
    # Read RIFF/WAVE — IEEE Float, mono, 24kHz
    with open(wav_path, "rb") as f:
        data = f.read()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF WAV")
    pos = 12
    sr = 0; n_ch = 0; bps = 0; fmt = 0
    samples = None
    while pos < len(data) - 8:
        cid = data[pos:pos+4]
        csz = int.from_bytes(data[pos+4:pos+8], "little")
        if cid == b"fmt ":
            fmt = int.from_bytes(data[pos+8:pos+10], "little")
            n_ch = int.from_bytes(data[pos+10:pos+12], "little")
            sr = int.from_bytes(data[pos+12:pos+16], "little")
            bps = int.from_bytes(data[pos+22:pos+24], "little")
        elif cid == b"data":
            raw = data[pos+8:pos+8+csz]
            if fmt == 3 and bps == 32:
                samples = np.frombuffer(raw, dtype="<f4")
            elif fmt == 1 and bps == 16:
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            else:
                raise ValueError(f"unsupported WAV format fmt={fmt} bps={bps}")
            break
        pos += 8 + csz + (csz % 2)
    if samples is None or sr != 24000 or n_ch != 1:
        raise ValueError(f"expected 24kHz mono, got sr={sr} ch={n_ch}")
    # Take first 3 seconds (smaller diff = faster, frame count divisible by all strides)
    return samples[:24000 * 3].astype(np.float32)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    ref_path = Path(__file__).resolve().parents[2] / "ref" / "Qwen3-TTS"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    print(f"  loading Qwen3-TTS-Tokenizer-12Hz from {model_dir}")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    encoder = model.encoder

    # Fixed input: first 3 seconds of clone.wav at 24kHz
    audio_np = _load_clone_audio()
    print(f"  input audio: {len(audio_np)} samples ({len(audio_np)/24000:.2f}s)")

    out: Dict[str, np.ndarray] = {}
    if "cenc_input_audio" in stages:
        out["cenc_input_audio"] = audio_np

    captures: Dict[str, np.ndarray] = {}

    # Hook 1: SEANet output (after self.encoder)
    if "cenc_seanet_out" in stages:
        def cap_seanet(_mod, _inp, output):
            if "cenc_seanet_out" not in captures:
                t = output[0] if isinstance(output, tuple) else output
                # output is [B, 512, T_enc] — store as (T_enc, 512)
                captures["cenc_seanet_out"] = t.detach().cpu().float().squeeze(0).T.numpy()
        encoder.encoder.register_forward_hook(cap_seanet)

    # Intra-SEANet checkpoints. The encoder.encoder is a ModuleList where
    # layer indices map to: 0=init conv, {3,6,9,12}=stride convs (s0..s3).
    # Hooks on those four MimiConv1d modules give us per-stride outputs.
    seanet_intra = {
        "cenc_se_init": 0,
        "cenc_se_s0":   3,
        "cenc_se_s1":   6,
        "cenc_se_s2":   9,
        "cenc_se_s3":   12,
    }
    for name, idx in seanet_intra.items():
        if name in stages:
            def make_cap(nm):
                def cap(_mod, _inp, out):
                    if nm not in captures:
                        t = out[0] if isinstance(out, tuple) else out
                        # MimiConv1d returns [B, C, T] — store as (T, C)
                        captures[nm] = t.detach().cpu().float().squeeze(0).T.numpy()
                return cap
            encoder.encoder.layers[idx].register_forward_hook(make_cap(name))

    # Hook 2: encoder_transformer output (post-hook on the transformer)
    # The transformer returns (last_hidden_state, ...) tuple where last_hidden_state
    # is [B, T_enc, 512]. Store as (T_enc, 512).
    if "cenc_xfmr_out" in stages:
        def cap_xfmr(_mod, _inp, output):
            if "cenc_xfmr_out" not in captures:
                # Output is BaseModelOutputWithPast(last_hidden_state=..., past_key_values=...)
                if hasattr(output, "last_hidden_state"):
                    t = output.last_hidden_state
                elif isinstance(output, tuple):
                    t = output[0]
                else:
                    t = output
                # (B, T_enc, 512) — squeeze B
                captures["cenc_xfmr_out"] = t.detach().cpu().float().squeeze(0).numpy()
        encoder.encoder_transformer.register_forward_hook(cap_xfmr)

    # Hook 3: downsample output (after the stride-2 conv)
    if "cenc_ds_out" in stages:
        def cap_ds(_mod, _inp, output):
            if "cenc_ds_out" not in captures:
                t = output[0] if isinstance(output, tuple) else output
                # output is [B, 512, T_frames] — store as (T_frames, 512)
                captures["cenc_ds_out"] = t.detach().cpu().float().squeeze(0).T.numpy()
        encoder.downsample.register_forward_hook(cap_ds)

    # Run the encoder
    audio_pt = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    with torch.no_grad():
        result = encoder.encode(audio_pt)
    # result is MimiEncoderOutput with audio_codes [B, n_q, T_codec]
    codes = result.audio_codes if hasattr(result, "audio_codes") else result[0]
    # Take first 16 quantizers (encoder_valid_num_quantizers)
    codes_16 = codes[:, :16, :]  # [B, 16, T_codec]
    if "cenc_codes" in stages:
        # Store as (T_codec, 16) time-first
        out["cenc_codes"] = codes_16[0].T.detach().cpu().numpy().astype(np.float32)

    out.update(captures)
    return out
