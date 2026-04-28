"""NeMo Parakeet-TDT reference dump backend.

Loads `nvidia/parakeet-tdt_ctc-0.6b-ja` (or v3) via NeMo and captures the
mel features and final encoder output for crispasr-diff comparison
against the C++ runtime. Intended for diagnosing the JA TDT decoder bug
(emits 1 token then collapses to blanks) where the C++ encoder
diverged from the NeMo reference at the FIRST output frame.

Stages:

  raw_audio        (N,)            input PCM
  mel_spectrogram  (n_mels, T_mel) NeMo preprocessor output, batch-stripped
  encoder_output   (T_enc, d_model) FastConformer encoder output

Captures match the names that `examples/cli/crispasr_diff_main.cpp`
already looks up for the "parakeet" backend.

Usage:

  python tools/dump_reference.py --backend parakeet \\
      --model-dir nvidia/parakeet-tdt_ctc-0.6b-ja \\
      --audio /tmp/parakeet-ja/ja16k.wav \\
      --output /tmp/parakeet-ja/ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "pre_encode_output",
    "encoder_output",
] + [f"encoder_layer_{i}" for i in range(24)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Parakeet-TDT reference forward and return stage captures.

    `model_dir` may be either a local path to an extracted .nemo or a
    HuggingFace pretrained name (e.g. "nvidia/parakeet-tdt_ctc-0.6b-ja").

    Per-layer captures (`pre_encode_output`, `encoder_layer_K`) use
    forward hooks on `model.encoder.pre_encode` and
    `model.encoder.layers[K]` so we get the exact tensor each module
    produces — no manual reconstruction. All captures are transposed
    to (T, d_model) row-major to match crispasr's flat layout.
    """
    import torch
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading NeMo Parakeet-TDT model from {pretrained}")
    if pretrained.startswith("nvidia/") or "/" not in pretrained:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
    else:
        model = nemo_asr.models.ASRModel.restore_from(pretrained)
    model.eval()
    dev = next(model.parameters()).device

    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # ---- Forward hooks: capture per-layer encoder activations ----
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def _save(name: str):
        def hook(_mod, _inp, output):
            # NeMo conformer modules return either a Tensor or a tuple
            # whose first element is the (B, T, D) hidden state. Some
            # versions return (B, D, T); we normalise both to (B, T, D)
            # by sniffing the dim that matches the encoder d_model.
            t = output[0] if isinstance(output, (tuple, list)) else output
            captured[name] = t.detach().cpu().float()
        return hook

    enc = model.encoder
    if "pre_encode_output" in stages and hasattr(enc, "pre_encode"):
        handles.append(enc.pre_encode.register_forward_hook(_save("pre_encode_output")))

    layers = getattr(enc, "layers", None)
    if layers is not None:
        for i in range(len(layers)):
            stage = f"encoder_layer_{i}"
            if stage in stages:
                handles.append(layers[i].register_forward_hook(_save(stage)))

    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=sig, length=sig_len)
        # feats: (B=1, n_mels, T_mel). The C++ runtime stores mel in the
        # TimeMels layout (T_mel, n_mels) — ne[0]=n_mels is the fast axis.
        # Transpose so flat-element ordering matches what
        # parakeet_compute_mel returns.
        if "mel_spectrogram" in stages:
            m = feats[0].transpose(0, 1).contiguous()
            out["mel_spectrogram"] = m.detach().cpu().float().numpy()

        encf, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        # encf: (B=1, d_model, T_enc) in NeMo's convention. crispasr-diff's
        # parakeet_encoder_r returns (T_enc, d_model), so transpose to match.
        if "encoder_output" in stages:
            T_enc = int(enc_len.item())
            e = encf[0, :, :T_enc].transpose(0, 1).contiguous()
            out["encoder_output"] = e.detach().cpu().float().numpy()

    for h in handles:
        h.remove()

    # ---- Normalise per-layer captures to (T, d_model) ----
    # NeMo's ConvSubsampling (pre_encode) returns (B, T, D) directly;
    # ConformerLayer also returns (B, T, D). The final encoder transposes
    # to (B, D, T) at the end, but per-layer outputs are time-major.
    T_enc = int(enc_len.item()) if "encoder_output" in stages else None
    for name, t in captured.items():
        if t.ndim == 3:
            # (B, T, D)
            arr = t[0]  # (T, D)
            if T_enc is not None and arr.shape[0] >= T_enc:
                arr = arr[:T_enc]
            out[name] = arr.contiguous().numpy()
        elif t.ndim == 2:
            out[name] = t[0].contiguous().numpy()

    return out
