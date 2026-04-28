#!/usr/bin/env python3
"""
Convert Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}
(HuggingFace safetensors) → GGUF F16 for the CrispASR `qwen3-tts` backend.

Architecture — confirmed against the public config.json files (April 2026):

  Top-level config:
    model_type            = "qwen3_tts"
    architectures         = ["Qwen3TTSForConditionalGeneration"]
    tokenizer_type        = "qwen3_tts_tokenizer_12hz"
    tts_model_size        = "06b" | "1b7"
    tts_model_type        = "base" | "custom_voice" | "voice_design"
    tts_bos/eos/pad_token_id  = 151672 / 151673 / 151671   (text-side TTS sentinels)

  speaker_encoder_config:
    enc_dim          = 1024     (output: 1024-d voice embedding)
    sample_rate      = 24000

  talker_config — the autoregressive LM that generates audio codes:
    model_type       = "qwen3_tts_talker"
    hidden_size      = 1024  (0.6B)  /  2048  (1.7B)
    num_hidden_layers = 28
    num_attention_heads = 16
    num_key_value_heads = 8
    head_dim         = 128
    intermediate_size = 3072 (0.6B) /  6144 (1.7B)
    vocab_size       = 3072         (audio code vocabulary)
    text_vocab_size  = 151936       (Qwen3 BPE)
    text_hidden_size = 2048
    num_code_groups  = 16           (16 RVQ codebooks per frame)
    max_position_embeddings = 32768
    rope_theta       = 1000000
    rope_scaling     = {interleaved: true, mrope_section: [24, 20, 20]}
    hidden_act       = "silu"
    rms_norm_eps     = 1e-06
    use_sliding_window = false

  code_predictor_config — small head that picks 16 codes per frame:
    num_hidden_layers     = 5
    hidden_size           = 1024
    num_attention_heads   = 16
    num_key_value_heads   = 8
    num_code_groups       = 16
    vocab_size            = 2048    (per-codebook size from the codec)
    max_length            = 20      (max codes generated per group during AR step)

The audio decoder (RVQ → waveform) lives in the SEPARATE
`Qwen/Qwen3-TTS-Tokenizer-12Hz` repo and gets its own converter
(`convert-qwen3-tts-tokenizer-to-gguf.py`).

Usage:

    python models/convert-qwen3-tts-to-gguf.py \\
        --input Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --output qwen3-tts-0.6b-base.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map a HuggingFace tensor name to the GGUF name the C++ runtime
    expects. Returns None to skip (e.g. clipping scalars, vision)."""

    # Skip safetensors metadata-style entries
    if hf_name.endswith("num_batches_tracked"):
        return None

    n = hf_name

    # ── Talker (the audio-code AR LM) ──────────────────────────────────
    # HF ships the talker under `model.talker.`. Map every layer
    # tensor to GGUF's stock `blk.{i}.…` layout so we can reuse the
    # existing core_attn / core_ffn helpers.
    n = n.replace("model.talker.model.embed_tokens.", "talker.token_embd.")
    n = n.replace("model.talker.model.embed_tokens_text.", "talker.token_embd_text.")
    n = n.replace("model.talker.model.norm.", "talker.output_norm.")
    n = n.replace("model.talker.lm_head.", "talker.output.")
    n = n.replace("model.talker.model.layers.", "talker.blk.")
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".self_attn.q_norm.", ".attn_q_norm.")
    n = n.replace(".self_attn.k_norm.", ".attn_k_norm.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    # ── Code predictor (5-layer head that emits the 16 RVQ codes) ──────
    n = n.replace("model.code_predictor.", "code_pred.")

    # ── Speaker encoder (voice embedding extractor, 24 kHz path) ───────
    n = n.replace("model.speaker_encoder.", "speaker.")

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3-TTS to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    talker = cfg.get("talker_config", {})
    speaker = cfg.get("speaker_encoder_config", {})
    cp = talker.get("code_predictor_config", {})

    print(f"\nQwen3-TTS — {cfg.get('tts_model_size', '?')} {cfg.get('tts_model_type', '?')}")
    print(f"  Talker:        {talker.get('num_hidden_layers')}L  "
          f"hidden={talker.get('hidden_size')}  "
          f"heads={talker.get('num_attention_heads')}/{talker.get('num_key_value_heads')}  "
          f"head_dim={talker.get('head_dim')}  "
          f"ff={talker.get('intermediate_size')}  vocab={talker.get('vocab_size')}")
    print(f"  Code predict:  {cp.get('num_hidden_layers')}L  "
          f"hidden={cp.get('hidden_size')}  vocab={cp.get('vocab_size')}  "
          f"groups={cp.get('num_code_groups')}")
    print(f"  Speaker enc:   {speaker.get('enc_dim')}  sr={speaker.get('sample_rate')}")
    print(f"  Text vocab:    {talker.get('text_vocab_size')}  "
          f"text_hidden={talker.get('text_hidden_size')}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="qwen3tts", use_temp_file=True)

    # ----- metadata -----------------------------------------------------
    w.add_name(f"qwen3-tts-{cfg.get('tts_model_size', '?')}-{cfg.get('tts_model_type', '?')}")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))
    def boolv(k, v): w.add_bool(k, bool(v))

    # talker
    u32("qwen3tts.talker.n_layers",   talker.get("num_hidden_layers", 28))
    u32("qwen3tts.talker.d_model",    talker.get("hidden_size", 1024))
    u32("qwen3tts.talker.n_heads",    talker.get("num_attention_heads", 16))
    u32("qwen3tts.talker.n_kv_heads", talker.get("num_key_value_heads", 8))
    u32("qwen3tts.talker.head_dim",   talker.get("head_dim", 128))
    u32("qwen3tts.talker.ff_dim",     talker.get("intermediate_size", 3072))
    u32("qwen3tts.talker.vocab_size", talker.get("vocab_size", 3072))
    u32("qwen3tts.talker.text_vocab_size", talker.get("text_vocab_size", 151936))
    u32("qwen3tts.talker.text_hidden_size", talker.get("text_hidden_size", 2048))
    u32("qwen3tts.talker.n_code_groups", talker.get("num_code_groups", 16))
    u32("qwen3tts.talker.max_pos",    talker.get("max_position_embeddings", 32768))
    u32("qwen3tts.talker.position_id_per_seconds",
        talker.get("position_id_per_seconds", 13))
    f32("qwen3tts.talker.rope_theta", talker.get("rope_theta", 1_000_000))
    f32("qwen3tts.talker.rms_norm_eps", talker.get("rms_norm_eps", 1e-6))
    boolv("qwen3tts.talker.use_sliding_window", talker.get("use_sliding_window", False))

    rope = talker.get("rope_scaling", {}) or {}
    boolv("qwen3tts.talker.rope_interleaved", rope.get("interleaved", True))
    mrope = rope.get("mrope_section", [])
    if mrope:
        w.add_array("qwen3tts.talker.mrope_section", list(mrope))

    # code predictor
    u32("qwen3tts.code_pred.n_layers",     cp.get("num_hidden_layers", 5))
    u32("qwen3tts.code_pred.d_model",      cp.get("hidden_size", 1024))
    u32("qwen3tts.code_pred.n_heads",      cp.get("num_attention_heads", 16))
    u32("qwen3tts.code_pred.n_kv_heads",   cp.get("num_key_value_heads", 8))
    u32("qwen3tts.code_pred.n_code_groups", cp.get("num_code_groups", 16))
    u32("qwen3tts.code_pred.vocab_size",   cp.get("vocab_size", 2048))
    u32("qwen3tts.code_pred.max_length",   cp.get("max_length", 20))

    # speaker encoder
    u32("qwen3tts.speaker.enc_dim",     speaker.get("enc_dim", 1024))
    u32("qwen3tts.speaker.sample_rate", speaker.get("sample_rate", 24000))

    # token sentinels
    u32("qwen3tts.tts_bos_token_id", cfg.get("tts_bos_token_id", 151672))
    u32("qwen3tts.tts_eos_token_id", cfg.get("tts_eos_token_id", 151673))
    u32("qwen3tts.tts_pad_token_id", cfg.get("tts_pad_token_id", 151671))
    u32("qwen3tts.im_start_token_id", cfg.get("im_start_token_id", 151644))
    u32("qwen3tts.im_end_token_id",  cfg.get("im_end_token_id", 151645))
    u32("qwen3tts.assistant_token_id", cfg.get("assistant_token_id", 77091))

    # codec sentinels (audio-code vocab)
    for k in ("codec_bos_id", "codec_eos_token_id", "codec_pad_id",
              "codec_think_id", "codec_nothink_id",
              "codec_think_bos_id", "codec_think_eos_id"):
        if k in talker:
            u32(f"qwen3tts.talker.{k}", talker[k])

    # tokenizer (BPE: vocab.json + merges.txt — same as qwen3-asr)
    vocab_p = model_dir / "vocab.json"
    merges_p = model_dir / "merges.txt"
    if vocab_p.exists():
        with open(vocab_p) as f:
            vocab_d = json.load(f)
        toks = [""] * len(vocab_d)
        for tok, idx in vocab_d.items():
            if idx < len(toks):
                toks[idx] = tok
        w.add_token_list(toks)
        print(f"  Tokens:        {len(toks)} entries from vocab.json")
    if merges_p.exists():
        with open(merges_p) as f:
            merges = [
                ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
        # merges produce GGUF type-9 which our reader rejects; persist as
        # add_string per merge — same workaround as qwen3-asr.
        # NOTE: omit by default, BPE tokens alone are usable for inference.

    # ----- tensors ------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    skipped_examples = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue

        # Detect if anything was actually renamed; if the prefix didn't
        # match any rule the GGUF name will still start with `model.` —
        # that's a bug in our mapping, not a benign skip.
        if gn.startswith(("model.", "lm_head.")):
            if len(skipped_examples) < 20:
                skipped_examples.append(f"  [WARN unmapped] {hf_name}")
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 100 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if skipped_examples:
        print("\n".join(skipped_examples), file=sys.stderr)
        print(f"\n  WARNING: {len(skipped_examples)} unmapped tensor(s) — re-check map_tensor_name()",
              file=sys.stderr)

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
