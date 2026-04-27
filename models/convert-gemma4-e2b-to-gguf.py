#!/usr/bin/env python3
"""Convert Google Gemma-4-E2B model to GGUF format for CrispASR.

Architecture: USM Conformer audio encoder (12L, 1024d) + Gemma4 LLM decoder (35L, 1536d).
The model has 2011 tensors, 9.5 GB in BF16. Conversion needs ~16 GB RAM.

Usage:
    python models/convert-gemma4-e2b-to-gguf.py --input google/gemma-4-E2B-it --output gemma4-e2b-it.gguf
    python models/convert-gemma4-e2b-to-gguf.py --input /path/to/local/dir --output model.gguf --outtype f16

Designed to run on Kaggle (16 GB RAM) or better.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. Install with: pip install huggingface_hub")
    sys.exit(1)


def load_model_dir(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.is_dir():
        print(f"Using local model directory: {model_path}")
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json",
                        "tokenizer_config.json", "generation_config.json"],
    )
    print(f"Downloaded to: {path}")
    return Path(path)


# ── Tensor name mapping ─────────────────────────────────────────────────────
# We strip the clipped-linear min/max scalars (quantization-aware training artifacts)
# and map the rest to a clean GGUF naming scheme.

SKIP_PATTERNS = [
    ".input_max", ".input_min", ".output_max", ".output_min",  # clipping scalars
    "model.vision_tower.",   # skip vision encoder (not needed for ASR)
    "model.embed_vision.",   # skip vision embedding
]


def should_skip(name: str) -> bool:
    return any(p in name for p in SKIP_PATTERNS)


def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace tensor name to GGUF name. Returns None to skip."""
    if should_skip(hf_name):
        return None

    name = hf_name

    # Audio tower
    name = name.replace("model.audio_tower.layers.", "audio.layers.")
    name = name.replace("model.audio_tower.output_proj.", "audio.output_proj.")
    name = name.replace("model.audio_tower.subsample_conv_projection.", "audio.subsample.")

    # Clipped linear: .linear.weight → .weight (strip the .linear. indirection)
    name = name.replace(".linear.weight", ".weight")

    # Audio conformer sub-modules
    name = name.replace(".feed_forward1.", ".ffn1.")
    name = name.replace(".feed_forward2.", ".ffn2.")
    name = name.replace(".ffw_layer_1.", ".up.")
    name = name.replace(".ffw_layer_2.", ".down.")
    name = name.replace(".pre_layer_norm.", ".pre_ln.")
    name = name.replace(".post_layer_norm.", ".post_ln.")
    name = name.replace(".lconv1d.", ".conv.")
    name = name.replace(".depthwise_conv1d.", ".dw_conv.")
    name = name.replace(".linear_start.", ".gate_proj.")
    name = name.replace(".linear_end.", ".out_proj.")
    name = name.replace(".conv_norm.", ".conv_ln.")
    name = name.replace(".norm_pre_attn.", ".attn_pre_ln.")
    name = name.replace(".norm_post_attn.", ".attn_post_ln.")
    name = name.replace(".norm_out.", ".out_ln.")
    name = name.replace(".self_attn.", ".attn.")
    name = name.replace(".q_proj.", ".q.")
    name = name.replace(".k_proj.", ".k.")
    name = name.replace(".v_proj.", ".v.")
    name = name.replace(".post.", ".o.")  # output projection in audio attn
    name = name.replace(".relative_k_proj.", ".rel_k.")

    # Audio subsampling
    name = name.replace(".input_proj_linear.", ".input_proj.")
    name = name.replace(".layer0.conv.", ".conv0.")
    name = name.replace(".layer0.norm.", ".norm0.")
    name = name.replace(".layer1.conv.", ".conv1.")
    name = name.replace(".layer1.norm.", ".norm1.")

    # Audio embedding projection
    name = name.replace("model.embed_audio.embedding_projection.", "audio.embed_proj.")

    # Language model
    name = name.replace("model.language_model.", "llm.")
    name = name.replace(".self_attn.", ".attn.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".post_attn_norm.")
    name = name.replace(".pre_feedforward_layernorm.", ".pre_ffn_norm.")
    name = name.replace(".post_feedforward_layernorm.", ".post_ffn_norm.")
    name = name.replace(".post_per_layer_input_norm.", ".post_ple_norm.")
    name = name.replace(".per_layer_input_gate.", ".ple_gate.")
    name = name.replace(".per_layer_projection.", ".ple_proj.")
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma-4-E2B to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    ac = config.get("audio_config", {})
    tc = config.get("text_config", {})

    print(f"\nGemma-4-E2B")
    print(f"  Audio: {ac.get('num_hidden_layers')}L, hidden={ac.get('hidden_size')}, "
          f"heads={ac.get('num_attention_heads')}, conv_k={ac.get('conv_kernel_size')}")
    print(f"  LLM:   {tc.get('num_hidden_layers')}L, hidden={tc.get('hidden_size')}, "
          f"heads={tc.get('num_attention_heads')}, kv_heads={tc.get('num_key_value_heads')}, "
          f"head_dim={tc.get('head_dim')}")
    print(f"  Vocab: {tc.get('vocab_size')}")

    # Output dtype
    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    # Open safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No .safetensors files found in {model_dir}")
        sys.exit(1)
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Create GGUF writer
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "gemma4e2b", use_temp_file=True)

    # Write metadata
    writer.add_name(config.get("_name_or_path", "gemma-4-E2B-it"))

    # Audio encoder params
    writer.add_uint32("gemma4e2b.audio.hidden_size", ac.get("hidden_size", 1024))
    writer.add_uint32("gemma4e2b.audio.num_layers", ac.get("num_hidden_layers", 12))
    writer.add_uint32("gemma4e2b.audio.num_heads", ac.get("num_attention_heads", 8))
    writer.add_uint32("gemma4e2b.audio.conv_kernel_size", ac.get("conv_kernel_size", 5))
    writer.add_uint32("gemma4e2b.audio.chunk_size", ac.get("attention_chunk_size", 12))
    writer.add_uint32("gemma4e2b.audio.context_left", ac.get("attention_context_left", 13))
    writer.add_uint32("gemma4e2b.audio.output_proj_dims", ac.get("output_proj_dims", 1536))
    writer.add_float32("gemma4e2b.audio.residual_weight", ac.get("residual_weight", 0.5))
    writer.add_float32("gemma4e2b.audio.attention_logit_cap", ac.get("attention_logit_cap", 50.0))

    # LLM params
    writer.add_uint32("gemma4e2b.llm.hidden_size", tc.get("hidden_size", 1536))
    writer.add_uint32("gemma4e2b.llm.num_layers", tc.get("num_hidden_layers", 35))
    writer.add_uint32("gemma4e2b.llm.num_heads", tc.get("num_attention_heads", 8))
    writer.add_uint32("gemma4e2b.llm.num_kv_heads", tc.get("num_key_value_heads", 1))
    writer.add_uint32("gemma4e2b.llm.head_dim", tc.get("head_dim", 256))
    writer.add_uint32("gemma4e2b.llm.intermediate_size", tc.get("intermediate_size", 6144))
    writer.add_uint32("gemma4e2b.llm.vocab_size", tc.get("vocab_size", 262144))
    writer.add_uint32("gemma4e2b.llm.max_position_embeddings", tc.get("max_position_embeddings", 131072))
    writer.add_uint32("gemma4e2b.llm.sliding_window", tc.get("sliding_window", 512))
    writer.add_float32("gemma4e2b.llm.rope_theta", tc.get("rope_theta", 10000.0))
    writer.add_float32("gemma4e2b.llm.final_logit_softcapping", tc.get("final_logit_softcapping", 30.0))
    writer.add_float32("gemma4e2b.llm.rms_norm_eps", tc.get("rms_norm_eps", 1e-6))

    # Tokenizer: store BPE tokens from tokenizer.json
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path) as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            tokens = [""] * len(vocab)
            for token, idx in vocab.items():
                if idx < len(tokens):
                    tokens[idx] = token
            writer.add_token_list(tokens)
            merges = tok_data.get("model", {}).get("merges", [])
            if merges:
                writer.add_token_merges(merges)
            print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges")

    # Map and write tensors (streaming, one at a time)
    mapped = 0
    skipped = 0
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            skipped += 1
            continue

        data = handles[tensor_names[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        # Scalar tensors → 1-element F32
        if data.ndim == 0:
            data = np.array([data.item()], dtype=np.float32)
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        # 1D tensors (norms, biases, scales) → F32
        elif data.ndim <= 1:
            data = np.ascontiguousarray(data.astype(np.float32))
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        else:
            data = np.ascontiguousarray(data.astype(out_dtype))
            writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)

        mapped += 1
        if mapped % 100 == 0:
            print(f"  [{mapped}] {gguf_name}")

    print(f"\nMapped: {mapped}, Skipped: {skipped} (clipping scalars + vision)")
    print(f"Writing to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"Done! {outfile} ({size_mb:.1f} MB, {mapped} tensors)")


if __name__ == "__main__":
    main()
