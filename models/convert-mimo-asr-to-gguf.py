#!/usr/bin/env python3
"""Convert XiaomiMiMo/MiMo-V2.5-ASR to GGUF format for CrispASR.

Architecture: 6-layer input_local_transformer (audio token processor) +
36-layer Qwen2 LLM (4096d, 32 heads, 8 KV heads, SiLU, RoPE).
Audio input via 8-channel RVQ tokens from MiMo-Audio-Tokenizer (separate model).

Needs ~20 GB RAM (model is ~16 GB in BF16 across 7 safetensors shards).

Usage:
    python models/convert-mimo-asr-to-gguf.py --input XiaomiMiMo/MiMo-V2.5-ASR --output mimo-asr.gguf
"""

import argparse
import json
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
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(model_id,
                             allow_patterns=["*.safetensors", "config.json",
                                             "tokenizer.json", "tokenizer_config.json"])
    return Path(path)


def map_tensor_name(hf_name: str) -> str:
    """Map HF tensor name to GGUF name."""
    name = hf_name

    # Speech embeddings
    name = name.replace("speech_embeddings.", "audio.emb.")
    name = name.replace("speech_group_downcast.", "audio.group_proj.")
    name = name.replace("hidden_states_downcast.", "audio.hidden_proj.")

    # Input local transformer (audio processor, 6 layers)
    name = name.replace("input_local_transformer.layers.", "audio.blk.")
    name = name.replace("input_local_transformer.norm.", "audio.norm.")

    # Main LLM (Qwen2, 36 layers)
    name = name.replace("local_transformer.layers.", "llm.blk.")
    name = name.replace("local_transformer.norm.", "llm.norm.")
    name = name.replace("local_transformer_lm_heads.", "llm.codebook_head.")

    # Token embeddings + LM head
    name = name.replace("model.embed_tokens.", "llm.embed.")
    name = name.replace("model.norm.", "llm.final_norm.")
    name = name.replace("lm_head.", "llm.lm_head.")

    # Attention + MLP
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.o_proj.", ".attn.o.")
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert MiMo-V2.5-ASR to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    ac = config.get("audio_config", {})

    print(f"\nMiMo-V2.5-ASR")
    print(f"  LLM: {config['num_hidden_layers']}L, hidden={config['hidden_size']}, "
          f"heads={config['num_attention_heads']}, kv_heads={config['num_key_value_heads']}")
    print(f"  Audio: {ac.get('input_local_layers', 6)}L input transformer, "
          f"dim={ac.get('input_local_dim', 1024)}, {ac.get('audio_channels', 8)} codebook channels")
    print(f"  Vocab: {config['vocab_size']}")

    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    # Open safetensors (sharded)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model-*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Create GGUF
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "mimo_asr", use_temp_file=True)
    writer.add_name("MiMo-V2.5-ASR")

    # LLM params (Qwen2)
    writer.add_uint32("mimo_asr.llm.hidden_size", config["hidden_size"])
    writer.add_uint32("mimo_asr.llm.num_layers", config["num_hidden_layers"])
    writer.add_uint32("mimo_asr.llm.num_heads", config["num_attention_heads"])
    writer.add_uint32("mimo_asr.llm.num_kv_heads", config["num_key_value_heads"])
    writer.add_uint32("mimo_asr.llm.intermediate_size", config["intermediate_size"])
    writer.add_uint32("mimo_asr.llm.vocab_size", config["vocab_size"])
    writer.add_uint32("mimo_asr.llm.max_position_embeddings", config.get("max_position_embeddings", 8192))
    writer.add_float32("mimo_asr.llm.rope_theta", config.get("rope_theta", 640000))
    writer.add_float32("mimo_asr.llm.rms_norm_eps", config.get("rms_norm_eps", 1e-6))

    # Audio params
    writer.add_uint32("mimo_asr.audio.channels", ac.get("audio_channels", 8))
    writer.add_uint32("mimo_asr.audio.group_size", ac.get("group_size", 4))
    writer.add_uint32("mimo_asr.audio.input_layers", ac.get("input_local_layers", 6))
    writer.add_uint32("mimo_asr.audio.input_dim", ac.get("input_local_dim", 1024))
    writer.add_uint32("mimo_asr.audio.input_heads", ac.get("input_local_attn_heads", 64))
    writer.add_uint32("mimo_asr.audio.input_head_dim", ac.get("input_local_head_dim", 16))
    writer.add_uint32("mimo_asr.audio.input_intermediate", ac.get("input_local_intermediate_size", 4096))
    writer.add_uint32("mimo_asr.audio.out_hidden_size", ac.get("out_hidden_size", 4096))

    # Speech vocab sizes per channel
    speech_vocab = ac.get("speech_vocab_size", "1025-1025-129-129-129-129-129-129")
    for i, v in enumerate(speech_vocab.split("-")):
        writer.add_uint32(f"mimo_asr.audio.speech_vocab.{i}", int(v))

    # Tokenizer: BPE from tokenizer.json
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

    # Map and write tensors
    mapped = 0
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        data = handles[tensor_names[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        if data.ndim == 0:
            data = np.array([data.item()], dtype=np.float32)
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        elif data.ndim <= 1:
            data = np.ascontiguousarray(data.astype(np.float32))
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        else:
            data = np.ascontiguousarray(data.astype(out_dtype))
            writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)
        mapped += 1
        if mapped % 50 == 0:
            print(f"  [{mapped}] {gguf_name}")

    print(f"\nWriting {mapped} tensors to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = outfile.stat().st_size / 1024**3
    print(f"Done! {outfile} ({size_gb:.1f} GB, {mapped} tensors)")


if __name__ == "__main__":
    main()
