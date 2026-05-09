#!/usr/bin/env python3
"""IndexTTS-1.5 reference inference — dump intermediates for diff-testing.

Usage:
  python tools/reference_backends/indextts.py \
      --model-dir /path/to/IndexTTS-1.5 \
      --text "Hello world." \
      --output-dir /tmp/indextts-ref/
"""

import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, help='IndexTTS-1.5 model directory')
    parser.add_argument('--text', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ref-audio', default=None, help='Reference WAV for voice cloning')
    parser.add_argument('--max-mel-tokens', type=int, default=1500)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = args.model_dir

    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(model_dir, 'bpe.model'))

    # Tokenize
    text_tokens = sp.Encode(args.text)
    print(f'Text: {repr(args.text)} -> {len(text_tokens)} tokens: {text_tokens}')
    np.save(os.path.join(args.output_dir, 'text_tokens.npy'), np.array(text_tokens, dtype=np.int32))

    # Load GPT checkpoint
    ckpt = torch.load(os.path.join(model_dir, 'gpt.pth'), map_location='cpu', weights_only=False)
    sd = ckpt['model']
    print(f'Loaded {len(sd)} tensors from gpt.pth')

    # Hyperparams
    D = 1280
    n_heads = 20
    n_layers = 24
    head_dim = D // n_heads
    n_mel_codes = 8194
    start_mel = 8192
    stop_mel = 8193

    # Build conditioning latents (zeros for now — no reference audio)
    cond_latents = torch.zeros(1, 32, D)
    np.save(os.path.join(args.output_dir, 'cond_latents.npy'), cond_latents.numpy())

    # Text embeddings + positional
    text_emb = sd['text_embedding.weight']  # [12001, 1280]
    text_pos = sd['text_pos_embedding.emb.weight']  # [602, 1280]
    mel_emb = sd['mel_embedding.weight']  # [8194, 1280]
    mel_pos = sd['mel_pos_embedding.emb.weight']  # [803, 1280]

    # Build prefix embeddings: [cond_latents | text_embs+pos | start_mel+pos]
    text_ids = torch.tensor(text_tokens, dtype=torch.long)
    text_embedded = text_emb[text_ids] + text_pos[:len(text_tokens)]
    start_mel_embedded = mel_emb[start_mel:start_mel+1] + mel_pos[0:1]

    prefix = torch.cat([cond_latents[0], text_embedded, start_mel_embedded], dim=0)  # [T, D]
    print(f'Prefix: {prefix.shape} (cond=32 + text={len(text_tokens)} + start_mel=1)')
    np.save(os.path.join(args.output_dir, 'prefix_embeds.npy'), prefix.numpy())

    # GPT-2 forward pass on prefix
    # Using manual implementation (no HF dependency)
    # Cast all weights to float32
    sd = {k: v.float() for k, v in sd.items()}
    x = prefix.unsqueeze(0).float()  # [1, T, D]
    T = x.shape[1]

    # Causal mask
    mask = torch.full((T, T), float('-inf'))
    mask = torch.triu(mask, diagonal=1)

    for il in range(n_layers):
        p = f'gpt.h.{il}'
        residual = x

        # Pre-attention LN
        ln1_w = sd[f'{p}.ln_1.weight']
        ln1_b = sd[f'{p}.ln_1.bias']
        h = F.layer_norm(x, [D], ln1_w, ln1_b)

        # QKV (Conv1D: weight is [in, out])
        c_attn_w = sd[f'{p}.attn.c_attn.weight']  # [1280, 3840]
        c_attn_b = sd[f'{p}.attn.c_attn.bias']     # [3840]
        qkv = h @ c_attn_w + c_attn_b  # [1, T, 3840]
        Q, K, V = qkv.chunk(3, dim=-1)

        # Multi-head attention
        Q = Q.view(1, T, n_heads, head_dim).transpose(1, 2)
        K = K.view(1, T, n_heads, head_dim).transpose(1, 2)
        V = V.view(1, T, n_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, D)

        c_proj_w = sd[f'{p}.attn.c_proj.weight']
        c_proj_b = sd[f'{p}.attn.c_proj.bias']
        attn_out = attn_out @ c_proj_w + c_proj_b

        x = residual + attn_out

        # Pre-FFN LN
        residual = x
        ln2_w = sd[f'{p}.ln_2.weight']
        ln2_b = sd[f'{p}.ln_2.bias']
        h = F.layer_norm(x, [D], ln2_w, ln2_b)

        # FFN (Conv1D)
        fc_w = sd[f'{p}.mlp.c_fc.weight']
        fc_b = sd[f'{p}.mlp.c_fc.bias']
        proj_w = sd[f'{p}.mlp.c_proj.weight']
        proj_b = sd[f'{p}.mlp.c_proj.bias']

        mlp = F.gelu(h @ fc_w + fc_b, approximate='tanh')
        mlp = mlp @ proj_w + proj_b

        x = residual + mlp

        if il == 0 or il == n_layers - 1:
            np.save(os.path.join(args.output_dir, f'gpt_layer_{il}.npy'),
                    x[0].detach().numpy())

    # Final norm
    fn_w = sd['final_norm.weight']
    fn_b = sd['final_norm.bias']
    x = F.layer_norm(x, [D], fn_w, fn_b)

    # Logits from last position
    mel_head_w = sd['mel_head.weight']  # [8194, 1280]
    mel_head_b = sd['mel_head.bias']    # [8194]
    last_hidden = x[0, -1]  # [D]
    logits = last_hidden @ mel_head_w.t() + mel_head_b  # [8194]

    np.save(os.path.join(args.output_dir, 'prefill_logits.npy'), logits.detach().numpy())
    top5 = logits.topk(5)
    print(f'Prefill logits top5: ids={top5.indices.tolist()} vals={[f"{v:.3f}" for v in top5.values.tolist()]}')

    print(f'\nDumped to {args.output_dir}')


if __name__ == '__main__':
    main()
