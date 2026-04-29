"""Run the official Microsoft VibeVoice-Realtime-0.5B and dump per-frame intermediates.

Captures (via `tools/reference_backends/_iter_capture.py`):
  - pos_cond / neg_cond — TTS-LM hidden states fed into each diffusion call
  - noise               — initial z per frame (the trajectory's first row)
  - v_cfg_step0         — CFG-mixed eps at step 0 of every frame's diffusion
  - latent              — final z per frame
  - acoustic_embed      — connector output per frame

Output is `<output_dir>/perframe_<stage>_f<NNN>.bin` (float32) plus
`noise.bin` for the C++ side to replay via VIBEVOICE_TTS_NOISE.
The C++ runtime writes the same filenames when started with
VIBEVOICE_TTS_DUMP_PERFRAME=1, so the two dump dirs diff directly.

Usage:
  python tools/run_official_vibevoice.py \\
      --text "It was in the summer of '89 ..." \\
      --voice .local/issue39/voices_pt/en-Davis_man.pt \\
      --output-wav .local/issue39/issue39_OFFICIAL_davis.wav \\
      --output-dir .local/issue39/ref_dump_davis
"""
import argparse
import copy
import sys
import wave
from pathlib import Path

import numpy as np
import torch

# trust_remote_code path: the streaming model isn't a vanilla transformers arch.
import vibevoice  # noqa
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

# Local helpers (sibling directory)
sys.path.insert(0, str(Path(__file__).parent))
from reference_backends import _hooks, _iter_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--output-wav", required=True)
    ap.add_argument("--output-dir", default=None,
                    help="Directory for per-frame .bin dumps (default: skip)")
    ap.add_argument("--model", default="microsoft/VibeVoice-Realtime-0.5B")
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    device = "cpu"
    print(f"loading model: {args.model}")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model, dtype=torch.float32, device_map=device,
    )
    model.eval()

    print(f"loading voice: {args.voice}")
    all_prefilled_outputs = torch.load(args.voice, map_location=device, weights_only=False)
    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained("Qwen/Qwen2.5-0.5B")

    text_with_nl = args.text + "\n"
    text_ids = tokenizer.encode(text_with_nl, add_special_tokens=False)
    print(f"  tokens: {len(text_ids)}: {text_ids}")
    tts_text_ids = torch.tensor([text_ids], dtype=torch.long, device=device)

    lm_seq = all_prefilled_outputs["lm"].past_key_values.key_cache[0].shape[2]
    tts_seq = all_prefilled_outputs["tts_lm"].past_key_values.key_cache[0].shape[2]
    print(f"  lm_seq_len: {lm_seq}, tts_seq_len: {tts_seq}")

    input_ids = torch.zeros((1, lm_seq), dtype=torch.long, device=device)
    tts_lm_input_ids = torch.zeros((1, tts_seq), dtype=torch.long, device=device)

    # ── per-frame capture ─────────────────────────────────────────────────
    # Two capture sites:
    #   1. monkey-patch sample_speech_tokens (pos/neg cond + noise + v_cfg + latent)
    #   2. forward_hook on acoustic_connector (acoustic_embed)
    # _iter_capture.append normalises detach/cpu/clone the same way for both.
    captured = {}
    handles = []

    def make_hooked_sample(_orig):
        @torch.no_grad()
        def hooked_sample(condition, neg_condition, cfg_scale=3.0):
            _iter_capture.append(captured, "pos_cond", condition[0])
            _iter_capture.append(captured, "neg_cond", neg_condition[0])
            model.model.noise_scheduler.set_timesteps(model.ddpm_inference_steps)
            condition_pair = torch.cat([condition, neg_condition], dim=0).to(
                model.model.prediction_head.device)
            speech = torch.randn(condition_pair.shape[0], model.config.acoustic_vae_dim).to(condition_pair)
            _iter_capture.append(captured, "noise", speech[0])
            first_step = True
            for t in model.model.noise_scheduler.timesteps:
                half = speech[: len(speech) // 2]
                combined = torch.cat([half, half], dim=0)
                eps = model.model.prediction_head(
                    combined, t.repeat(combined.shape[0]).to(combined), condition=condition_pair,
                )
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                if first_step:
                    _iter_capture.append(captured, "v_cfg_step0", half_eps[0])
                    first_step = False
                eps_full = torch.cat([half_eps, half_eps], dim=0)
                speech = model.model.noise_scheduler.step(eps_full, t, speech).prev_sample
            final = speech[: len(speech) // 2]
            _iter_capture.append(captured, "latent", final[0])
            return final

        return hooked_sample

    handles.append(_iter_capture.patch_method(model, "sample_speech_tokens", make_hooked_sample))

    # acoustic_connector is a regular nn.Module; use a forward_hook that
    # appends per-call (= per speech frame).
    def hook_connector(_module, _inp, out_t):
        _iter_capture.append(captured, "acoustic_embed", out_t[0, 0, :])
    handles.append(model.model.acoustic_connector.register_forward_hook(hook_connector))

    print("running generate()...")
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        tts_lm_input_ids=tts_lm_input_ids,
        tts_lm_attention_mask=torch.ones_like(tts_lm_input_ids),
        tts_text_ids=tts_text_ids,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
        cfg_scale=args.cfg_scale,
        tokenizer=tokenizer,
        return_speech=True,
        max_new_tokens=1024,
        show_progress_bar=False,
    )
    _iter_capture.drop_handles(handles)

    # ── audio out ─────────────────────────────────────────────────────────
    audio = output.speech_outputs[0].detach().cpu().float().squeeze().numpy()
    sr = 24000
    samples = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(args.output_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    print(f"  audio: {len(samples)} samples = {len(samples)/sr:.2f}s -> {args.output_wav}")

    # ── per-frame dumps ───────────────────────────────────────────────────
    if out_dir is not None:
        n = _iter_capture.dump_perframe_dir(captured, out_dir)
        counts = {k: len(v) for k, v in captured.items()}
        print("  captured frames: " + " ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"  wrote {n} per-frame .bin files to {out_dir} (+ noise.bin)")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
