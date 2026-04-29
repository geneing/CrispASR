"""Run the official Microsoft VibeVoice-Realtime-0.5B and dump per-frame intermediates.

Hooks the official VibeVoiceStreamingForConditionalGenerationInference to capture:
  - per-frame positive_condition (TTS-LM last hidden before each diffusion call)
  - per-frame negative_condition
  - per-frame initial noise (z at step 0 of diffusion)
  - per-frame v_cfg at step 0 (CFG-mixed eps)
  - per-frame final speech latent (z after all diffusion steps)
  - per-frame acoustic_embed (connector output)
  - final audio WAV (24 kHz mono)

Outputs to <output_dir>/perframe_<stage>_f<NNN>.bin for cross-comparison with the C++
runtime when launched with VIBEVOICE_TTS_DUMP_PERFRAME=1 and the same VIBEVOICE_TTS_NOISE.

Usage:
  python tools/run_official_vibevoice.py \\
      --text "It was in the summer of '89 ..." \\
      --voice .local/issue39/voices_pt/en-Davis_man.pt \\
      --output-wav .local/issue39/issue39_OFFICIAL_davis.wav \\
      --output-dir .local/issue39/ref_dump_davis
"""
import argparse, copy, sys, os
from pathlib import Path
import numpy as np
import torch

# Force VibeVoice to be importable so trust_remote_code finds it
import vibevoice  # noqa
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast


def dump_f32(path, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(str(path))


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
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"loading model: {args.model}")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model, dtype=torch.float32, device_map=device
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

    # ── per-frame capture hooks ──────────────────────────────────────────────
    # Capture conditions at the EXACT point sample_speech_tokens is called: the official's
    # generate() reads positive/negative conditions from tts_lm_outputs.last_hidden_state[-1]
    # right before each diffusion call (line 773-774). Patching sample_speech_tokens to
    # snapshot its inputs is the cleanest way.
    captured = {
        "pos_cond": [],           # positive condition fed into diffusion, per frame
        "neg_cond": [],           # negative condition fed into diffusion, per frame
        "noise": [],              # initial z per frame (vae_dim, the trajectory's first row)
        "v_cfg_step0": [],        # CFG'd eps at step 0 per frame (vae_dim)
        "latent": [],             # final z per frame
        "acoustic_embed": [],     # connector output per frame
    }

    @torch.no_grad()
    def hooked_sample(condition, neg_condition, cfg_scale=3.0):
        # Snapshot conditions BEFORE running diffusion
        captured["pos_cond"].append(condition[0].detach().cpu().clone())
        captured["neg_cond"].append(neg_condition[0].detach().cpu().clone())

        model.model.noise_scheduler.set_timesteps(model.ddpm_inference_steps)
        condition_pair = torch.cat([condition, neg_condition], dim=0).to(model.model.prediction_head.device)
        speech = torch.randn(condition_pair.shape[0], model.config.acoustic_vae_dim).to(condition_pair)
        captured["noise"].append(speech[0].detach().cpu().clone())
        first_step = True
        for t in model.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = model.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined), condition=condition_pair
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            if first_step:
                captured["v_cfg_step0"].append(half_eps[0].detach().cpu().clone())
                first_step = False
            eps_full = torch.cat([half_eps, half_eps], dim=0)
            speech = model.model.noise_scheduler.step(eps_full, t, speech).prev_sample
        final = speech[: len(speech) // 2]
        captured["latent"].append(final[0].detach().cpu().clone())
        return final
    model.sample_speech_tokens = hooked_sample

    # Hook acoustic_connector to capture per-frame embed
    def hook_connector(module, inp, out_t):
        captured["acoustic_embed"].append(out_t[0, 0, :].detach().cpu().clone())
    model.model.acoustic_connector.register_forward_hook(hook_connector)

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

    audio = output.speech_outputs[0].detach().cpu().float().squeeze().numpy()
    import wave
    sr = 24000
    audio_clip = np.clip(audio, -1, 1)
    samples = (audio_clip * 32767).astype(np.int16)
    with wave.open(args.output_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    print(f"  audio: {len(samples)} samples = {len(samples)/sr:.2f}s -> {args.output_wav}")

    # Per-frame dumps
    if out_dir is not None:
        for stage in captured:
            n = len(captured[stage])
            for i in range(n):
                dump_f32(out_dir / f"perframe_{stage}_f{i:03d}.bin", captured[stage][i])
        # Pin a single noise.bin for the C++ side to replay
        if captured["noise"]:
            stacked = torch.stack(captured["noise"])
            dump_f32(out_dir / "noise.bin", stacked)
        # Counts summary
        print(f"  captured frames: pos_cond={len(captured['pos_cond'])} "
              f"neg_cond={len(captured['neg_cond'])} noise={len(captured['noise'])} "
              f"v_cfg_step0={len(captured['v_cfg_step0'])} latent={len(captured['latent'])} "
              f"acoustic_embed={len(captured['acoustic_embed'])}")
        print(f"  dumps in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
