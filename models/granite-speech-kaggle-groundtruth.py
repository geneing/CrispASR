#!/usr/bin/env python3
"""
Granite Speech 4.0-1B ground truth dump — for Kaggle (16GB+ RAM).
Dumps intermediate activations for comparing with C++ runtime.
"""

import json, math, os, subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.50")
install("safetensors")
install("scipy")
install("huggingface_hub")
install("torchaudio")
install("soundfile")

import numpy as np
import torch
import requests

GH_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except: GH_TOKEN = os.environ.get("GH_TOKEN")

# Download model + audio
from huggingface_hub import snapshot_download

# Use HF token for faster downloads (add HF_TOKEN secret in Kaggle)
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except: hf_token = os.environ.get("HF_TOKEN")

model_dir = snapshot_download("ibm-granite/granite-4.0-1b-speech", local_dir="/tmp/granite-speech",
                               token=hf_token)
audio_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
audio_path = "/tmp/jfk.wav"
if not os.path.exists(audio_path):
    import urllib.request
    urllib.request.urlretrieve(audio_url, audio_path)

import scipy.io.wavfile as wavfile
sr, data = wavfile.read(audio_path)
audio = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
print(f"Audio: {len(audio)} samples, {len(audio)/16000:.2f}s")

results = {}

def to_np(x):
    if hasattr(x, 'cpu'): x = x.cpu()
    if hasattr(x, 'detach'): x = x.detach()
    if hasattr(x, 'float'): x = x.float()
    if hasattr(x, 'numpy'): x = x.numpy()
    return np.asarray(x, dtype=np.float32)

def save(name, arr, desc=""):
    arr = to_np(arr)
    results[name] = {
        "shape": list(arr.shape), "desc": desc,
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "first_8": arr.flatten()[:8].tolist(),
    }
    np.save(f"/tmp/granite-{name}.npy", arr)
    print(f"  {name}: shape={arr.shape} min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")

# Load model
print("\nLoading model...")
from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
processor = AutoProcessor.from_pretrained(model_dir)
model = GraniteSpeechForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")
model.eval()
print("Model loaded!")

# Process audio through the processor
print("\n=== Processor output ===")
# GraniteSpeechProcessor expects text + audio separately, not conversation dicts
import soundfile as sf
audio_sf, sr_sf = sf.read(audio_path, dtype="float32")
if sr_sf != 16000:
    print(f"  WARNING: audio is {sr_sf} Hz, processor expects 16000 Hz")

# Build prompt text with audio placeholder
prompt_text = "USER: <|audio|> Transcribe the audio.\n ASSISTANT:"
try:
    inputs = processor(text=prompt_text, audio=audio_sf, return_tensors="pt")
except Exception as e:
    print(f"  Processor with text+audio failed: {e}")
    # Try alternative: pass audio as list
    try:
        inputs = processor(text=[prompt_text], audio=[audio_sf], return_tensors="pt")
    except Exception as e2:
        print(f"  Alternative also failed: {e2}")
        # Last resort: build inputs manually
        print("  Building inputs manually...")
        fe = processor.feature_extractor if hasattr(processor, 'feature_extractor') else processor.audio_processor
        mel_out = fe(audio_sf, return_tensors="pt")
        tok_out = processor.tokenizer(prompt_text, return_tensors="pt")
        inputs = {**mel_out, **tok_out}
print(f"Input keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape} {v.dtype}")

if "input_features" in inputs:
    save("input_features", inputs["input_features"][0], "mel features from processor")
if "input_ids" in inputs:
    ids = inputs["input_ids"][0].tolist()
    save("input_ids", np.array(ids[:50], dtype=np.float32), f"first 50 input_ids: {ids[:50]}")
    n_audio = sum(1 for i in ids if i == 100352)
    print(f"  audio tokens in prompt: {n_audio}")
    print(f"  total tokens: {len(ids)}")

# Encoder output
print("\n=== Encoder output ===")
with torch.no_grad():
    # Run just the encoder
    feats = inputs.get("input_features", inputs.get("input_values"))
    if feats is not None:
        enc_out = model.encoder(feats)
        save("encoder_out", enc_out.last_hidden_state[0], f"encoder output")

        # Projector output
        print("\n=== Projector output ===")
        proj_out = model.projector(enc_out.last_hidden_state)
        save("projector_out", proj_out[0], f"projector output (audio tokens for LLM)")

# Full generation
print("\n=== Generation ===")
try:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        gen_ids = outputs[0].tolist()
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        save("gen_ids", np.array(gen_ids[:50], dtype=np.float32), f"first 50 gen_ids")
        results["gen_text"] = decoded[0] if decoded else ""
        print(f"  Generated: {decoded[0]!r}")
except Exception as e:
    print(f"  Generation failed: {e}")
    import traceback; traceback.print_exc()

# Upload to gist
print("\n=== Uploading ===")
summary = json.dumps(results, indent=2, default=str)
if GH_TOKEN:
    gist_files = {"granite-speech-groundtruth.json": {"content": summary}}
    resp = requests.post("https://api.github.com/gists",
        headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
        json={"description": "Granite Speech 4.0-1B ground truth (jfk.wav)", "public": False, "files": gist_files})
    if resp.status_code == 201:
        print(f"  GIST: {resp.json()['html_url']}")
    else:
        print(f"  Gist failed: {resp.status_code}")
else:
    with open("/tmp/granite-groundtruth.json", "w") as f: f.write(summary)
    print("  Saved to /tmp/granite-groundtruth.json")

print("\nDone!")
