---
title: CrispASR
sdk: docker
app_port: 7860
pinned: false
---

# CrispASR Space

This folder contains a Hugging Face Docker Space wrapper for CrispASR:

- a Gradio UI for upload / microphone transcription
- a local CrispASR server started inside the same container
- OpenAI-compatible transcription requests routed to `http://127.0.0.1:8080/v1/audio/transcriptions`

## Environment variables

- `CRISPASR_MODEL=/models/model.gguf`
- `CRISPASR_BACKEND=whisper` or another backend name
- `CRISPASR_LANGUAGE=en`
- `CRISPASR_AUTO_DOWNLOAD=1`
- `CRISPASR_CACHE_DIR=/cache`
- `CRISPASR_API_KEYS=` optional comma-separated API keys for the local CrispASR server
- `CRISPASR_EXTRA_ARGS=`

## Local build

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_MODEL=/models/ggml-base.en.bin \
  -v "$PWD/models:/models" \
  crispasr-hf-space
```

For auto-downloads, mount a writable cache volume if you want models to survive container restarts:

```bash
docker volume create crispasr-cache
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_AUTO_DOWNLOAD=1 \
  -v crispasr-cache:/cache \
  crispasr-hf-space
```

Build parallelism can be tuned with `--build-arg CRISPASR_BUILD_JOBS=8`.
