# Server mode (HTTP API)

`crispasr --server` starts a persistent HTTP server with the model
loaded once and reused across requests. Compatible with the OpenAI
audio-transcription protocol, so any tool that already speaks
OpenAI's API (LiteLLM, LangChain, custom clients) can point at
CrispASR with zero code changes.

## Quick start

```bash
# Start server with model loaded once
crispasr --server -m model.gguf --port 8080

# Transcribe via HTTP (model stays loaded between requests):
curl -F "file=@audio.wav" http://localhost:8080/inference
# {"text": "...", "segments": [...], "backend": "parakeet", "duration": 11.0}

# Hot-swap to a different model at runtime:
curl -F "model=path/to/other-model.gguf" http://localhost:8080/load

# Check server status:
curl http://localhost:8080/health
# {"status": "ok", "backend": "parakeet"}

# List available backends:
curl http://localhost:8080/backends
# {"backends": ["whisper","parakeet","canary",...], "active": "parakeet"}
```

The server loads the model once at startup and keeps it in memory.
Subsequent `/inference` requests reuse the loaded model with no reload
overhead. Requests are mutex-serialized. Use `--host 0.0.0.0` to
accept remote connections.

## API keys

To require API keys, set the `CRISPASR_API_KEYS` env var
(comma-separated). **Do not** pass keys as CLI arguments — they would
be visible in `ps` / `top`. Protected endpoints accept either
`Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health`
remains public for container health checks.

```bash
CRISPASR_API_KEYS=key-one,key-two crispasr --server -m model.gguf

curl -H "Authorization: Bearer key-one" \
  -F "file=@audio.wav" \
  http://localhost:8080/v1/audio/transcriptions
```

## OpenAI-compatible endpoint

`POST /v1/audio/transcriptions` is a drop-in replacement for the
[OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

```bash
# Same curl syntax as the OpenAI API:
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=json"
# {"text": "And so, my fellow Americans, ask not what your country can do for you..."}

# Verbose JSON with per-segment timestamps (matches OpenAI's format):
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
# {"task": "transcribe", "language": "en", "duration": 11.0, "text": "...", "segments": [...]}

# SRT subtitles:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt"

# Plain text:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**Supported form fields:**

| Field | Description |
|---|---|
| `file` | Audio file (required) |
| `model` | Ignored (uses the loaded model) |
| `language` | ISO-639-1 code (default: server's `-l` setting) |
| `prompt` | Initial prompt / context |
| `response_format` | `json` (default), `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | Sampling temperature (default: 0.0) |

`GET /v1/models` returns an OpenAI-compatible model list with the
currently loaded model.

## Docker Compose

The repo includes a root-level
[`docker-compose.yml`](../docker-compose.yml) for running the
persistent HTTP server against a mounted model directory.

```bash
cp .env.example .env
# Edit CRISPASR_MODEL to point at a file mounted under ./models

docker compose up --build

# Health check
curl http://localhost:8080/health

# OpenAI-compatible transcription API
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

By default the compose stack:
- builds from `.devops/main.Dockerfile`
- mounts `./models` into `/models`
- stores auto-downloaded models in the Docker-managed
  `crispasr-cache` volume at `/cache`
- serves on `http://localhost:8080`

If you want `/cache` to be a host directory instead, replace the
`crispasr-cache:/cache` volume with `./cache:/cache` and make it
writable by the container user before startup:

```bash
mkdir -p cache models
sudo chown -R "$(id -u):$(id -g)" cache models
```

You can cap or raise build parallelism with `CRISPASR_BUILD_JOBS`:

```bash
docker compose build --build-arg CRISPASR_BUILD_JOBS=8
```

For CUDA builds, use the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

## Prebuilt CUDA images — choosing a tag

We publish two CUDA tags on `ghcr.io/crispstrobe/crispasr`. Pick the
one that matches your host driver:

| Tag | CUDA | Min NVIDIA driver | Supported arches | Notes |
|---|---|---|---|---|
| `main-cuda` | 13.0 | **R535+** (R580+ for full features) | sm_75…sm_120 incl. RTX 50xx (Blackwell) | Default. Pull this on modern hosts. |
| `main-cuda-12` | 12.4 | **R510+** | sm_75…sm_90 (RTX 20/30/40-series, Hopper) | Legacy compat — use on RHEL 7/8, older Ubuntu LTS, or any host that hasn't updated drivers in a while. RTX 50xx is **not** supported here. |

Quick check: `nvidia-smi` shows your driver version in the top-right.
If it's R535 or higher, pull `main-cuda`. If it's R510–R534, pull
`main-cuda-12`. If it's older than R510, update your driver — neither
image will work.

```bash
docker pull ghcr.io/crispstrobe/crispasr:main-cuda      # modern hosts
docker pull ghcr.io/crispstrobe/crispasr:main-cuda-12   # legacy driver
```

## Hugging Face Space wrapper

There is also a Gradio-based Hugging Face Space wrapper under
[`hf-space/`](../hf-space/README.md). It starts the CrispASR HTTP
server inside the container and provides a small browser UI on top of
the OpenAI-compatible transcription endpoint.

Build it locally with:

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_MODEL=/models/ggml-base.en.bin \
  -v "$PWD/models:/models" \
  crispasr-hf-space
```

The compose files default to local image tags (`crispasr-local:*`)
so they don't depend on pulling a published registry image first.

## Environment overrides

You can override the loaded model and startup flags through `.env`:

| Variable | Purpose |
|---|---|
| `CRISPASR_MODEL` | Model path inside the container (e.g. `/models/parakeet-tdt-0.6b-v2.gguf`) |
| `CRISPASR_BACKEND` | Force a specific backend |
| `CRISPASR_LANGUAGE` | ISO-639-1 code or `auto` for LID |
| `CRISPASR_AUTO_DOWNLOAD` | Set to `1` to enable `-m auto` resolution |
| `CRISPASR_CACHE_DIR` | Where auto-downloaded models live (defaults to `/cache`) |
| `CRISPASR_API_KEYS` | Comma-separated API keys (see [API keys](#api-keys)) |
| `CRISPASR_EXTRA_ARGS` | Forwarded verbatim to the server CLI (e.g. `--no-punctuation`) |

The service is configured to avoid serving as root by default:
- `user: "${CRISPASR_UID:-1000}:${CRISPASR_GID:-1000}"`
- `security_opt: ["no-new-privileges:true"]`
