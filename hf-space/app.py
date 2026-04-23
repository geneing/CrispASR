import json
import os
import time
from pathlib import Path

import gradio as gr
import requests


SERVER_URL = os.environ.get("CRISPASR_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
SPACE_TITLE = os.environ.get("CRISPASR_SPACE_TITLE", "CrispASR")
DEFAULT_LANGUAGE = os.environ.get("CRISPASR_LANGUAGE", "en")
DEFAULT_MODEL = os.environ.get("CRISPASR_MODEL", "auto")
API_KEY = next((key.strip() for key in os.environ.get("CRISPASR_API_KEYS", "").split(",") if key.strip()), "")

MODEL_CHOICES = {
    "Whisper base multilingual (~147 MB)": ("whisper", "auto", "en"),
    "Parakeet TDT 0.6B v3 Q4_K (~467 MB)": ("parakeet", "auto", "en"),
    "Qwen3 ASR 0.6B Q4_K (~500 MB)": ("qwen3", "auto", "en"),
    "Cohere Transcribe Q4_K (~550 MB)": ("cohere", "auto", "en"),
}


def log(message: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] hf-space-app: {message}", flush=True)


def _request(method: str, path: str, **kwargs):
    if API_KEY:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Authorization", f"Bearer {API_KEY}")
        kwargs["headers"] = headers
    return requests.request(method, f"{SERVER_URL}{path}", timeout=300, **kwargs)


def fetch_status():
    try:
        log("fetch_status: probing /health and /v1/models")
        health = _request("GET", "/health")
        health.raise_for_status()
        models = _request("GET", "/v1/models")
        models.raise_for_status()
        health_json = health.json()
        models_json = models.json()
        model_names = [item.get("id", "") for item in models_json.get("data", [])]
        log(f"fetch_status: ready models={model_names if model_names else ['(none)']}")
        return (
            "ready",
            json.dumps(health_json, indent=2, ensure_ascii=False),
            "\n".join(model_names) if model_names else "(no models reported)",
        )
    except Exception as exc:
        log(f"fetch_status: waiting error={type(exc).__name__}: {exc}")
        return "starting", f"{type(exc).__name__}: {exc}", DEFAULT_MODEL


def wait_for_server():
    log("wait_for_server: start")
    last_status = "starting"
    last_health = ""
    last_models = DEFAULT_MODEL
    for i in range(300):
        last_status, last_health, last_models = fetch_status()
        if last_status == "ready":
            log(f"wait_for_server: ready after {i + 1} probe(s)")
            break
        time.sleep(1)
    if last_status != "ready":
        log("wait_for_server: timeout, app staying up in starting state")
    return last_status, last_health, last_models


def transcribe(audio_path: str, language: str, prompt: str, temperature: float, response_format: str):
    if not audio_path:
        raise gr.Error("Upload or record audio first.")

    file_path = Path(audio_path)
    if not file_path.exists():
        raise gr.Error("Audio file is not available anymore.")

    log(
        f"transcribe: file={file_path.name} language={language or 'default'} "
        f"response_format={response_format} temperature={temperature:.2f} prompt={'yes' if prompt else 'no'}"
    )

    data = {
        "model": "loaded-model",
        "response_format": response_format,
        "temperature": f"{temperature:.2f}",
    }

    if language and language != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    with file_path.open("rb") as f:
        response = _request(
            "POST",
            "/v1/audio/transcriptions",
            files={"file": (file_path.name, f, "application/octet-stream")},
            data=data,
        )

    if response.status_code >= 400:
        log(f"transcribe: error status={response.status_code} body={response.text[:400]}")
        raise gr.Error(f"{response.status_code}: {response.text}")

    content_type = response.headers.get("content-type", "")
    log(f"transcribe: ok status={response.status_code} content_type={content_type}")
    if response_format == "verbose_json" or "application/json" in content_type:
        payload = response.json()
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        log(f"transcribe: json text_len={len(text)}")
        return text, json.dumps(payload, indent=2, ensure_ascii=False)

    text = response.text.strip()
    log(f"transcribe: text text_len={len(text)}")
    return text, text


def load_model(choice: str, language: str):
    backend, model, default_language = MODEL_CHOICES.get(choice, MODEL_CHOICES["Whisper base multilingual (~147 MB)"])
    language = language or default_language
    log(f"load_model: choice={choice} backend={backend} model={model} language={language}")
    response = _request(
        "POST",
        "/load",
        files={
            "backend": (None, backend),
            "model": (None, model),
            "language": (None, language),
        },
    )
    if response.status_code >= 400:
        log(f"load_model: error status={response.status_code} body={response.text[:400]}")
        raise gr.Error(f"{response.status_code}: {response.text}")
    status, health, models = fetch_status()
    log(f"load_model: ok backend={backend}")
    return status, health, models, language


with gr.Blocks(title=SPACE_TITLE) as demo:
    gr.Markdown(
        f"""# {SPACE_TITLE}

Offline speech transcription via CrispASR's OpenAI-compatible server.

- Server URL: `{SERVER_URL}`
- Model path: `{DEFAULT_MODEL}`
"""
    )

    with gr.Row():
        status = gr.Textbox(label="Server status", interactive=False)
        models = gr.Textbox(label="Loaded model(s)", interactive=False)
    health = gr.Code(label="/health", language="json", interactive=False)
    refresh = gr.Button("Refresh server status")

    with gr.Row():
        model_choice = gr.Dropdown(list(MODEL_CHOICES.keys()), value="Whisper base multilingual (~147 MB)", label="Model")
        load = gr.Button("Load selected model")

    with gr.Row():
        audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
        with gr.Column():
            language = gr.Textbox(value=DEFAULT_LANGUAGE, label="Language", placeholder="auto or ISO-639-1 code")
            response_format = gr.Dropdown(
                ["text", "verbose_json"], value="verbose_json", label="Response format"
            )
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            prompt = gr.Textbox(label="Prompt", placeholder="Optional prompt or context")
            submit = gr.Button("Transcribe", variant="primary")

    transcript = gr.Textbox(label="Transcript", lines=12)
    raw = gr.Code(label="Raw response", language="json")

    refresh.click(fetch_status, outputs=[status, health, models])
    load.click(load_model, inputs=[model_choice, language], outputs=[status, health, models, language])
    submit.click(
        transcribe,
        inputs=[audio, language, prompt, temperature, response_format],
        outputs=[transcript, raw],
    )
    demo.load(wait_for_server, outputs=[status, health, models])


if __name__ == "__main__":
    log(f"launch: server_url={SERVER_URL} default_model={DEFAULT_MODEL} default_language={DEFAULT_LANGUAGE}")
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
