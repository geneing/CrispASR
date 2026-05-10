# Streaming & live transcription

CrispASR supports three streaming modes — pipe input, microphone
capture, and continuous live mode — and per-token confidence output.
All work with every supported backend.

## Pipe mode (`--stream`)

```bash
# Pipe audio from ffmpeg, sox, or any tool that outputs raw PCM:
ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | \
    crispasr --stream -m model.gguf
```

Sliding-window chunking, default 10 s window with 3 s step and 200 ms
overlap. Tune via `--stream-step`, `--stream-length`, `--stream-keep`.

Quality-control flags supported in streaming mode:

- `--vad`, `--vad-model`, `--vad-threshold`, `--vad-min-speech-duration-ms`, `--vad-min-silence-duration-ms`, `--vad-speech-pad-ms`
- `--punc-model` and `--no-punctuation`

Notes:

- With VAD enabled, each streaming window is segmented before ASR. Silent windows are skipped instead of being decoded.
- `--punc-model` applies after streamed chunk transcription, matching file-mode post-processing.
- `--alt` / `--alt-n` are file-mode features. They currently do not print token alternatives from `--stream`, `--mic`, or `--live`.
- File-oriented output flags such as `-osrt`, `-ovtt`, `-oj`, and `-of` do not apply to `--stream` / `--mic` / `--live`; streaming writes transcripts to stdout (or JSON-Lines events with `--stream-json`).

## Structured streaming output (`--stream-json`)

For wrappers (browser bridges, live-translation pipelines, captioning
UIs) that need to distinguish a still-evolving partial from a
finalized utterance, pass `--stream-json`. CrispASR then emits one
JSON object per line on stdout — never plain text — and FireRed VAD
diagnostics stay off stderr unless you opt in with
`--firered-vad-debug`.

```bash
ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream --stream-json -m model.gguf \
      --vad --vad-model firered-vad.gguf \
      --stream-final-on-silence-ms 800
```

Event types:

| `type` | When | Fields |
|---|---|---|
| `partial` | A streaming step produced new text. The same `utterance_id` repeats with updated `text` until the utterance is finalized. | `utterance_id`, `text`, `t0`, `t1` |
| `final` | Trailing silence ≥ `--stream-final-on-silence-ms` (default `800`) closed the open utterance. Echoes the last `text`. | `utterance_id`, `text`, `t0`, `t1` |
| `silence` | A streaming step produced no text and no utterance is open (or the silence threshold has not yet been reached). | `t` |

Sample stream:

```json
{"type":"partial","utterance_id":1,"text":"is that they can be tuned and adjusted","t0":10.20,"t1":13.20}
{"type":"partial","utterance_id":1,"text":"is that they can be tuned and adjusted for a specific","t0":10.20,"t1":16.20}
{"type":"final","utterance_id":1,"text":"is that they can be tuned and adjusted for a specific hardware target.","t0":10.20,"t1":17.80}
{"type":"silence","t":18.60}
```

Live-translation wrappers can show `partial` events in a draft pane
and only ship `final` events to the translation API. Set
`--stream-final-on-silence-ms 0` to disable auto-finalization (useful
when the wrapper finalizes on its own signal — e.g., a UI button —
instead of trailing silence).

`t0` / `t1` are wall-clock seconds since stream start, derived from
the cumulative sample count, so they map to the same timeline as the
input PCM.

## Microphone (`--mic`)

```bash
# Live microphone transcription (auto-detects arecord/sox/ffmpeg):
crispasr --mic -m model.gguf
```

CrispASR auto-detects whichever audio capture tool is on `$PATH`.

## Continuous live mode (`--live`)

```bash
# Continuous live mode (prints each chunk as a new line, never stops):
crispasr --live -m model.gguf

# With progress monitor symbols (▶ processing, ✓ got text, · silence):
crispasr --live --monitor -m model.gguf
```

`--live` runs indefinitely, emitting one transcript line per processed
chunk. `--monitor` adds visual feedback so you can tell processing
state at a glance.

## Per-token confidence

```bash
crispasr -m model.gguf -f audio.wav --alt
```

`--alt` prints alternative candidate tokens with probabilities — useful
for filtering low-confidence file transcriptions or for downstream
rescoring. Streaming modes do not currently emit this alternatives
block.

## Tuning the sliding window

| Flag | Default | Effect |
|---|---|---|
| `--stream-step N` | `3000` ms | Step between consecutive windows. Smaller = more frequent partial transcripts. |
| `--stream-length N` | `10000` ms | Rolling context window cap. The decode buffer accumulates audio up to this many ms, then drops the oldest samples from the front. Larger = better accuracy on long-form content but higher per-step cost. |
| `--stream-keep N` | `200` ms | Legacy — kept for compatibility, currently a no-op. The rolling buffer above subsumes it (see issue #84). |

> **Note (issue #84).** Before May 2026, `--stream-length` was a
> *ceiling* on `keep + step` rather than a true rolling cap, so
> `--stream-length 18000 --stream-keep 200 --stream-step 3000`
> actually decoded ~3.4 s of audio per step instead of 18 s. The
> streaming loop was rewritten to accumulate up to `length_samples`
> and drop the oldest frame on overflow, which matches the documented
> behaviour. `--stream-keep` is now informational only.

For native streaming-architecture backends (`voxtral4b`,
`moonshine-streaming`, `kyutai-stt`), the encoder runs incrementally —
the sliding window flags above still apply but the per-chunk cost is
lower than for batch backends.
