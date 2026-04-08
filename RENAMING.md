# Renaming proposal

Right now this fork is called `cohere-whisper.cpp` because it started as a
Cohere Transcribe port of `whisper.cpp`. It has since grown into a hub for
**five distinct ASR runtimes** (`cohere-main`, `parakeet-main`, `canary-main`,
`cohere-align`, `canary-align`) plus a wav2vec2-CTC aligner. The `cohere`
prefix in the repo name is misleading — only one of the five runtimes has
anything to do with Cohere.

This document collects the renaming options and the trade-offs, so we can
decide once cleanly rather than re-litigating it in every session.

## Status quo (what we have today)

| Layer | Current name | Note |
| --- | --- | --- |
| GitHub repo | `CrispStrobe/cohere-whisper.cpp` (`parakeet` branch) | Misleading: 4/5 runtimes have nothing to do with Cohere |
| Code library | `src/cohere.{h,cpp}`, `src/parakeet.{h,cpp}`, `src/canary.{h,cpp}`, `src/canary_ctc.{h,cpp}`, `src/wav2vec2-ggml.{h,cpp}`, `src/align.{h,cpp}` | OK as-is — model-specific |
| CLI binaries | `cohere-main`, `cohere-align`, `parakeet-main`, `canary-main`, `canary-align` | Mostly OK; `canary-align` is misleading (the tool is universal, only the model came from canary's .nemo) |
| HF model repos | `cstr/cohere-transcribe-03-2026-GGUF`, `cstr/parakeet-tdt-0.6b-v3-GGUF`, `cstr/parakeet_de_med-GGUF`, `cstr/canary-1b-v2-GGUF`, `cstr/canary-ctc-aligner-GGUF` | OK as-is — model-specific |
| HF ONNX repos | `cstr/cohere-transcribe-onnx-int4`, `cstr/cohere-transcribe-onnx-int8` | OK as-is |

## What needs renaming

### 1. The GitHub repo (high priority)

**Problem.** `cohere-whisper.cpp` no longer reflects what's inside. Most new
visitors arrive looking for `parakeet` or `canary` and the `cohere` in the
name is confusing.

**Options:**

| Name | Pros | Cons |
| --- | --- | --- |
| Keep `cohere-whisper.cpp` | Zero churn, all old links still resolve | Misleading name |
| **`speech.cpp`** ⭐ | Clean, follows `whisper.cpp`/`llama.cpp` convention, generic enough to grow | Common name; might collide |
| `asr.cpp` | Even more on-the-nose | Same collision risk |
| `nemo.cpp` | 4/5 runtimes ARE NeMo | Cohere isn't NeMo, locks us in |
| `multi-asr.cpp` | Self-explanatory | Ugly |
| `speechforge.cpp` / `asrforge.cpp` | Branded | Unnecessary branding |
| `granary.cpp` | Reflects the dataset 4/5 runtimes share | Most users don't know Granary |

**Recommendation: rename to `speech.cpp`.** GitHub keeps a permanent redirect
from `cohere-whisper.cpp` → `speech.cpp` so all existing links continue to
work indefinitely. CI / clones / forks all keep working with the old URL.

A second-best alternative is `asr.cpp` if `speech.cpp` collides with anything.

### 2. The CLI binaries (medium priority)

**Problem.** Most are fine. The one outlier is **`canary-align`**, which is
named after the model that *bundled* the aux CTC checkpoint, but the tool is
**general-purpose** and works with transcripts from any of the 5 runtimes
(or hand-typed text) in any of the 25 supported languages.

**Options for the alignment binary:**

| Name | Pros | Cons |
| --- | --- | --- |
| Keep `canary-align` | Honest about model origin | Suggests it only works with canary output (it doesn't) |
| **`nfa-align`** ⭐ | Mirrors NeMo Forced Aligner (the official NVIDIA tool that uses this same model the same way), short, memorable, distinctive from `cohere-align` | Less self-explanatory if you don't know NFA |
| `asr-align` | Generic, says what it does | Doesn't disambiguate from `cohere-align` |
| `multi-align` | Truthful | Ugly |
| `subword-align` | Technically accurate (vs cohere-align which is char-level) | Awkward |
| `ctc-align` | Says how it works | Cohere-align is also CTC; conflict |

**Recommendation: rename `canary-align` → `nfa-align`** because NeMo Forced
Aligner is what NVIDIA themselves call this tool when they use this same
model the same way internally. It's short, memorable, and unambiguously
distinct from `cohere-align`. Users who don't know NFA will figure it out
from the help text.

The other CLIs (`cohere-main`, `cohere-align`, `parakeet-main`, `canary-main`)
are fine as-is — each has a clear identity and the `<model>-<action>` pattern
is consistent.

### 3. The HF model repos (low priority)

These are all named after their source model and that's the right thing.
The only one worth considering is `cstr/canary-ctc-aligner-GGUF` —
following the CLI rename, it would become `cstr/nfa-aligner-GGUF` or
`cstr/canary-nfa-aligner-GGUF` (the second is more searchable).

**Recommendation: leave the HF repos alone for now.** Renaming them costs
broken links and download counts; the benefit is small.

## When to do the rename

The rename costs ~30 minutes of mechanical work (rename repo via GitHub UI,
update remote URLs, update README badges, update docs that reference the
old name) and breaks any external link that doesn't follow GitHub's redirect.

**Recommended timing:**

- The CTC aligner work in this session is the natural inflection point — the
  fork now has 5 runtimes and a sixth tool is unlikely without a similar
  scope expansion.
- The next time we add another runtime (e.g. a new NeMo release, or a
  Whisper-large variant), do the rename simultaneously.
- Or just do it now while the parakeet branch is still pre-merge to main.

## What does NOT need renaming

- The `parakeet` branch — it's the working branch, and it'll merge to `main`
  eventually with the new repo name.
- The internal lib names (`libcohere.a`, `libparakeet.a`, `libcanary.a`,
  `libcanary_ctc.a`) — they're model-specific and that's correct.
- The model `.gguf` filenames — they're tied to the source model and the HF
  download URL.
- The `parakeet-todo.md`, `canary-todo.md`, `canary-ctc.md` planning docs.

## Decision checklist

When you're ready to rename:

- [ ] Pick one of `speech.cpp` / `asr.cpp` / other
- [ ] Pick one of `nfa-align` / `asr-align` / other for the new alignment binary name
- [ ] Rename the GitHub repo via UI (preserves history + redirects)
- [ ] `git remote set-url origin https://github.com/CrispStrobe/<new-name>`
- [ ] Update README.md hero, badges, all instances of the old name
- [ ] Update HF model READMEs (the ones we control) to point at the new repo
- [ ] Rename `examples/canary-align/` → `examples/nfa-align/` (or chosen name)
- [ ] Update `examples/CMakeLists.txt`
- [ ] Update all tests, benchmarks, docs to use the new binary name
- [ ] Single commit titled `repo: rename to <new-name>` so the diff is grep-able later
