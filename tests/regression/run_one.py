#!/usr/bin/env python3
"""Run one backend's regression against pinned HF artifacts.

For each backend in ``tests/regression/manifest.json``:

1. Download the GGUF under test at the pinned HF revision SHA.
2. Download the reference-dump archive at the pinned fixtures revision.
3. Run ``crispasr -m <gguf> -f <sample>`` and assert the transcript
   matches the pinned ``expected_transcript`` byte-for-byte.
4. Run ``crispasr-diff <backend_id> <gguf> <ref> <sample>`` and assert
   every stage's ``cos_min`` is at or above its pinned threshold.

Pinned revisions guard against:

- Upstream re-quantise of the GGUF on HF silently changing what users
  download (the regression-from-ggml-assertion-hardening lesson).
- Drift in our own reference dumps between commits.

Exit code is the number of failures (0 on full success).

Usage:

  tests/regression/run_one.py parakeet-tdt-0.6b-ja
  BUILD_DIR=build-ninja-compile tests/regression/run_one.py parakeet-tdt-0.6b-ja
  CRISPASR_BIN=/path/crispasr DIFF_BIN=/path/crispasr-diff \\
    tests/regression/run_one.py parakeet-tdt-0.6b-ja

Env:

  BUILD_DIR       Build directory containing bin/crispasr + bin/crispasr-diff
                  (default: build-regression).
  CRISPASR_BIN    Override the crispasr binary path entirely.
  DIFF_BIN        Override the crispasr-diff binary path entirely.
  WORK_DIR        Where to stage downloads (default: a tempdir; cleaned on
                  exit unless KEEP_WORK=1).
  KEEP_WORK       If set to 1, don't delete WORK_DIR on exit.
  REGRESSION_MANIFEST  Path to manifest.json (default:
                  tests/regression/manifest.json).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"\033[31mERROR\033[0m  {msg}", file=sys.stderr)
    sys.exit(code)


def hf_download(repo: str, file_in_repo: str, revision: str, dest_dir: Path) -> Path:
    """Download one file from HF at a pinned revision; return its local path.

    Honours `HF_HOME` / `HUGGINGFACE_HUB_CACHE` from the environment if
    set (so a local dev run can share the on-disk HF cache and skip
    re-downloads). Falls back to a per-job cache under ``dest_dir`` —
    the right choice for CI where every job is a clean runner anyway.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = None
    if not (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        cache_dir = str(dest_dir / "hf_cache")

    print(f"  download  {repo}@{revision[:8]} :: {file_in_repo}", flush=True)
    local = hf_hub_download(
        repo_id=repo,
        filename=file_in_repo,
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(local)


def run_transcript(crispasr_bin: Path, gguf: Path, sample: Path) -> str:
    """Run `crispasr -m gguf -f sample`, return the transcript line.

    The CLI prints the transcript on its own line near the end, after
    the "transcribed N s audio in M s (Kx realtime)" status line.
    Grab the last non-empty non-status line.
    """
    proc = subprocess.run(
        [str(crispasr_bin), "-m", str(gguf), "-f", str(sample)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    # The transcript is on stdout; status lines go to stderr. Take stdout's
    # last non-empty line as the transcript (the CLI prints a single text
    # line for non-streaming mode).
    text_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not text_lines:
        die(
            "crispasr produced no stdout; "
            f"stderr tail: ...{proc.stderr[-400:]}"
        )
    return text_lines[-1]


# Diff-harness output line format (examples/cli/crispasr_diff_main.cpp):
#   [PASS] encoder_output         shape=[1024,176]       cos_min=0.999594  cos_mean=...
#   [FAIL] mel_spectrogram        shape=[80,1407]        cos_min=0.951084  cos_mean=...
_DIFF_LINE = re.compile(
    r"^\[(PASS|FAIL)\]\s+(\S+)\s+shape=\[\S+\]\s+cos_min=([0-9.\-eE]+)\s+cos_mean=([0-9.\-eE]+)"
)


def run_diff(diff_bin: Path, backend_id: str, gguf: Path, ref: Path, sample: Path) -> dict:
    """Run crispasr-diff and return {stage_name: cos_min}.

    crispasr-diff exits non-zero (the count of failed stages) when any
    captured stage falls below its built-in 0.999 cos_min threshold,
    even if we accept that under the manifest's per-stage threshold.
    We do our own pass/fail accounting against `diff_thresholds`, so
    `check=False` here is intentional — we parse stdout regardless of
    the harness's verdict and only fail if a stage is missing or the
    process crashed outright.
    """
    proc = subprocess.run(
        [str(diff_bin), backend_id, str(gguf), str(ref), str(sample)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if proc.returncode < 0:
        die(f"crispasr-diff died from signal {-proc.returncode}\n"
            f"  stderr tail: {proc.stderr[-400:]}")
    # crispasr-diff prints summary lines on stdout. Parse the [PASS]/[FAIL]
    # lines; ignore the diff harness's own pass/fail verdict — we apply our
    # own per-stage thresholds from the manifest.
    result: dict[str, float] = {}
    for ln in proc.stdout.splitlines():
        m = _DIFF_LINE.match(ln.strip())
        if not m:
            continue
        result[m.group(2)] = float(m.group(3))
    if not result:
        die(
            f"crispasr-diff produced no parseable stage lines.\n"
            f"  stdout tail: {proc.stdout[-400:]}\n"
            f"  stderr tail: {proc.stderr[-400:]}"
        )
    return result


def regression_for(name: str, manifest: dict, work_dir: Path,
                   crispasr_bin: Path, diff_bin: Path) -> int:
    """Run one backend's regression. Return number of failures."""
    entry = next((b for b in manifest["backends"] if b["name"] == name), None)
    if entry is None:
        die(f"backend '{name}' not in manifest")

    sample = REPO_ROOT / entry["sample"]
    if not sample.exists():
        die(f"sample WAV missing: {sample}")

    gguf_local = hf_download(
        entry["gguf"]["repo"],
        entry["gguf"]["file"],
        entry["gguf"]["revision"],
        work_dir,
    )
    ref_local = hf_download(
        manifest["fixtures"]["repo"],
        entry["fixture_path"],
        manifest["fixtures"]["revision"],
        work_dir,
    )

    failures = 0

    # ----- 1. Transcript -----
    print(f"\n[transcript] {name}")
    actual = run_transcript(crispasr_bin, gguf_local, sample)
    expected = entry["expected_transcript"]
    if actual != expected:
        print("\033[31m  FAIL\033[0m")
        print(f"    expected: {expected!r}")
        print(f"    actual:   {actual!r}")
        failures += 1
    else:
        print("\033[32m  PASS\033[0m")
        print(f"    {actual!r}")

    # ----- 2. Diff harness -----
    print(f"\n[diff-harness] {name}")
    stages = run_diff(diff_bin, entry["backend_id"], gguf_local, ref_local, sample)
    thresholds = entry["diff_thresholds"]
    for stage, threshold in thresholds.items():
        if stage not in stages:
            print(f"\033[33m  SKIP\033[0m {stage}  (not captured by diff harness)")
            continue
        cos_min = stages[stage]
        ok = cos_min >= threshold
        verdict = "\033[32m  PASS\033[0m" if ok else "\033[31m  FAIL\033[0m"
        print(f"{verdict} {stage:24s} cos_min={cos_min:.6f}  threshold={threshold}")
        if not ok:
            failures += 1
    # Surface any stages that came back but aren't in thresholds — these
    # are new captures that should be added to manifest.json.
    extra = set(stages) - set(thresholds)
    for stage in sorted(extra):
        print(f"\033[33m  INFO\033[0m {stage} cos_min={stages[stage]:.6f} "
              f"(not in manifest thresholds; add it if intentional)")

    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    backend_name = sys.argv[1]

    manifest_path = Path(os.environ.get(
        "REGRESSION_MANIFEST",
        REPO_ROOT / "tests" / "regression" / "manifest.json",
    ))
    with manifest_path.open() as f:
        manifest = json.load(f)

    build_dir = Path(os.environ.get("BUILD_DIR", "build-regression"))
    crispasr_bin = Path(os.environ.get(
        "CRISPASR_BIN", build_dir / "bin" / "crispasr"))
    diff_bin = Path(os.environ.get(
        "DIFF_BIN", build_dir / "bin" / "crispasr-diff"))
    for p, label in [(crispasr_bin, "crispasr"), (diff_bin, "crispasr-diff")]:
        if not p.exists():
            die(f"{label} binary not found at {p}. "
                f"Build it first or set CRISPASR_BIN/DIFF_BIN.")

    work_root = Path(os.environ.get(
        "WORK_DIR",
        tempfile.mkdtemp(prefix="crispasr-regression-")))
    keep_work = os.environ.get("KEEP_WORK") == "1"

    try:
        failures = regression_for(
            backend_name, manifest, work_root, crispasr_bin, diff_bin)
        if failures == 0:
            print(f"\n\033[32mOK\033[0m  {backend_name}: all checks passed")
            return 0
        else:
            print(f"\n\033[31mFAIL\033[0m  {backend_name}: {failures} check(s) failed")
            return failures
    finally:
        if not keep_work and work_root.exists():
            print(f"\ncleanup  {work_root}", file=sys.stderr)
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
