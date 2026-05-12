# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — full-suite regression / re-bake (Kaggle)
#
# Two modes, picked by a flag at the top of cell 1:
#
# **`MODE = "validate"`** (default) — same contract as the nightly
# GitHub Actions workflow, but for every backend in one run on faster
# infra. For each `tests/regression/manifest.json` entry:
#   1. Download the GGUF under test at the pinned HF revision.
#   2. Download the reference dump from
#      `cstr/crispasr-regression-fixtures` at the pinned revision.
#   3. Run `crispasr` and assert the transcript matches
#      `expected_transcript` byte-for-byte.
#   4. Run `crispasr-diff` and assert per-stage `cos_min ≥ threshold`.
# Use this to confirm a release branch before tagging, or to validate
# any backend that's too heavy for the nightly GH Actions runner
# (vibevoice 7 GB, voxtral4b 4 GB, …).
#
# **`MODE = "rebake"`** — regenerates the reference dumps from the
# real Python source models (NeMo / transformers / torch) for every
# backend. Stages them locally, optionally uploads to HF. The new
# `cstr/crispasr-regression-fixtures` commit SHA is printed at the
# end; the maintainer pastes it into `manifest.json`'s
# `fixtures.revision`. Run this whenever:
#   - A reference module changes (`tools/reference_backends/...`).
#   - A new backend is added.
#   - Upstream PyTorch / NeMo / transformers bumps with a known
#     numerical effect we want to accept as the new baseline.
# Re-bake is intentional, never silent — the upload step is gated
# on `UPLOAD=True` and prints the manifest patch for review.
#
# Why Kaggle, not GitHub Actions:
#   - Real ML stack (NeMo, transformers, torch) totals 5-10 GB; ten
#     minutes to install per nightly run is unaffordable on GH free.
#   - ~1 Gbit pipe + GPU vs ~250 Mbit on GH Actions makes the long
#     downloads tolerable.
#   - 20 GB scratch fits the heavy backends that don't fit on GH.
#
# Requirements:
# - Kaggle accelerator: any (CPU works; GPU only matters if you
#   want to validate the CUDA path of CrispASR).
# - Internet ON (model downloads + optional HF upload).
# - Optional Kaggle secrets:
#     `HF_TOKEN` — required for `MODE="rebake"` + `UPLOAD=True`.
#                  Read-only token is fine for `MODE="validate"`
#                  (only public HF repos are read).
#     `GH_TOKEN` — to post a summary as a comment on the latest
#                  main commit. Optional.

# ─────────────────────────── cell 1 (code) ───────────────────────────
# ── Configuration ──────────────────────────────────────────────────────────
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODE = os.environ.get("CRISPASR_REGRESSION_MODE", "validate")  # or "rebake"

# Only consulted in rebake mode.
UPLOAD = os.environ.get("CRISPASR_REGRESSION_UPLOAD", "0") == "1"

# Restrict to a subset of backends (comma-separated names). Empty == all.
BACKEND_FILTER = os.environ.get("CRISPASR_REGRESSION_BACKENDS", "").strip()

# Build flags. Default to CPU; flip to "cuda" to test the GPU path.
BUILD_FLAVOUR = os.environ.get("CRISPASR_REGRESSION_BUILD", "cpu")  # cpu | cuda

# CrispASR commit to test. Default to main; pin a SHA to bisect.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# ── Workspace layout ──────────────────────────────────────────────────────
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
HF_CACHE = WORK / "hf_cache"
RESULTS = WORK / "results"
REBAKE_STAGE = WORK / "rebake-stage"

for d in (HF_CACHE, RESULTS, REBAKE_STAGE):
    d.mkdir(parents=True, exist_ok=True)

# Funnel all HF downloads into the same cache so validate-after-rebake
# in the same notebook session is a free re-read.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)

print(f"crispasr-regression {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  MODE             = {MODE}")
print(f"  BUILD_FLAVOUR    = {BUILD_FLAVOUR}")
print(f"  CRISPASR_REF     = {CRISPASR_REF}")
print(f"  BACKEND_FILTER   = {BACKEND_FILTER or '(all)'}")
print(f"  UPLOAD           = {UPLOAD}")

# ─────────────────────────── cell 2 (code) ───────────────────────────
# ── Wire HF auth, install Python deps ─────────────────────────────────────
def kaggle_secret(name: str) -> str | None:
    """Pull a Kaggle secret if available; return None silently otherwise."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


hf_token = kaggle_secret("HF_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("HF auth: token present (will verify next)")
else:
    print("HF auth: anonymous (rebake+upload will fail without HF_TOKEN)")

# Need huggingface_hub before we can preflight the token. Pulled here
# (small, ~MB) even if the token check ends up failing — the cost of
# the import is negligible against the fail-fast benefit.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub",
])

# ── Preflight: prove the token / fixture chain works before we spend
#    10 minutes building + downloading models we'll never use. Better
#    to die loudly in cell 2 than 25 minutes later in cell 7. ────────
def preflight_hf() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=hf_token) if hf_token else HfApi()

    # 1. If a token is present, prove it's valid and tell us whose it is.
    if hf_token:
        try:
            info = api.whoami()
            user = info.get("name") or info.get("fullname") or "?"
            orgs = [o.get("name") for o in info.get("orgs", []) if o.get("name")]
            print(f"HF auth: token OK — user={user!r}  orgs={orgs}")
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"HF auth: token REJECTED by /api/whoami-v2 ({exc}).\n"
                f"  Generate a fresh token at https://huggingface.co/settings/tokens\n"
                f"  and store it as the Kaggle secret HF_TOKEN (read+write for rebake)."
            )

    # 2. For rebake+upload, the token must additionally have *write*
    #    access to the fixtures repo. Probe via repo_info(); HF returns
    #    a `private`/`gated` flag we can sanity-check, and the call
    #    itself 401s if the token can't see private repos when needed.
    #    Writability is harder to introspect cleanly — the cheapest
    #    proof is the upload step itself — but we can at least confirm
    #    the repo exists and the user can see it.
    fixtures_repo = "cstr/crispasr-regression-fixtures"
    try:
        info = api.repo_info(repo_id=fixtures_repo, repo_type="model")
        print(f"HF fixtures: {fixtures_repo} reachable (last_modified={info.last_modified})")
    except HfHubHTTPError as exc:
        msg = (
            f"HF fixtures: cannot reach {fixtures_repo} ({exc}).\n"
            f"  validate mode CAN'T proceed without the fixtures repo;\n"
            f"  rebake+upload CAN'T proceed without write access to it."
        )
        raise SystemExit(msg)

    if MODE == "rebake" and UPLOAD:
        if not hf_token:
            raise SystemExit(
                "rebake+UPLOAD=1 requires HF_TOKEN with write access to "
                f"{fixtures_repo}. Add it as a Kaggle secret."
            )
        # Best-effort write probe: open a no-op preupload (computes
        # remote-cas hash for a 1-byte blob; HF returns 401 if we
        # can't write, OK otherwise). Cheaper than actually committing.
        try:
            api.preupload_lfs_files(
                repo_id=fixtures_repo,
                repo_type="model",
                additions=[],  # zero files — just exercises the auth check
            )
            print(f"HF fixtures: write access to {fixtures_repo} confirmed")
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"HF fixtures: token can READ {fixtures_repo} but write "
                f"probe failed ({exc}). Generate a write-scoped token at "
                f"https://huggingface.co/settings/tokens."
            )
        except Exception as exc:
            # preupload_lfs_files API may shift; fall back to a warning
            # rather than blocking the rebake. The real upload step
            # will surface any actual auth error.
            print(f"HF fixtures: write probe inconclusive ({type(exc).__name__}: "
                  f"{exc}). Proceeding; real upload will catch it.")


preflight_hf()
if MODE == "rebake":
    # The heavy ML stack only matters when re-baking. validate mode
    # never touches NeMo / transformers / torch.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "nemo_toolkit[asr]", "transformers", "torch", "torchaudio",
        "numpy", "gguf",
    ])

# ─────────────────────────── cell 3 (code) ───────────────────────────
# ── Clone + build CrispASR ────────────────────────────────────────────────
def sh(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)


if not REPO.exists():
    sh(f"git clone --recursive https://github.com/CrispStrobe/CrispASR.git {REPO}")
sh(f"git fetch origin && git checkout {CRISPASR_REF}", cwd=REPO)
sh("git submodule update --init --recursive", cwd=REPO)

build_flags = []
if BUILD_FLAVOUR == "cuda":
    build_flags.append("-DGGML_CUDA=ON")

sh("apt-get update -qq && apt-get install -y --no-install-recommends "
   "cmake ninja-build g++ libopenblas-dev jq")
sh(
    f"cmake -S {REPO} -B {BUILD} -G Ninja "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF "
    + " ".join(build_flags)
)
# CMake target `crispasr-cli` produces bin/crispasr (target/output names
# intentionally diverge per examples/cli/CMakeLists.txt:12). Asking for
# target `crispasr` here builds only the library, leaving bin/crispasr
# absent — exactly what burned the GH regression workflow on its first
# run (commit 08d1872f) and what just burned this Kaggle one.
sh(f"cmake --build {BUILD} --target crispasr-cli crispasr-diff -j$(nproc)")

# ─────────────────────────── cell 4 (code) ───────────────────────────
# ── Load manifest + backend filter ────────────────────────────────────────
MANIFEST_PATH = REPO / "tests" / "regression" / "manifest.json"
with MANIFEST_PATH.open() as f:
    MANIFEST = json.load(f)

want = set(b.strip() for b in BACKEND_FILTER.split(",") if b.strip())
BACKENDS = [
    b for b in MANIFEST["backends"]
    if not want or b["name"] in want
]
if want:
    missing = want - {b["name"] for b in BACKENDS}
    if missing:
        print(f"WARN: requested backends not in manifest: {sorted(missing)}")

print(f"\nProcessing {len(BACKENDS)} backend(s):")
for b in BACKENDS:
    size_mb = b["gguf"].get("approx_size_mb", "?")
    print(f"  - {b['name']:30s} (gguf ~{size_mb} MB)")


# ─────────────────────────── cell 5 (code) ───────────────────────────
# ── VALIDATE mode: download pinned artifacts, run regression ──────────────
def run_validate() -> list[dict]:
    """Per-backend validate. Returns one result dict per backend."""
    sys.path.insert(0, str(REPO / "tests" / "regression"))
    import run_one  # noqa: E402 — added to path above

    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== validate :: {name} ==========")
        t0 = time.time()
        try:
            from huggingface_hub import hf_hub_download
            gguf_local = Path(hf_hub_download(
                repo_id=entry["gguf"]["repo"],
                filename=entry["gguf"]["file"],
                revision=entry["gguf"]["revision"],
            ))
            ref_local = Path(hf_hub_download(
                repo_id=MANIFEST["fixtures"]["repo"],
                filename=entry["fixture_ref_path"],
                revision=MANIFEST["fixtures"]["revision"],
            ))
            if "fixture_sample_path" in entry:
                sample = Path(hf_hub_download(
                    repo_id=MANIFEST["fixtures"]["repo"],
                    filename=entry["fixture_sample_path"],
                    revision=MANIFEST["fixtures"]["revision"],
                ))
            else:
                sample = REPO / entry["sample"]

            crispasr_bin = BUILD / "bin" / "crispasr"
            diff_bin = BUILD / "bin" / "crispasr-diff"

            actual = run_one.run_transcript(crispasr_bin, gguf_local, sample)
            transcript_ok = (actual == entry["expected_transcript"])
            stages = run_one.run_diff(
                diff_bin, entry["backend_id"], gguf_local, ref_local, sample)
            passes, fails, missing, extras = run_one.evaluate_stage_thresholds(
                stages, entry["diff_thresholds"])
            ok = transcript_ok and not fails and not missing
            results.append({
                "backend": name,
                "mode": "validate",
                "ok": ok,
                "elapsed_s": round(time.time() - t0, 2),
                "transcript_match": transcript_ok,
                "transcript_actual": actual if not transcript_ok else None,
                "stages": {s: stages.get(s) for s in entry["diff_thresholds"]},
                "extras": dict(extras),
                "missing": missing,
            })
            print(f"  -> ok={ok}  transcript={transcript_ok}  "
                  f"passes={len(passes)}  fails={len(fails)}  missing={len(missing)}")
        except Exception as exc:
            results.append({
                "backend": name,
                "mode": "validate",
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  -> ERROR  {type(exc).__name__}: {exc}")

    return results


# ─────────────────────────── cell 6 (code) ───────────────────────────
# ── REBAKE mode: run real Python references, stage new ref.gguf files ────
def run_rebake() -> list[dict]:
    """Per-backend re-bake. Writes new ref.gguf files into REBAKE_STAGE
    at the manifest's `fixture_path`. Does NOT upload — that's a
    separate gated step.
    """
    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== rebake :: {name} ==========")
        t0 = time.time()
        # `backend_id` is the registered name in tools/dump_reference.py;
        # `fixture_path` is what `manifest.json` says we'll ship.
        out_path = REBAKE_STAGE / entry["fixture_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Source-model spec. Per-backend modules know how to interpret
        # `--model-dir` (HF id or local path). The manifest carries
        # `source_model` exactly for re-bake — fall back is a hard
        # error rather than a guess, because guessing a wrong NeMo
        # checkpoint silently changes what we baseline against.
        source = entry.get("source_model")
        if not source:
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": False,
                "elapsed_s": 0.0,
                "error": "manifest entry has no `source_model`; "
                         "add it before re-baking",
            })
            print(f"  -> SKIP (no source_model)")
            continue
        # For re-bake, the sample WAV must be on disk so the Python
        # reference can read it. Pull from the fixtures HF repo if the
        # manifest points there; otherwise expect it in-tree.
        if "fixture_sample_path" in entry:
            from huggingface_hub import hf_hub_download
            sample = Path(hf_hub_download(
                repo_id=MANIFEST["fixtures"]["repo"],
                filename=entry["fixture_sample_path"],
                revision=MANIFEST["fixtures"]["revision"],
            ))
        else:
            sample = REPO / entry["sample"]

        cmd = [
            sys.executable, "-u", str(REPO / "tools" / "dump_reference.py"),
            "--backend", entry["backend_id"],
            "--model-dir", source,
            "--audio", str(sample),
            "--output", str(out_path),
        ]
        try:
            subprocess.check_call(cmd, cwd=str(REPO))
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": True,
                "elapsed_s": round(time.time() - t0, 2),
                "out_path": str(out_path),
                "out_size_b": out_path.stat().st_size,
            })
            print(f"  -> wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")
        except subprocess.CalledProcessError as exc:
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"dump_reference exit={exc.returncode}",
            })

    return results


# ─────────────────────────── cell 7 (code) ───────────────────────────
# ── Dispatch + upload ─────────────────────────────────────────────────────
if MODE == "validate":
    RESULTS_DATA = run_validate()
elif MODE == "rebake":
    RESULTS_DATA = run_rebake()
else:
    raise SystemExit(f"unknown MODE={MODE!r}; want 'validate' or 'rebake'")

# Write a single JSON Lines artifact for downstream diffing.
results_jsonl = RESULTS / f"results-{MODE}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
with results_jsonl.open("w") as f:
    for r in RESULTS_DATA:
        f.write(json.dumps(r) + "\n")
print(f"\nResults: {results_jsonl}")

# Summary line for stdout (so a Kaggle screenshot is self-contained).
n_ok = sum(1 for r in RESULTS_DATA if r.get("ok"))
n_fail = sum(1 for r in RESULTS_DATA if not r.get("ok"))
print(f"\nSUMMARY  mode={MODE}  ok={n_ok}/{len(RESULTS_DATA)}  fail={n_fail}")
for r in RESULTS_DATA:
    flag = "✓" if r.get("ok") else "✗"
    print(f"  {flag} {r['backend']:30s} {r['elapsed_s']:6.1f}s  {r.get('error', '')}")

if MODE == "rebake" and UPLOAD:
    if any(not r.get("ok") for r in RESULTS_DATA):
        raise SystemExit(
            "rebake had failures; refusing to upload. Inspect failures, "
            "fix, re-run rebake with the same workspace until all backends "
            "pass, then re-run with UPLOAD=1.")
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nUploading {REBAKE_STAGE}/ → cstr/crispasr-regression-fixtures")
    # Use upload_folder so the structure mirrors the staging dir
    # exactly. delete_patterns kept empty: never silently delete a
    # ref.gguf that's still in the manifest.
    commit_info = api.upload_folder(
        repo_id="cstr/crispasr-regression-fixtures",
        repo_type="model",
        folder_path=str(REBAKE_STAGE),
        commit_message=f"rebake {len(RESULTS_DATA)} backend(s) — "
                       f"crispasr ref {CRISPASR_REF}",
    )
    print(f"\nNew fixtures commit: {commit_info.oid}")
    print(f"  → bump manifest.json's fixtures.revision to {commit_info.oid}")
    print(f"  → https://huggingface.co/cstr/crispasr-regression-fixtures/commit/{commit_info.oid}")

# Exit non-zero so a Kaggle scheduled run shows up as failed when
# anything regressed. Without this, Kaggle treats any successful
# notebook execution as "green" regardless of cell-internal state.
sys.exit(n_fail)
