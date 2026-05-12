# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — automatic reference re-bake (Kaggle)
#
# Sibling kernel to `chr1str/crispasr-regression-suite`. Same code,
# different defaults: this one runs `MODE=rebake` + `UPLOAD=1`, so a
# scheduled execution mints fresh reference dumps from the real
# NeMo / transformers / torch source models and pushes them to
# `cstr/crispasr-regression-fixtures`.
#
# Requires the Kaggle secret `HF_TOKEN` with **write** scope to the
# fixtures repo. Without it, the preflight fails at cell 2 with a
# clear "rebake+UPLOAD=1 requires HF_TOKEN" message — never wastes
# the 5–10 min ML stack install on an unauthenticated push attempt.
#
# After a successful re-bake, the script prints the new fixtures
# commit SHA. **The manifest pin in
# `tests/regression/manifest.json` is NOT auto-bumped** — that
# stays a reviewable human commit, otherwise drift sneaks into
# nightly without anyone noticing. The maintainer compares the
# old vs new cos numbers, decides whether to accept the drift,
# and bumps the SHA explicitly.
#
# This file is a thin bootstrap shim:
#   - Sets `CRISPASR_REGRESSION_MODE=rebake` + `UPLOAD=1` as
#     defaults (still overridable via Kaggle "Variables").
#   - Clones the latest `main` of CrispASR.
#   - Exec's the canonical `tools/kaggle/crispasr-regression.py`
#     from the clone, so every Kaggle run picks up the freshest
#     bootstrap logic without re-pushing the kernel.

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR-bootstrap"

os.environ.setdefault("CRISPASR_REGRESSION_MODE", "rebake")
os.environ.setdefault("CRISPASR_REGRESSION_UPLOAD", "1")

# Minimal clone — we only need tools/kaggle + tests/regression + the
# source needed to build crispasr-cli / crispasr-diff. Full clone is
# easier than a sparse-checkout pattern and only ~50 MB anyway.
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--recursive", "--depth", "20",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])

# Hand control to the canonical script.
script = REPO / "tools" / "kaggle" / "crispasr-regression.py"
sys.argv[0] = str(script)
exec(compile(script.read_text(), str(script), "exec"))
