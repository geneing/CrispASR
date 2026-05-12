# %% [markdown]
# # CrispASR — automatic reference re-bake (Kaggle)
#
# Sibling kernel to `chr1str/crispasr-regression-suite`. Same code,
# different defaults: this one runs `MODE=rebake` + `UPLOAD=1`, so a
# scheduled execution mints fresh reference dumps from the real
# NeMo / transformers / torch source models and pushes them to
# `cstr/crispasr-regression-fixtures`.
#
# Notebook-type kernel with **real Jupytext cell separators**
# (`# %% [code]`). UserSecretsClient must be called from a dedicated
# top-of-notebook cell; running the secret read in a monolithic
# script kernel (as we did initially) causes a flaky ConnectionError
# even when the JWT is properly injected — that's why this file
# diverges from the `kernel_type: "script"` validate kernel.

# %% [code]
# ── Cell 1: read the HF_TOKEN Kaggle Secret as the FIRST thing the
#    kernel does. Matches Kaggle's documented pattern (secrets in
#    their own cell, at the top, before any heavy imports / pip
#    work).
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token_secret = user_secrets.get_secret("HF_TOKEN")
print("[cell 1] HF_TOKEN secret read OK from Kaggle Secrets")

# %% [code]
# ── Cell 2: export the secret as an env var so the canonical
#    regression script's auth chain (which prefers env vars) picks
#    it up without going through UserSecretsClient a second time.
import os
os.environ["HF_TOKEN"] = hf_token_secret
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret

# Re-bake defaults — overridable via Kaggle's "Add-ons → Variables"
# pane if you want one-off validate-mode runs from this kernel.
os.environ.setdefault("CRISPASR_REGRESSION_MODE", "rebake")
os.environ.setdefault("CRISPASR_REGRESSION_UPLOAD", "1")
print(f"[cell 2] env primed: MODE={os.environ['CRISPASR_REGRESSION_MODE']} "
      f"UPLOAD={os.environ['CRISPASR_REGRESSION_UPLOAD']}")

# %% [code]
# ── Cell 3: shallow-clone CrispASR's main and hand control to the
#    canonical `tools/kaggle/crispasr-regression.py`. Doing the
#    clone in its own cell keeps the secret-handling cells small
#    and self-contained.
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR-bootstrap"
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--recursive", "--depth", "20",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])
print(f"[cell 3] clone OK: {REPO}")

script = REPO / "tools" / "kaggle" / "crispasr-regression.py"
sys.argv[0] = str(script)
exec(compile(script.read_text(), str(script), "exec"))
