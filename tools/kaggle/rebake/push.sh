#!/usr/bin/env bash
# Push the CrispASR re-bake notebook to Kaggle.
#
# Sibling to ../push.sh — same pattern, different kernel id.
# `kaggle kernels push` uploads + immediately runs. First run on a
# kernel without the HF_TOKEN Kaggle secret will fail at the
# preflight step with a clear message; add the secret + retrigger
# from the kernel page UI.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "kaggle kernels push -p $DIR"
kaggle kernels push -p "$DIR"

ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo
echo "Push triggered. Watch live at:"
echo "  https://www.kaggle.com/code/${ID}"
echo
echo "Poll status via CLI:"
echo "  kaggle kernels status $ID"
echo
echo "After the first run lands (will fail without HF_TOKEN),"
echo "open the URL above and:"
echo "  1. Add-ons → Secrets → add label HF_TOKEN, value = write-scoped"
echo "     HF token (https://huggingface.co/settings/tokens)."
echo "  2. Settings → 'Schedule a notebook run' → Monthly · 1st · 04:00 UTC."
echo "     Monthly is plenty for re-bake; the validate kernel runs weekly."
