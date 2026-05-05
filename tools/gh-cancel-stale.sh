#!/usr/bin/env bash
# tools/gh-cancel-stale.sh
#
# Cancel in-progress / queued GitHub Actions runs that are not the
# newest run for their (workflow_name, branch_or_tag) tuple. Keeps the
# latest active run for each tuple; cancels all older active runs.
#
# Usage:
#   ./tools/gh-cancel-stale.sh           # dry-run, prints what would be cancelled
#   ./tools/gh-cancel-stale.sh --yes     # actually cancel
#   REPO=owner/repo ./tools/gh-cancel-stale.sh --yes
#
# Defaults to the current repo (gh resolves it from the working dir).

set -euo pipefail

REPO="${REPO:-}"
DRY_RUN=true
[[ "${1:-}" == "--yes" ]] && DRY_RUN=false
[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

command -v gh >/dev/null || { echo "error: gh not on PATH" >&2; exit 2; }
command -v jq >/dev/null || { echo "error: jq not on PATH" >&2; exit 2; }

repo_args=()
[[ -n "$REPO" ]] && repo_args=(--repo "$REPO")

# Pull up to 200 most recent runs, filter to active ones, group by
# (workflow name, branch), keep the newest per group, output the rest.
mapfile -t stale < <(
    gh run list "${repo_args[@]}" --limit 200 \
        --json databaseId,name,status,headBranch,createdAt \
    | jq -r '
        [ .[] | select(.status == "in_progress" or .status == "queued"
                    or .status == "waiting" or .status == "requested"
                    or .status == "pending") ]
        | group_by({name, headBranch})
        | map(sort_by(.createdAt) | reverse | .[1:])
        | flatten
        | .[]
        | "\(.databaseId)\t\(.name)\t\(.headBranch)\t\(.createdAt)"
    '
)

if (( ${#stale[@]} == 0 )); then
    echo "No stale active runs to cancel."
    exit 0
fi

echo "Stale active runs (${#stale[@]}):"
printf '  %s\n' "${stale[@]}"
echo

if $DRY_RUN; then
    echo "(dry-run; pass --yes to actually cancel)"
    exit 0
fi

for line in "${stale[@]}"; do
    id="${line%%$'\t'*}"
    echo "cancelling $id ..."
    gh run cancel "$id" "${repo_args[@]}" || true
done

echo
echo "Done. Verify with: gh run list ${repo_args[*]} --limit 20"
