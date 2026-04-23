#!/usr/bin/env bash
set -euo pipefail

ts() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(ts)] crispasr-docker: $*" >&2
}

fail_dir_not_writable() {
    local dir="$1"
    log "ERROR: '$dir' is not writable by uid=$(id -u) gid=$(id -g)."
    log "       Current permissions:"
    ls -ld "$dir" >&2 || true
    log "       If this is a bind mount, create it before starting the container and chown it:"
    log "       mkdir -p cache models && sudo chown -R $(id -u):$(id -g) cache models"
    log "       Or set CRISPASR_UID/CRISPASR_GID in .env to match the host owner."
    exit 70
}

ensure_writable_dir() {
    local dir="$1"
    local label="$2"

    if [[ ! -d "$dir" ]]; then
        if ! mkdir -p "$dir" 2>/dev/null; then
            log "ERROR: could not create $label directory '$dir'."
            fail_dir_not_writable "$(dirname "$dir")"
        fi
    fi

    if [[ ! -w "$dir" ]]; then
        fail_dir_not_writable "$dir"
    fi
}

SERVER_HOST="${CRISPASR_SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${CRISPASR_PORT:-${CRISPASR_SERVER_PORT:-8080}}"
MODEL_PATH="${CRISPASR_MODEL:-/models/model.gguf}"
LANGUAGE="${CRISPASR_LANGUAGE:-auto}"
BACKEND="${CRISPASR_BACKEND:-}"
AUTO_DOWNLOAD="${CRISPASR_AUTO_DOWNLOAD:-0}"
CACHE_DIR="${CRISPASR_CACHE_DIR:-/cache}"
EXTRA_ARGS="${CRISPASR_EXTRA_ARGS:-}"
API_KEYS="${CRISPASR_API_KEYS:-}"

ensure_writable_dir "$CACHE_DIR" "cache"

if [[ "$AUTO_DOWNLOAD" != "1" ]]; then
    if [[ ! -r "$MODEL_PATH" ]]; then
        log "ERROR: model '$MODEL_PATH' is not readable."
        log "       Mount a model under /models, set CRISPASR_MODEL to a readable file, or set CRISPASR_AUTO_DOWNLOAD=1."
        if [[ -e "$(dirname "$MODEL_PATH")" ]]; then
            ls -ld "$(dirname "$MODEL_PATH")" >&2 || true
        fi
        exit 66
    fi
fi

declare -a args
args=(crispasr --server --host "$SERVER_HOST" --port "$SERVER_PORT" --cache-dir "$CACHE_DIR")

if [[ "$AUTO_DOWNLOAD" == "1" ]]; then
    args+=(-m auto --auto-download)
else
    args+=(-m "$MODEL_PATH")
fi

if [[ -n "$BACKEND" ]]; then
    args+=(--backend "$BACKEND")
fi

if [[ -n "$LANGUAGE" ]]; then
    args+=(-l "$LANGUAGE")
fi

if [[ -n "$API_KEYS" ]]; then
    args+=(--api-keys "$API_KEYS")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    eval "args+=($EXTRA_ARGS)"
fi

display_args=("${args[@]}")
if [[ -n "$API_KEYS" ]]; then
    for i in "${!display_args[@]}"; do
        if [[ "${display_args[$i]}" == "--api-keys" && $((i + 1)) -lt ${#display_args[@]} ]]; then
            display_args[$((i + 1))]="(redacted)"
        fi
    done
fi

log "server_host=$SERVER_HOST server_port=$SERVER_PORT backend=${BACKEND:-default} language=${LANGUAGE:-default} auto_download=$AUTO_DOWNLOAD cache_dir=$CACHE_DIR"
if [[ -n "$API_KEYS" ]]; then
    log "api_keys=enabled"
fi
log "launching: ${display_args[*]}"
exec "${args[@]}"
