#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_BIN="${LABELONE_NODE:-$(command -v node || true)}"
CODEX_DEPENDENCIES="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies"
CODEX_NODE="$CODEX_DEPENDENCIES/node/bin/node"
CODEX_PNPM="$CODEX_DEPENDENCIES/bin/fallback/pnpm"
USER_UV="$HOME/.local/bin/uv"

node_major() {
  "$1" -p "Number(process.versions.node.split('.')[0])" 2>/dev/null || printf '0'
}

if [[ -z "$NODE_BIN" || "$(node_major "$NODE_BIN")" -lt 22 ]]; then
  if [[ -x "$CODEX_NODE" && "$(node_major "$CODEX_NODE")" -ge 22 ]]; then
    NODE_BIN="$CODEX_NODE"
  else
    printf 'LabelOne requires Node.js 22.13 or newer. Set LABELONE_NODE to a compatible executable.\n' >&2
    exit 1
  fi
fi

export PATH="$(dirname "$NODE_BIN"):$PATH"

if [[ ! -d "$PROJECT_DIR/node_modules" ]]; then
  PNPM_BIN="${LABELONE_PNPM:-$(command -v pnpm || true)}"
  if [[ -z "$PNPM_BIN" && -x "$CODEX_PNPM" ]]; then
    PNPM_BIN="$CODEX_PNPM"
  fi
  if [[ -z "$PNPM_BIN" ]]; then
    printf 'Frontend dependencies are missing and no bundled pnpm was found. Install pnpm once, then rerun this launcher.\n' >&2
    exit 1
  fi
  export COREPACK_HOME="${COREPACK_HOME:-$PROJECT_DIR/.runtime/corepack}"
  mkdir -p "$COREPACK_HOME"
  (cd "$PROJECT_DIR" && "$PNPM_BIN" install)
fi

SERVER_PYTHON="$PROJECT_DIR/server/.venv/bin/python"
if [[ ! -x "$SERVER_PYTHON" ]]; then
  UV_BIN="${LABELONE_UV:-$(command -v uv || true)}"
  if [[ -z "$UV_BIN" && -x "$USER_UV" ]]; then
    UV_BIN="$USER_UV"
  fi
  if [[ -z "$UV_BIN" ]]; then
    printf 'Server environment is missing and no bundled uv was found. Install uv once, then rerun this launcher.\n' >&2
    exit 1
  fi
  (cd "$PROJECT_DIR/server" && "$UV_BIN" sync --extra dev)
fi

if [[ ! -x "$SERVER_PYTHON" ]]; then
  printf 'LabelOne server environment could not be created.\n' >&2
  exit 1
fi

# Models also work without configuration. Reuse a local checkout when one is
# present; otherwise prepare a private, gitignored metadata source once.
if [[ -z "${LABELONE_X_ANYLABELING_ROOT:-}" ]]; then
  for CANDIDATE in \
    "$PROJECT_DIR/.runtime/x-anylabeling" \
    "$PROJECT_DIR/../X-AnyLabeling" \
    "$PROJECT_DIR/../x-anylabeling"; do
    if [[ -f "$CANDIDATE/anylabeling/configs/models.yaml" ]]; then
      export LABELONE_X_ANYLABELING_ROOT="$CANDIDATE"
      break
    fi
  done
fi

if [[ -z "${LABELONE_X_ANYLABELING_ROOT:-}" ]]; then
  MODEL_SOURCE="$PROJECT_DIR/.runtime/x-anylabeling"
  GIT_BIN="$(command -v git || true)"
  if [[ -n "$GIT_BIN" && ! -e "$MODEL_SOURCE" ]]; then
    mkdir -p "$PROJECT_DIR/.runtime"
    printf 'First run: preparing X-AnyLabeling model metadata…\n'
    if "$GIT_BIN" clone --depth 1 https://github.com/CVHub520/X-AnyLabeling.git "$MODEL_SOURCE"; then
      export LABELONE_X_ANYLABELING_ROOT="$MODEL_SOURCE"
    else
      printf 'Warning: model metadata could not be downloaded. Dataset, annotation, pipeline and Agent features remain available.\n' >&2
    fi
  fi
fi

cleanup() {
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ACCESS_LOG_ARGS=()
if [[ "${LABELONE_ACCESS_LOG:-0}" != "1" ]]; then
  ACCESS_LOG_ARGS=(--no-access-log)
fi

(cd "$PROJECT_DIR/server" && "$SERVER_PYTHON" -m uvicorn --app-dir src labelone.main:app --host 127.0.0.1 --port 8766 --reload --reload-dir src "${ACCESS_LOG_ARGS[@]}") &
SERVER_PID=$!

for _ in {1..50}; do
  if curl -fsS http://127.0.0.1:8766/api/v1/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! curl -fsS http://127.0.0.1:8766/api/v1/health >/dev/null 2>&1; then
  printf 'LabelOne local API did not become ready on port 8766.\n' >&2
  exit 1
fi

(cd "$PROJECT_DIR" && "$NODE_BIN" "$PROJECT_DIR/node_modules/vinext/dist/cli.js" dev) &
WEB_PID=$!

printf 'LabelOne is running:\n  Web: http://localhost:3000\n  API: http://127.0.0.1:8766/api/v1\n'

# macOS still ships Bash 3.2, so avoid `wait -n`. Detect either child exiting
# and let the EXIT trap stop the survivor instead of silently leaving users
# with an API-only or Web-only launcher.
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    set +e
    wait "$SERVER_PID"
    STATUS=$?
    set -e
    exit "$STATUS"
  fi
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    set +e
    wait "$WEB_PID"
    STATUS=$?
    set -e
    exit "$STATUS"
  fi
  sleep 0.5
done
