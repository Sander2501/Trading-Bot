#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[test] .venv not found; running setup first"
  "$ROOT_DIR/scripts/dev_setup.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$ROOT_DIR"
pytest -q
