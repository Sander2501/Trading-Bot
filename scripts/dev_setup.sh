#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

printf "[setup] project root: %s\n" "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] error: '$PYTHON_BIN' not found in PATH"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -r "$ROOT_DIR/requirements.txt"

python - <<'PY'
import importlib
required = ["pandas", "pytest", "dotenv"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(f"[setup] missing packages after install: {missing}")
print("[setup] dependency check passed")
PY

echo "[setup] done. Activate with: source .venv/bin/activate"
