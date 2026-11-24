#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x "$PWD/.venv/bin/python" ]]; then
  export PYTHON_BIN="$PWD/.venv/bin/python"
else
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
exec bash run.sh "$@"
