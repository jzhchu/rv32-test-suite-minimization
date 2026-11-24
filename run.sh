#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m atm.experiment "$@"
"$PYTHON_BIN" -m atm.reporting "$@"
"$PYTHON_BIN" -m atm.verify_results "$@"
