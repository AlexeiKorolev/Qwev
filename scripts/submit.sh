#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QWEV_ENV:-}" ]]; then
  echo "Set QWEV_ENV to the absolute path of the installed Python environment." >&2
  exit 2
fi
if [[ -z "${QWEV_HF_HOME:-}" ]]; then
  echo "Set QWEV_HF_HOME to the scratch directory containing prefetched weights." >&2
  exit 2
fi
if [[ ! -f "$QWEV_ENV/bin/activate" ]]; then
  echo "QWEV_ENV has no bin/activate: $QWEV_ENV" >&2
  exit 2
fi

QWEV_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export QWEV_REPO_DIR QWEV_ENV QWEV_HF_HOME

# Arguments such as --partition and --account are passed directly to sbatch.
sbatch "$@" --export=ALL "$QWEV_REPO_DIR/scripts/benchmark.sbatch"
