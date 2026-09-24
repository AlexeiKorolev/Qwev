#!/usr/bin/env bash
set -euo pipefail

# Run on a login node with network access before submitting compute jobs.
if [[ -z "${QWEV_HF_HOME:-}" ]]; then
  echo "Set QWEV_HF_HOME to a scratch directory before downloading models." >&2
  exit 2
fi

export HF_HOME="$QWEV_HF_HOME"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$HF_HOME"

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' command is unavailable. Activate the Qwev environment first." >&2
  exit 2
fi

if (( $# == 0 )); then
  set -- \
    Qwen/Qwen3-VL-2B-Instruct \
    Qwen/Qwen3-VL-4B-Instruct \
    Qwen/Qwen3-VL-8B-Instruct
fi

for model in "$@"; do
  echo "Downloading $model into $HF_HOME"
  hf download "$model" --revision "${QWEV_REVISION:-main}"
done
