#!/usr/bin/env bash
#SBATCH --job-name=qwev-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

: "${QWEV_REPO_DIR:?Use scripts/submit.sh to submit this job}"
: "${QWEV_ENV:?Set QWEV_ENV before submission}"
: "${QWEV_HF_HOME:?Set QWEV_HF_HOME before submission}"

if [[ -n "${QWEV_PYTHON_MODULE:-}" ]]; then
  module load "$QWEV_PYTHON_MODULE"
fi
source "$QWEV_ENV/bin/activate"
export HF_HOME="$QWEV_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

case "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array job}" in
  0) model=Qwen/Qwen3-VL-2B-Instruct ;;
  1) model=Qwen/Qwen3-VL-4B-Instruct ;;
  2) model=Qwen/Qwen3-VL-8B-Instruct ;;
  *) echo "Unsupported array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

config="${QWEV_CONFIG:-$QWEV_REPO_DIR/configs/matrix.json}"
output_dir="${QWEV_OUTPUT_DIR:-$QWEV_REPO_DIR/results}"
mkdir -p "$output_dir"
slug="${model##*/}"
output="$output_dir/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${slug}.jsonl"

cd "$QWEV_REPO_DIR"
echo "Host: $(hostname)"
echo "Model: $model"
echo "Config: $config"
echo "Output: $output"
python -m qwev_bench run --config "$config" --model "$model" --output "$output"
