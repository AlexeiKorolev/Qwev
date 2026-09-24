# Qwev: Qwen3-VL action latency benchmark

Qwev measures how long a single Qwen3-VL model takes to turn a text or image observation into a small action decision on one CUDA GPU. It records setup costs, actual input and output sizes, repeated request times, and failures in JSONL so runs on Princeton clusters can be compared after checking hardware and software metadata.

The suite compares four paths: `prefill` runs the vision encoder and language model through the current observation; `head` adds a small **untrained** linear readout to the final hidden state; `one_token` uses constrained generation over the A–G action letters; `json` generates a JSON action response up to the configured cap. The seven letters give each task a fixed output size; the prompts define grid actions such as left and pickup or GUI actions such as click Apply. The random readout measures latency and memory only; it does not show that a trained classifier would choose a good action. The synthetic fixtures likewise do not establish task accuracy. Assess action quality separately on labeled, representative observations.

## Prepare on a login node

Use Python 3.10 or newer and a matching PyTorch/torchvision build that sees the cluster's CUDA driver. [Qwen3-VL's model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) uses `Qwen3VLForConditionalGeneration`; this project requires Transformers 4.57.1 or newer in the 4.x series. The project does not quantize or shard weights. Check available GPU memory before selecting 4B or 8B.

Choose a scratch directory that is visible from compute nodes. On Adroit, Princeton documents `/scratch/network/$USER`; on Della, use your allocated `/scratch/gpfs/...` path. The commands below use an Adroit path as an example. Replace it if your scratch location differs. Princeton's [Python](https://researchcomputing.princeton.edu/support/knowledge-base/python), [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch), and [Adroit](https://researchcomputing.princeton.edu/systems/adroit) guides describe current modules, scratch storage, and GPU jobs.

```bash
cd /path/to/Qwev
export QWEV_SCRATCH="/scratch/network/$USER/qwev"
export QWEV_ENV="$QWEV_SCRATCH/venv"
export QWEV_HF_HOME="$QWEV_SCRATCH/huggingface"
mkdir -p "$QWEV_SCRATCH"

# Load a site Python module first if needed; use the same module in batch jobs.
python3 -m venv "$QWEV_ENV"
source "$QWEV_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .

python -m qwev_bench fixtures --output assets
scripts/download_models.sh Qwen/Qwen3-VL-2B-Instruct
python -m qwev_bench run --config configs/smoke.json --output results/smoke.jsonl --dry-run
```

The dry run validates inputs and prints the resolved configuration; it does not load the model or create a result file. If you used a Python module to create the environment, set `QWEV_PYTHON_MODULE` to that module's full name before submitting so the batch script loads it too. Use a module version available on your cluster; the script does not select one for you.

The compute nodes may lack internet access, so `scripts/download_models.sh` runs on the login node and stores weights under `QWEV_HF_HOME`. The batch script sets Hugging Face and Transformers to offline mode, and both shipped configurations use `local_files_only: true`. If model loading reports a missing file, complete the download on the login node and resubmit. The [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) documents `hf download` and cache behavior.

## Run on Slurm

First submit the small smoke configuration for the 2B model:

```bash
export QWEV_CONFIG="$PWD/configs/smoke.json"
scripts/submit.sh --array=0
```

After the smoke result succeeds, prefetch the other two models and submit the matrix:

```bash
scripts/download_models.sh
unset QWEV_CONFIG
scripts/submit.sh
```

`scripts/submit.sh` passes additional Slurm options through, such as `--partition`, `--account`, or a site GPU constraint. It exports the absolute repository path so the job can find code and configuration regardless of where Slurm stages its script. `scripts/benchmark.sbatch` requests one GPU, four CPUs, 64 GB of host memory, and a four hour limit per array task; indexes 0, 1, and 2 run the 2B, 4B, and 8B Instruct models respectively, one at a time. Adjust resources and select a GPU type that fits each full precision model. For a controlled comparison, use the same GPU model and software environment for all three array tasks; inspect Slurm logs and result metadata before combining runs. The defaults do not select a partition, account, or GPU model.

Set `QWEV_OUTPUT_DIR` to a shared scratch directory if desired. Otherwise results go in `results/` under the repository, one new JSONL file per Slurm task. The CLI refuses to overwrite an existing result file. Slurm writes `slurm-<job>_<task>.out` in the submission directory. Use `squeue --me` and `sacct` to inspect job status on the cluster.

```bash
python -m qwev_bench summarize results/*.jsonl --output results/summary.csv
```

For a direct single-GPU run outside the array, use `python -m qwev_bench run --config configs/smoke.json --output results/manual.jsonl` in a GPU allocation. Add `--model Qwen/Qwen3-VL-4B-Instruct` to override the config model, after downloading those weights. This command is intended for an allocated GPU, not a login node.

## What the configurations vary

`configs/smoke.json` contains one grid image and two measured repeats. `configs/matrix.json` uses 30 measured repeats for each case and mode. It contains grid and GUI images, each with a short prompt and a longer instruction packet at three image pixel ceilings (131,072, 393,216, and 786,432), plus text-only controls. The minimum pixel setting is 65,536. The fixture generator creates deterministic images in `assets/`; image paths in a configuration are resolved relative to the configuration file, so `../assets/grid.png` works from any working directory.

The pixel ceilings are processor settings, not guaranteed visual token counts. The runner records actual image tokens, prompt tokens, generated tokens, model revision, software versions, and GPU details where available. Each mode is warmed before its repeated timings; model loading is recorded separately. CUDA synchronization brackets the primary request latency. A second, instrumented pass estimates vision and language stage time; those stage times are diagnostic and are not added to primary latency. End-to-end request latency can include image decoding, preprocessing, transfer, and model computation; inspect the per-stage fields to compare like with like. JSON generation has no constrained schema and can vary in length. JSON validity and action-letter validity are formatting checks, not accuracy measures; compare them with recorded token count and time. The `max_new_tokens` value of 32 is a cap, not a fixed amount of work.

Keep the raw JSONL files when sharing a summary. Averages alone hide failed requests, hardware changes, variable output lengths, and outliers. For publication-grade runs, pin each model to a commit revision, use representative labeled observations, and report quality separately from latency.

## Read the results

Start with the `head` and `prefill` rows: these measure the cost to reach a decision-ready representation without the language vocabulary projection. Compare `one_token` next; it includes the Transformers generation machinery, vocabulary projection, and KV-cache creation, so the difference is not purely the cost of decoding one token. `json` includes subsequent autoregressive steps and uses a slightly different output-format instruction; check `input_tokens` before attributing differences solely to decoding.

| Field | Interpretation |
| --- | --- |
| `load_ms` (run metadata) | Processor and model loading, placement on GPU; excluded from warmed timings. |
| `request_e2e_ms` | Sum of image decoding, processor, transfer, primary model request, and output text decoding. Excludes the diagnostic second pass, file logging, and JSON validation. |
| `request_wall_ms` | Synchronized, uninstrumented model-call latency. |
| `vision_cuda_ms` | Vision module elapsed time in the diagnostic pass. |
| `language_prefill_cuda_ms` | First language module call in that pass. |
| `language_decode_cuda_ms` | Subsequent language module calls; zero when none occur. |
| `readout_cuda_ms` | Random linear head elapsed time, only in the diagnostic pass. |
| `image_tokens` / `input_tokens` | Actual visual tokens / total multimodal input tokens. |
| `peak_allocated_bytes` | Primary request peak including resident weights. Reserved memory is also recorded. |

CUDA-event durations can include gaps while the CPU launches work. They are not a kernel-level profiler, and stage measurements from the second pass should not be subtracted from the first pass to infer an exact residual. Every repeat reopens and re-encodes the same fixture; there is no image-feature or cross-request KV cache. This tests fresh encoding cost, not visual diversity. The second pass doubles model invocations per sample.

The CSV reports median, p95, and valid count for each numeric metric, plus failures, format checks, run-completion status, and hardware metadata. It keeps separate input files/runs separate. Failed runs stop immediately and return a nonzero exit code; inspect their JSONL and Slurm log before rerunning. Thirty repeats are exploratory tail estimates; the two-repeat smoke configuration is only a functional check.

The generated `assets/manifest.json` includes image hashes and manual visual questions for a basic readability check. Replace fixtures with your own screenshots by editing the case image paths and prompts. No result from the random head should be interpreted as action accuracy.

## Local validation

The control-path checks do not require a GPU, model weights, or the ML packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m qwev_bench fixtures --output assets
PYTHONPATH=src python3 -m qwev_bench run --config configs/matrix.json --output results/check.jsonl --dry-run
bash -n scripts/submit.sh scripts/download_models.sh scripts/benchmark.sbatch
```

The tests cover input validation, action token constraints, the decision-ready path avoiding the language output head, timing-stage separation, deterministic fixtures, and result aggregation. They do not substitute for the CUDA smoke job on your cluster.
