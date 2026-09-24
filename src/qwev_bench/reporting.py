"""Conservative within-run summaries: no mixing GPUs, failed runs or warmups."""

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "input_tokens", "image_tokens", "processor_ms", "transfer_ms",
    "request_wall_ms", "request_e2e_ms", "request_cuda_ms", "vision_cuda_ms",
    "language_cuda_ms", "language_prefill_cuda_ms", "language_decode_cuda_ms",
    "readout_cuda_ms", "image_decode_ms", "output_decode_ms", "generated_tokens",
    "peak_allocated_bytes", "peak_reserved_bytes",
)


def percentile(values, quantile):
    """Linear interpolation; avoids claiming meaningful tail precision at small n."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_files(patterns, output):
    paths = sorted({Path(match).resolve() for pattern in patterns for match in glob.glob(str(pattern))})
    if not paths:
        raise ValueError("No result files matched")
    groups = defaultdict(list)
    metadata = {}
    completed = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}; possibly an interrupted write") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                if record.get("type") == "run":
                    metadata[str(path)] = record
                    continue
                if record.get("type") == "complete":
                    completed.add(str(path))
                    continue
                if record.get("type") == "error":
                    record = dict(record, case_id="__run__", mode="__load_or_setup__")
                elif record.get("type") != "sample":
                    continue
                if record.get("warmup", False) and record.get("status") == "ok":
                    continue
                # source path additionally protects old files missing a run_id.
                key = (str(path), record.get("run_id", ""), record.get("model", ""),
                       record.get("case_id", ""), record.get("mode", ""))
                groups[key].append(record)
    rows = []
    for key, samples in sorted(groups.items()):
        row = dict(zip(["source_file", "run_id", "model", "case_id", "mode"], key))
        meta = metadata.get(key[0], {})
        for field in ("gpu_name", "gpu_total_memory_bytes", "dtype", "attention_implementation",
                      "transformers_version", "torch_version", "model_commit_hash"):
            row[field] = meta.get(field, "")
        row["run_completed"] = key[0] in completed
        successful = [sample for sample in samples if sample.get("status") == "ok"]
        row.update(samples=len(samples), successful=len(successful), failed=len(samples)-len(successful),
                   error_types=";".join(sorted({str(s.get("error_type", s.get("status", "unknown"))) for s in samples if s.get("status") != "ok"})))
        row["warmup_failures"] = sum(bool(s.get("warmup")) for s in samples if s.get("status") != "ok")
        for field in ("json_valid", "action_valid", "hit_token_limit"):
            checked = [s[field] for s in successful if isinstance(s.get(field), bool)]
            row[f"{field}_checked"] = len(checked)
            row[f"{field}_true"] = sum(checked)
        for metric in METRICS:
            values = [float(s[metric]) for s in successful
                      if isinstance(s.get(metric), (int, float)) and not isinstance(s.get(metric), bool)
                      and math.isfinite(s[metric])]
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_p50"] = percentile(values, 0.50)
            row[f"{metric}_p95"] = percentile(values, 0.95)
        rows.append(row)
    if not rows:
        raise ValueError("No measured sample records found (only warmups/metadata, or run failed before measurements)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
