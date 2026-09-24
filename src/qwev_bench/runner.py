"""Cluster benchmark driver. The public entry point is ``run(config, output)``."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .modeling import (
    action_token_ids,
    create_readout,
    decision_forward,
    generate_json,
    generate_one_token,
    input_counts,
    load_model,
    prepare_inputs,
    transfer_inputs,
)
from .timing import StageProfiler, milliseconds_since, synchronized_call


VALID_MODES = frozenset({"prefill", "head", "one_token", "json"})


def _write(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    stream.flush()


def _validate(config: dict[str, Any]) -> None:
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("config.model must be a nonempty model ID or local path")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config.cases must contain at least one case")
    ids = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("id") or not isinstance(case.get("prompt"), str):
            raise ValueError("each case needs an id and string prompt")
        ids.append(case["id"])
        if case.get("image") is not None and not Path(case["image"]).is_file():
            raise FileNotFoundError(f"Image for case {case['id']}: {case['image']}")
        minimum, maximum = case.get("min_pixels"), case.get("max_pixels")
        if minimum is not None and int(minimum) <= 0:
            raise ValueError("min_pixels must be positive")
        if maximum is not None and int(maximum) <= 0:
            raise ValueError("max_pixels must be positive")
        if minimum is not None and maximum is not None and int(minimum) > int(maximum):
            raise ValueError("min_pixels cannot exceed max_pixels")
    if len(set(ids)) != len(ids):
        raise ValueError("case ids must be unique")
    modes = config.get("modes", ["prefill", "head", "one_token", "json"])
    if not modes or not set(modes).issubset(VALID_MODES):
        raise ValueError(f"modes must be a nonempty subset of {sorted(VALID_MODES)}")
    for field in ("warmup", "repeats"):
        value = int(config.get(field, 2 if field == "warmup" else 10))
        if value < (0 if field == "warmup" else 1):
            raise ValueError(f"{field} must be {'nonnegative' if field == 'warmup' else 'positive'}")
    if int(config.get("max_new_tokens", 32)) < 1:
        raise ValueError("max_new_tokens must be positive")


def _hardware(torch: Any, device: str) -> dict[str, Any]:
    device_properties = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_name": device_properties.name,
        "gpu_total_memory_bytes": device_properties.total_memory,
        "gpu_compute_capability": f"{device_properties.major}.{device_properties.minor}",
        "cuda_runtime_version": torch.version.cuda,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
    }


def _fixture_hashes(cases: list[dict[str, Any]]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for case in cases:
        image_path = case.get("image")
        if image_path is None:
            hashes[case["id"]] = None
            continue
        digest = hashlib.sha256()
        with Path(image_path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[case["id"]] = digest.hexdigest()
    return hashes


def _run_one(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    mode: str,
    repeat: int,
    warmup: bool,
    model: Any,
    processor: Any,
    torch: Any,
    head: Any,
    allowed_ids: list[int] | None,
    run_id: str,
) -> dict[str, Any]:
    device = str(config.get("device", "cuda:0"))
    prompt = case.get("json_prompt", case["prompt"]) if mode == "json" else case["prompt"]
    record: dict[str, Any] = {
        "type": "sample",
        "run_id": run_id,
        "model": config["model"],
        "case_id": case["id"],
        "mode": mode,
        "repeat": repeat,
        "warmup": warmup,
        "status": "ok",
        "image": case.get("image"),
        "image_reuse_across_repeats": bool(case.get("image")),
        "feature_cache_used": False,
        "min_pixels": case.get("min_pixels"),
        "max_pixels": case.get("max_pixels"),
        "max_new_tokens": int(config.get("max_new_tokens", 32)) if mode == "json" else (1 if mode == "one_token" else 0),
        "prompt_kind": "json_prompt" if mode == "json" and "json_prompt" in case else "prompt",
        "instrumentation": "hook_free_primary_plus_cuda_module_hooks",
    }
    cpu_inputs, prep = prepare_inputs(processor, case, prompt)
    record.update(prep)
    record.update(input_counts(cpu_inputs, model))
    device_inputs, transfer_ms = transfer_inputs(cpu_inputs, device, torch)
    record["transfer_ms"] = transfer_ms
    # No cache object is passed across requests. Clearing RoPE state also makes
    # every measurement independent when alternating prefill and generation.
    if mode == "prefill":
        operation = lambda: decision_forward(model, device_inputs, None, torch)
        instrumented_operation = operation
    elif mode == "head":
        operation = lambda: decision_forward(model, device_inputs, head, torch)
        instrumented_operation = lambda: decision_forward(model, device_inputs, head, torch, measure_head=True)
    elif mode == "one_token":
        assert allowed_ids is not None
        operation = lambda: generate_one_token(model, device_inputs, allowed_ids)
        instrumented_operation = operation
    else:
        operation = lambda: generate_json(model, device_inputs, int(config.get("max_new_tokens", 32)))
        instrumented_operation = operation

    head_events = None
    model.model.rope_deltas = None
    with torch.inference_mode():
        primary = synchronized_call(torch, device, operation)
    result = primary.pop("output")
    record.update(primary)
    # A second forward on the same prepared inputs yields stage timings without
    # adding hook overhead to the reported primary request time. It performs
    # vision encoding again and has no shared KV cache.
    model.model.rope_deltas = None
    profiler = StageProfiler(model, torch)
    try:
        with torch.inference_mode():
            instrumented = synchronized_call(torch, device, instrumented_operation)
        stage_result = instrumented.pop("output")
        record["instrumented_request_wall_ms"] = instrumented["request_wall_ms"]
        record["instrumented_request_cuda_ms"] = instrumented["request_cuda_ms"]
        record["vision_cuda_ms"] = profiler.elapsed_ms("vision")
        record["language_cuda_ms"] = profiler.elapsed_ms("language")
        record["language_prefill_cuda_ms"] = profiler.first_elapsed_ms("language")
        record["language_decode_cuda_ms"] = profiler.remaining_elapsed_ms("language")
        record["vision_calls"] = profiler.count("vision")
        record["language_calls"] = profiler.count("language")
        if mode in ("prefill", "head"):
            _, head_events = stage_result
            record["readout_cuda_ms"] = head_events[0].elapsed_time(head_events[1]) if head_events else None
            record["generated_tokens"] = 0
            record["output_decode_ms"] = None
            record["generated_text"] = None
        else:
            start_decode = time.perf_counter()
            new_ids = result[0, device_inputs["input_ids"].shape[-1] :].tolist()
            record["generated_text"] = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            record["output_decode_ms"] = milliseconds_since(start_decode)
            record["generated_tokens"] = len(new_ids)
            record["readout_cuda_ms"] = None
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            record["hit_token_limit"] = len(new_ids) >= record["max_new_tokens"] and (not new_ids or new_ids[-1] not in eos_ids)
            if mode == "one_token":
                record["action_valid"] = len(new_ids) == 1 and new_ids[0] in allowed_ids
                record["json_valid"] = None
            else:
                try:
                    parsed = json.loads(record["generated_text"])
                except (ValueError, TypeError):
                    parsed = None
                record["json_valid"] = isinstance(parsed, dict)
                record["action_valid"] = (
                    isinstance(parsed, dict)
                    and set(parsed) == {"action"}
                    and parsed.get("action") in config.get("action_labels", ["A", "B", "C", "D", "E", "F", "G"])
                )
        record["request_e2e_ms"] = sum(
            value
            for value in (
                record.get("image_decode_ms"),
                record["processor_ms"],
                record["transfer_ms"],
                record["request_wall_ms"],
                record.get("output_decode_ms"),
            )
            if value is not None
        )
        return record
    finally:
        profiler.close()


def run(config: dict[str, Any], output: Path) -> int:
    """Run one loaded model and write append-safe JSONL; return zero on success.

    The caller resolves image paths before invoking this function. The output is
    one ``type=run`` metadata record followed by one ``type=sample`` record per
    case, mode, and repetition. Warmups have ``warmup=true`` and are excluded
    from summaries. Failures are written explicitly and return exit code 2.
    """

    _validate(config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    run_id = str(uuid.uuid4())
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with output.open("x", encoding="utf-8") as stream:
        try:
            model, processor, torch, load_meta = load_model(config)
            device = str(config.get("device", "cuda:0"))
            torch.manual_seed(int(config.get("seed", 42)))
            torch.cuda.manual_seed_all(int(config.get("seed", 42)))
            allowed_ids = (
                action_token_ids(processor, list(config.get("action_labels", ["A", "B", "C", "D", "E", "F", "G"])))
                if "one_token" in config.get("modes", ["prefill", "head", "one_token", "json"])
                else None
            )
            head = create_readout(model, torch, device, len(config.get("action_labels", ["A", "B", "C", "D", "E", "F", "G"])))
            _write(
                stream,
                {
                    "type": "run",
                    "run_id": run_id,
                    "started_utc": started_utc,
                    "model": config["model"],
                    "revision_requested": config.get("revision"),
                    "dtype": config.get("dtype", "bfloat16"),
                    "attention_implementation": config.get("attention_implementation", "sdpa"),
                    "device": device,
                    "seed": int(config.get("seed", 42)),
                    "batch_size": 1,
                    "local_files_only": bool(config.get("local_files_only", True)),
                    "warmup": int(config.get("warmup", 2)),
                    "repeats": int(config.get("repeats", 10)),
                    "modes": config.get("modes", ["prefill", "head", "one_token", "json"]),
                    "action_labels": config.get("action_labels", ["A", "B", "C", "D", "E", "F", "G"]),
                    "resolved_config": config,
                    "fixture_sha256": _fixture_hashes(config["cases"]),
                    "image_semantics": "Each repeat reopens and reprocesses the same case image; no visual features or KV cache are shared across requests.",
                    "head_semantics": "Random untrained linear readout. Latency and memory only; no accuracy claim.",
                    "output_validity_semantics": "json_valid requires a JSON object; JSON action_valid additionally requires exactly one key, action, whose value is an action label. One-token action_valid checks the generated token ID against the allowed set. hit_token_limit means generation ended at the configured maximum without EOS.",
                    "stage_semantics": "Each sample runs the model twice on the same freshly processed input. request_wall_ms/request_cuda_ms are hook-free primary timings; vision/language/readout CUDA events come from the second instrumented request. Event durations include stream idle gaps while Python queues kernels, so they are not pure kernel-compute time or additive. Language prefill is the first language-model call; language decode sums later calls.",
                    **load_meta,
                    **_hardware(torch, device),
                },
            )
            warmups = int(config.get("warmup", 2))
            repeats = int(config.get("repeats", 10))
            for case in config["cases"]:
                for mode in config.get("modes", ["prefill", "head", "one_token", "json"]):
                    for index in range(warmups + repeats):
                        try:
                            record = _run_one(
                                config=config,
                                case=case,
                                mode=mode,
                                repeat=index - warmups if index >= warmups else index,
                                warmup=index < warmups,
                                model=model,
                                processor=processor,
                                torch=torch,
                                head=head,
                                allowed_ids=allowed_ids,
                                run_id=run_id,
                            )
                            _write(stream, record)
                        except Exception as exc:
                            print(f"Benchmark failed in {case['id']}/{mode} (see {output}): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                            _write(stream, {
                                "type": "sample", "run_id": run_id, "model": config["model"],
                                "case_id": case["id"], "mode": mode, "repeat": index - warmups,
                                "warmup": index < warmups, "status": "error",
                                "error_type": type(exc).__name__, "error_message": str(exc),
                                "traceback": traceback.format_exc(limit=8),
                            })
                            return 2
            _write(stream, {"type": "complete", "run_id": run_id, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            return 0
        except Exception as exc:
            print(f"Benchmark setup failed (see {output}): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _write(stream, {
                "type": "error", "run_id": run_id, "status": "error",
                "error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            })
            return 2
