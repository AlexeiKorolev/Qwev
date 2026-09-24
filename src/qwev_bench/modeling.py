"""Qwen3-VL loading, preprocessing, and inference paths used by the benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def load_model(config: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load one dense Qwen3-VL model on one CUDA device, without sharding."""

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = str(config.get("device", "cuda:0"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The real benchmark requires an available CUDA device")
    torch.cuda.set_device(device)
    dtype_name = str(config.get("dtype", "bfloat16"))
    try:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc
    model_name = str(config["model"])
    options = {
        "revision": config.get("revision"),
        "local_files_only": bool(config.get("local_files_only", True)),
    }
    options = {key: value for key, value in options.items() if value is not None}
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_name, **options)
    processor._qwev_default_image_size = dict(processor.image_processor.size)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=str(config.get("attention_implementation", "sdpa")),
        low_cpu_mem_usage=True,
        **options,
    )
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    metadata = {
        "load_ms": (time.perf_counter() - start) * 1000.0,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": model.config.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if model.config.model_type != "qwen3_vl":
        raise ValueError(f"Expected dense qwen3_vl model, got {model.config.model_type}")
    return model, processor, torch, metadata


def set_pixel_budget(processor: Any, case: dict[str, Any]) -> None:
    """Qwen3-VL's official processor uses shortest/longest edge as pixel budgets."""

    size = dict(getattr(processor, "_qwev_default_image_size", processor.image_processor.size))
    minimum = case.get("min_pixels")
    maximum = case.get("max_pixels")
    if minimum is not None:
        size["shortest_edge"] = int(minimum)
    if maximum is not None:
        size["longest_edge"] = int(maximum)
    if size["shortest_edge"] > size["longest_edge"]:
        raise ValueError("min_pixels cannot exceed max_pixels")
    processor.image_processor.size = size


def prepare_inputs(processor: Any, case: dict[str, Any], prompt: str) -> tuple[Any, dict[str, float | None]]:
    """Open an image afresh and perform chat templating on CPU for every request."""

    from PIL import Image

    from .timing import milliseconds_since

    content: list[dict[str, Any]] = []
    image_decode_ms: float | None = None
    if case.get("image"):
        start = time.perf_counter()
        with Image.open(Path(case["image"])) as source:
            image = source.convert("RGB")
        image_decode_ms = milliseconds_since(start)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    set_pixel_budget(processor, case)
    start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Some processor versions return a field unused by this model's forward.
    inputs.pop("token_type_ids", None)
    return inputs, {"image_decode_ms": image_decode_ms, "processor_ms": milliseconds_since(start)}


def transfer_inputs(inputs: Any, device: str, torch: Any) -> tuple[dict[str, Any], float]:
    """Measure host-to-device transfer, synchronized to include actual copy time."""

    from .timing import milliseconds_since

    start = time.perf_counter()
    moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    torch.cuda.synchronize(device)
    return moved, milliseconds_since(start)


def input_counts(inputs: dict[str, Any], model: Any) -> dict[str, int]:
    grid = inputs.get("image_grid_thw")
    merge = int(model.config.vision_config.spatial_merge_size)
    image_tokens = int((grid.prod(dim=-1) // (merge * merge)).sum().item()) if grid is not None else 0
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "image_tokens": image_tokens,
        "raw_image_patches": int(inputs["pixel_values"].shape[0]) if "pixel_values" in inputs else 0,
    }


def action_token_ids(processor: Any, labels: list[str]) -> list[int]:
    """Require actual single-token alternatives for the constrained generation mode."""

    if not labels:
        raise ValueError("action_labels must contain at least one label")
    ids = []
    for label in labels:
        token_ids = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Action label {label!r} is not one token; choose single-token labels")
        ids.append(token_ids[0])
    if len(set(ids)) != len(ids):
        raise ValueError("action_labels map to duplicate token IDs")
    return ids


def create_readout(model: Any, torch: Any, device: str, classes: int) -> Any:
    """An untrained head: valid for latency and memory only."""

    head = torch.nn.Linear(int(model.config.text_config.hidden_size), classes, bias=True)
    head.to(device=device, dtype=next(model.parameters()).dtype)
    head.eval()
    return head


def decision_forward(
    model: Any,
    inputs: dict[str, Any],
    head: Any | None,
    torch: Any,
    *,
    measure_head: bool = False,
) -> tuple[Any, Any | None]:
    """Return a final hidden state without paying for the large vocabulary projection."""

    outputs = model.model(**inputs, use_cache=False)
    final_hidden = outputs.last_hidden_state[:, -1, :]
    if head is None:
        return final_hidden, None
    if not measure_head:
        return head(final_hidden), None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = head(final_hidden)
    end.record()
    # The outer request synchronizes before this elapsed time is queried.
    return result, (start, end)


def generate_one_token(model: Any, inputs: dict[str, Any], allowed_ids: list[int]) -> Any:
    """HF generate with a true one-token constrained vocabulary."""

    return model.generate(
        **inputs,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        use_cache=True,
        prefix_allowed_tokens_fn=lambda _batch_id, _ids: allowed_ids,
    )


def generate_json(model: Any, inputs: dict[str, Any], max_new_tokens: int) -> Any:
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
