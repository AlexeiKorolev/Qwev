"""Dependency-free configuration validation (also used for dry runs)."""

import json
from pathlib import Path


MODES = {"prefill", "head", "one_token", "json"}


def load_config(path, model=None):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if model:
        config["model"] = model
    defaults = {
        "revision": "main", "dtype": "bfloat16", "device": "cuda:0",
        "attention_implementation": "sdpa", "warmup": 2, "repeats": 10,
        "seed": 42, "modes": ["prefill", "head", "one_token", "json"],
        "max_new_tokens": 32, "local_files_only": True,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("model must be a nonempty Hugging Face model ID or local path")
    if config["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    if not isinstance(config["device"], str) or not config["device"].startswith("cuda"):
        raise ValueError("Measured runs require a CUDA device; use --dry-run for CPU validation")
    for key, minimum in [("warmup", 1), ("repeats", 1), ("max_new_tokens", 1), ("seed", 0)]:
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    modes = config["modes"]
    if not isinstance(modes, list) or not modes or any(mode not in MODES for mode in modes):
        raise ValueError(f"modes must be a nonempty list drawn from {sorted(MODES)}")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")
    if type(config["local_files_only"]) is not bool:
        raise ValueError("local_files_only must be true or false")
    labels = config.get("action_labels", list("ABCDEFG"))
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("action_labels must be a nonempty list of nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("action_labels must be distinct")
    if config["attention_implementation"] not in {"sdpa", "eager", "flash_attention_2"}:
        raise ValueError("Unsupported attention_implementation")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Each case needs a unique nonempty string id")
        identifiers.add(identifier)
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"Case {identifier}: prompt must be a nonempty string")
        if "json_prompt" in case and (not isinstance(case["json_prompt"], str) or not case["json_prompt"].strip()):
            raise ValueError(f"Case {identifier}: json_prompt must be a nonempty string")
        if case.get("image") is not None:
            if not isinstance(case["image"], str) or not case["image"]:
                raise ValueError(f"Case {identifier}: image must be a path or null")
            image = Path(case["image"]).expanduser()
            image = image if image.is_absolute() else path.parent / image
            image = image.resolve()
            if not image.is_file():
                raise ValueError(f"Case {identifier}: image does not exist: {image}. Run the fixtures command first.")
            case["image"] = str(image)
        for key, default in [("min_pixels", 4096), ("max_pixels", 262144)]:
            case.setdefault(key, default)
            if type(case[key]) is not int or case[key] <= 0:
                raise ValueError(f"Case {identifier}: {key} must be a positive integer")
        if case["min_pixels"] > case["max_pixels"]:
            raise ValueError(f"Case {identifier}: min_pixels exceeds max_pixels")
    config["config_path"] = str(path)
    return config
