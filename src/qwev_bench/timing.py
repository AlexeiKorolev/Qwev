"""CUDA measurements for one complete, synchronous inference request.

CUDA events measure device elapsed time. Wall time additionally includes Python,
kernel launch, and synchronization overhead. Module events are observational:
they do not synchronize inside a model forward pass.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable


def milliseconds_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class StageProfiler:
    """Record CUDA events around Qwen's vision and language modules."""

    def __init__(self, model: Any, torch: Any):
        self.torch = torch
        self.events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self.active: dict[str, list[Any]] = defaultdict(list)
        self.handles = []
        for name, module in (
            ("vision", model.model.visual),
            ("language", model.model.language_model),
        ):
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any) -> None:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.active[name].append(event)

        return hook

    def _post(self, name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            start = self.active[name].pop()
            self.events[name].append((start, end))

        return hook

    def elapsed_ms(self, name: str) -> float | None:
        intervals = self.events[name]
        if not intervals:
            return None
        return sum(start.elapsed_time(end) for start, end in intervals)

    def first_elapsed_ms(self, name: str) -> float | None:
        intervals = self.events[name]
        if not intervals:
            return None
        start, end = intervals[0]
        return start.elapsed_time(end)

    def remaining_elapsed_ms(self, name: str) -> float | None:
        intervals = self.events[name]
        if not intervals:
            return None
        return sum(start.elapsed_time(end) for start, end in intervals[1:])

    def count(self, name: str) -> int:
        return len(self.events[name])

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def synchronized_call(torch: Any, device: str, operation: Callable[[], Any]) -> dict[str, Any]:
    """Time a request with a clean CUDA stream and return its output.

    The caller owns all work after the returned output (for example token decode).
    The initial synchronization keeps previous work out of this request's wall time.
    """

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output = operation()
    end_event.record()
    torch.cuda.synchronize(device)
    return {
        "output": output,
        "request_wall_ms": milliseconds_since(start_wall),
        "request_cuda_ms": start_event.elapsed_time(end_event),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
