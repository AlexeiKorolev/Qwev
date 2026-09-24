"""Checks that fail fast before allocating a GPU or loading model weights."""

from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwev_bench.modeling import action_token_ids, decision_forward, set_pixel_budget
from qwev_bench.runner import _validate
from qwev_bench.timing import StageProfiler


class ConfigTests(unittest.TestCase):
    def test_rejects_missing_image_before_gpu_load(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _validate({"model": "Qwen/Qwen3-VL-2B-Instruct", "cases": [
                {"id": "test", "prompt": "Choose an action", "image": "/definitely/missing.png"}
            ]})

    def test_rejects_duplicate_case_ids(self) -> None:
        with self.assertRaises(ValueError):
            _validate({"model": "example", "cases": [
                {"id": "same", "prompt": "one"}, {"id": "same", "prompt": "two"}
            ]})

    def test_accepts_text_only_case(self) -> None:
        _validate({"model": "example", "cases": [{"id": "text", "prompt": "Choose an action"}]})


class ProcessorTests(unittest.TestCase):
    def test_pixel_budget_does_not_bleed_between_cases(self) -> None:
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 1048576}),
            _qwev_default_image_size={"shortest_edge": 4096, "longest_edge": 1048576},
        )
        set_pixel_budget(processor, {"image": "example.png", "min_pixels": 16384, "max_pixels": 65536})
        self.assertEqual(processor.image_processor.size["longest_edge"], 65536)
        set_pixel_budget(processor, {"image": "second.png"})
        self.assertEqual(processor.image_processor.size["longest_edge"], 1048576)

    def test_action_labels_must_be_single_distinct_tokens(self) -> None:
        class FakeTokenizer:
            def encode(self, label: str, add_special_tokens: bool = False) -> list[int]:
                return {"A": [1], "B": [2], "two words": [3, 4], "alias": [1]}[label]

        processor = SimpleNamespace(tokenizer=FakeTokenizer())
        self.assertEqual(action_token_ids(processor, ["A", "B"]), [1, 2])
        with self.assertRaises(ValueError):
            action_token_ids(processor, ["A", "two words"])
        with self.assertRaises(ValueError):
            action_token_ids(processor, ["A", "alias"])


class ModelPathTests(unittest.TestCase):
    def test_decision_forward_skips_vocabulary_projection(self) -> None:
        class Hidden:
            def __getitem__(self, key: object) -> str:
                self_key = key
                assert isinstance(self_key, tuple) and self_key[1] == -1
                return "last hidden state"

        class Core:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                self.kwargs = kwargs
                return SimpleNamespace(last_hidden_state=Hidden())

        core = Core()
        model = SimpleNamespace(model=core, lm_head=lambda _: self.fail("vocabulary projection was called"))
        result, head_timing = decision_forward(model, {"input_ids": "ids"}, None, None)
        self.assertEqual(result, "last hidden state")
        self.assertIsNone(head_timing)
        self.assertIs(core.kwargs["use_cache"], False)

    def test_primary_head_path_has_no_cuda_event_instrumentation(self) -> None:
        class Hidden:
            def __getitem__(self, key: object) -> str:
                return "last hidden state"

        core = lambda **kwargs: SimpleNamespace(last_hidden_state=Hidden())

        class NoEvents:
            class cuda:
                @staticmethod
                def Event(**kwargs: object) -> None:
                    raise AssertionError("primary path should not record stage events")

        readout, timing = decision_forward(
            SimpleNamespace(model=core), {"input_ids": "ids"},
            lambda hidden: f"head({hidden})", NoEvents(),
        )
        self.assertEqual(readout, "head(last hidden state)")
        self.assertIsNone(timing)

    def test_language_prefill_and_decode_stages_are_separate(self) -> None:
        class Event:
            def __init__(self, when: float):
                self.when = when

            def elapsed_time(self, other: "Event") -> float:
                return other.when - self.when

        profiler = StageProfiler.__new__(StageProfiler)
        profiler.events = defaultdict(list, {
            "language": [(Event(0), Event(12)), (Event(12), Event(15)), (Event(15), Event(19))]
        })
        self.assertEqual(profiler.first_elapsed_ms("language"), 12)
        self.assertEqual(profiler.remaining_elapsed_ms("language"), 7)
        self.assertEqual(profiler.elapsed_ms("language"), 19)


if __name__ == "__main__":
    unittest.main()
