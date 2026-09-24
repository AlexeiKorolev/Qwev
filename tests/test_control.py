import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from qwev_bench.config import load_config
from qwev_bench.fixtures import generate_fixtures
from qwev_bench.reporting import percentile, summarize_files


class ConfigTests(unittest.TestCase):
    def test_relative_inputs_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"image")
            path = root / "config.json"
            path.write_text(json.dumps({"model": "old", "cases": [{"id": "test", "image": "image.png", "prompt": "act"}]}))
            config = load_config(path, model="new")
            self.assertEqual(config["model"], "new")
            self.assertEqual(config["cases"][0]["image"], str((root / "image.png").resolve()))
            self.assertTrue(config["local_files_only"])

    def test_reject_invalid_cases_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            base = {"model": "model", "cases": [{"id": "test", "prompt": "act"}]}
            for patch in [{"warmup": 0}, {"repeats": True}, {"device": "cpu"}, {"modes": ["unknown"]},
                          {"cases": [{"id": "x", "prompt": "a"}, {"id": "x", "prompt": "b"}]},
                          {"cases": [{"id": "x", "prompt": "a", "image": "missing.png"}]}]:
                path.write_text(json.dumps(dict(base, **patch)))
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    load_config(path)


class ReportingTests(unittest.TestCase):
    def test_filters_warmups_errors_and_separates_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.jsonl", root / "two.jsonl"
            sample = {"type": "sample", "model": "m", "case_id": "c", "mode": "head", "run_id": "r", "status": "ok", "warmup": False}
            records = [dict(sample, request_wall_ms=10), dict(sample, request_wall_ms=20),
                       dict(sample, warmup=True, request_wall_ms=1000),
                       dict(sample, status="error", error_type="OOM", request_wall_ms=999)]
            first.write_text("\n".join(map(json.dumps, records)))
            second.write_text(json.dumps(dict(sample, run_id="r2", request_wall_ms=100)))
            output = root / "summary.csv"
            self.assertEqual(summarize_files([str(root / "*.jsonl")], output), 2)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["request_wall_ms_p50"], "15.0")
            self.assertEqual(rows[0]["failed"], "1")
            self.assertEqual(rows[1]["request_wall_ms_p50"], "100.0")
            with self.assertRaises(FileExistsError):
                summarize_files([str(first)], output)

    def test_failed_group_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "failed.jsonl"
            result.write_text(json.dumps({"type": "sample", "status": "error", "error_type": "OOM"}))
            output = root / "out.csv"
            summarize_files([str(result)], output)
            with output.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["successful"], "0")
            self.assertEqual(row["request_wall_ms_p50"], "")

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        self.assertIsNone(percentile([], .95))


class FixtureTests(unittest.TestCase):
    def test_png_dimensions_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = generate_fixtures(root)
            original = [p.read_bytes() for p in files]
            self.assertEqual(struct.unpack(">II", original[0][16:24]), (1024, 1024))
            self.assertEqual(struct.unpack(">II", original[1][16:24]), (1280, 800))
            generate_fixtures(root)
            self.assertEqual(original, [p.read_bytes() for p in files])


if __name__ == "__main__":
    unittest.main()
