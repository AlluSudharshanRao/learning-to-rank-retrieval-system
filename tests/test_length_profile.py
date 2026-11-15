"""Sampling and length-statistic checks independent of tokenizer downloads."""

import json
import tempfile
import unittest
from pathlib import Path

from ltr_system.length_profile import sample_records, summarize


class LengthProfileTests(unittest.TestCase):
    def test_statistics_and_exact_truncation_boundary(self):
        result = summarize([10, 20, 512, 513], 512)
        self.assertEqual(result["mean"], 263.75)
        self.assertEqual(result["percentiles_nearest_rank"]["50"], 20)
        self.assertEqual(result["fraction_above_limit"], 0.25)
        self.assertEqual(result["above_limit"], 1)
        with self.assertRaises(ValueError):
            summarize([], 512)

    def test_sampling_repeatability_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            entries = []
            for shard in range(2):
                name = f"part-{shard}.jsonl"
                (root / "shards" / name).write_text(
                    "\n".join(
                        json.dumps({"id": str(shard * 5 + i), "contents": "text"}) for i in range(5)
                    ),
                    encoding="utf-8",
                )
                entries.append({"name": name, "records": 5})
            manifest = {"records": 10, "shards": entries}
            first = sample_records(root, manifest, 4, 2026)
            self.assertEqual(first, sample_records(root, manifest, 4, 2026))
            self.assertEqual(len({r["id"] for r in first}), 4)
            self.assertEqual(
                [r["id"] for r in sample_records(root, manifest, 10, 2026)],
                [str(i) for i in range(10)],
            )
            with self.assertRaises(ValueError):
                sample_records(root, manifest, 11, 2026)
            entries[0]["records"] = 4
            with self.assertRaisesRegex(ValueError, "line count"):
                sample_records(root, manifest, 4, 2026)
