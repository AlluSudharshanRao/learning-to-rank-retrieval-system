"""Pilot source selection tests; real encoder timing requires HPC CUDA."""

import json
import tempfile
import unittest
from pathlib import Path

from ltr_system.encoding_pilot import load_prefix


class PilotTests(unittest.TestCase):
    def test_prefix_preserves_ids_and_adds_passage_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl"
            path.write_text(
                "\n".join(json.dumps({"id": str(i), "contents": f"text {i}"}) for i in range(3)),
                encoding="utf-8",
            )
            ids, texts = load_prefix(path, 2)
            self.assertEqual(ids, ["0", "1"])
            self.assertEqual(texts, ["passage: text 0", "passage: text 1"])

    def test_rejects_insufficient_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl"
            row = json.dumps({"id": "0", "contents": "text"}) + "\n"
            path.write_text(row, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prefix(path, 2)
            path.write_text(row * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prefix(path, 2)
