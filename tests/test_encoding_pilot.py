"""Pilot source selection tests; real encoder timing requires HPC CUDA."""

import json
import tempfile
import unittest
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.encoding_pilot import load_prefix, load_sample
from ltr_system.project import write_json


class PilotTests(unittest.TestCase):
    def sample_fixture(self, root):
        path = root / "sample.jsonl"
        path.write_text('{"id":"42","contents":"text"}\n', encoding="utf-8")
        config = {"repo_id": "fixture", "revision": "pinned", "max_length": 512}
        write_json(
            root / "manifest.json",
            {
                **config,
                "sample_sha256": sha256_file(path),
                "statistics": {"count": 1},
                "source_manifest_sha256": "source",
            },
        )
        return config

    def test_saved_sample_identity_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.sample_fixture(root)
            ids, texts, identity = load_sample(root, 1, config)
            self.assertEqual(ids, ["42"])
            self.assertEqual(texts, ["passage: text"])
            self.assertEqual(identity["sample_sha256"], sha256_file(root / "sample.jsonl"))
            self.assertEqual(identity["source_corpus_manifest_sha256"], "source")

    def test_rejects_changed_sample_model_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.sample_fixture(root)
            with self.assertRaisesRegex(ValueError, "complete saved sample"):
                load_sample(root, 2, config)
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_sample(root, 1, {**config, "revision": "different"})
            (root / "sample.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_sample(root, 1, config)

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
