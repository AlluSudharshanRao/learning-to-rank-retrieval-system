"""Index input identity checks; Lucene execution is validated separately on HPC."""

import tempfile
import unittest
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.lexical_index import verify_corpus
from ltr_system.project import read_json, write_json


class IndexInputTests(unittest.TestCase):
    def fixture(self, root):
        (root / "shards").mkdir()
        path = root / "shards/part-00000.jsonl"
        path.write_text('{"id":"0","contents":"text"}\n', encoding="utf-8")
        report = {
            "status": "corpus converted; Lucene index not built",
            "records": 1,
            "shards": [
                {
                    "name": path.name,
                    "records": 1,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            ],
        }
        write_json(root / "manifest.json", report)
        return report

    def test_valid_and_tampered_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            self.assertEqual(verify_corpus(root), report)
            (root / "shards/part-00000.jsonl").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_corpus(root)

    def test_unlisted_shard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "shards/extra.jsonl").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "directory differs"):
                verify_corpus(root)

    def test_manifest_counts_and_names(self):
        for modification in ("duplicate", "count", "traversal"):
            with (
                self.subTest(modification=modification),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.fixture(root)
                report = read_json(root / "manifest.json")
                if modification == "duplicate":
                    report["shards"].append(report["shards"][0].copy())
                elif modification == "count":
                    report["records"] = 2
                else:
                    report["shards"][0]["name"] = "../escape.jsonl"
                write_json(root / "manifest.json", report)
                with self.assertRaises(ValueError):
                    verify_corpus(root)
