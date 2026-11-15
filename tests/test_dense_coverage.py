"""Coverage rejects incomplete sets and reordered IDs even with updated checksums."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.dense_coverage import audit
from ltr_system.dense_encode import encode_shard
from ltr_system.project import read_json, write_json


class CoverageTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        (source / "shards").mkdir(parents=True)
        path = source / "shards/part-00000.jsonl"
        path.write_text(
            '{"id":"0","contents":"first"}\n{"id":"1","contents":"second"}\n', encoding="utf-8"
        )
        entry = {
            "name": path.name,
            "records": 2,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        write_json(
            source / "manifest.json",
            {
                "status": "corpus converted; Lucene index not built",
                "records": 2,
                "shards": [entry],
            },
        )
        output = root / "embeddings"
        output.mkdir()
        config = {
            "dimension": 2,
            "batch_size": 2,
            "source_corpus_manifest_sha256": sha256_file(source / "manifest.json"),
        }
        write_json(output / "configuration.json", config)
        encode_shard(
            path,
            entry,
            output / "part-00000",
            config,
            lambda texts: np.tile(np.array([[1, 0]], np.float32), (len(texts), 1)),
        )
        return source, output

    def test_complete_and_repeatable_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = self.fixture(Path(directory))
            result = audit(source, output)
            self.assertEqual(result["records"], 2)
            self.assertTrue(result["ordered_source_id_alignment"])
            self.assertEqual(audit(source, output), result)

    def test_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = self.fixture(Path(directory))
            (output / "part-00000").rename(output / "misplaced")
            with self.assertRaisesRegex(ValueError, "Missing"):
                audit(source, output)
            self.assertFalse((output / "coverage.json").exists())

    def test_reordered_ids_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = self.fixture(Path(directory))
            ids = output / "part-00000/passage_ids.json"
            write_json(ids, {"passage_ids": ["1", "0"]})
            manifest = output / "part-00000/manifest.json"
            report = read_json(manifest)
            report["files"]["passage_ids.json"] = {
                "sha256": sha256_file(ids),
                "bytes": ids.stat().st_size,
            }
            write_json(manifest, report)
            with self.assertRaisesRegex(ValueError, "alignment"):
                audit(source, output)
