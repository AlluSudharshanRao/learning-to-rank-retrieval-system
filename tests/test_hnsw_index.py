"""Full HNSW builder checks across multiple independently mapped shards."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.hnsw_index import build
from ltr_system.project import read_json, write_json


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class HnswIndexTests(unittest.TestCase):
    def fixture(self, root):
        embeddings = root / "embeddings"
        embeddings.mkdir()
        config = {"dimension": 3, "normalization": "L2", "precision": "float32"}
        write_json(embeddings / "configuration.json", config)
        entries = []
        for number, (vectors, ids) in enumerate(
            (
                (np.eye(3, dtype=np.float32), ["10", "50", "90"]),
                (-np.eye(3, dtype=np.float32), ["100", "500", "900"]),
            )
        ):
            shard = embeddings / f"part-{number:05d}"
            shard.mkdir()
            np.save(shard / "vectors.npy", vectors)
            write_json(shard / "passage_ids.json", {"passage_ids": ids})
            write_json(
                shard / "manifest.json",
                {
                    "status": "complete",
                    "identity": config,
                    "records": 3,
                    "files": {
                        name: {"sha256": sha256_file(shard / name)}
                        for name in ("vectors.npy", "passage_ids.json")
                    },
                },
            )
            entries.append(
                {
                    "name": shard.name,
                    "records": 3,
                    "manifest_sha256": sha256_file(shard / "manifest.json"),
                }
            )
        write_json(
            embeddings / "coverage.json",
            {
                "status": "all embedding shards verified",
                "ordered_source_id_alignment": True,
                "records": 6,
                "dimension": 3,
                "shard_count": 2,
                "shards": entries,
                "configuration_sha256": sha256_file(embeddings / "configuration.json"),
            },
        )
        return embeddings

    def test_two_shards_reload_external_ids_and_no_overwrite(self):
        import faiss

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            output = root / "index"
            report = build(embeddings, output, 4, 20, 1)
            self.assertEqual(report["vectors"], 6)
            self.assertEqual(report["reconstruction_checks"], 2)
            index = faiss.read_index(str(output / "hnsw.faiss"))
            values, ids = index.search(np.array([[-1, 0, 0]], np.float32), 1)
            np.testing.assert_array_equal(ids, [[100]])
            np.testing.assert_allclose(values, [[1]])
            with self.assertRaises(FileExistsError):
                build(embeddings, output, 4, 20, 1)

    def test_later_shard_tamper_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            (embeddings / "part-00001/passage_ids.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                build(embeddings, root / "index", 4, 20, 1)
            self.assertFalse((root / "index").exists())
            self.assertFalse((embeddings / ".writer.lock").exists())

    def test_coverage_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            report = read_json(embeddings / "coverage.json")
            report["records"] = 7
            write_json(embeddings / "coverage.json", report)
            with self.assertRaisesRegex(ValueError, "counts"):
                build(embeddings, root / "index", 4, 20, 1)
            self.assertFalse((root / "index").exists())
