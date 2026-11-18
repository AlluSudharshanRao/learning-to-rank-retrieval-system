"""Exact index persistence, ID mapping and audit-identity enforcement."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.dense_index import build, numeric_ids
from ltr_system.project import write_json


class NumericIDTests(unittest.TestCase):
    def test_roundtrip_and_invalid_ids(self):
        np.testing.assert_array_equal(numeric_ids(["0", "42"]), [0, 42])
        for ids in (["01"], ["-1"], [str(2**63)], ["1", "1"], ["abc"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                numeric_ids(ids)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class DenseIndexTests(unittest.TestCase):
    def fixture(self, root):
        embeddings = root / "embeddings"
        shard = embeddings / "part-00000"
        shard.mkdir(parents=True)
        vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        np.save(shard / "vectors.npy", vectors)
        write_json(shard / "passage_ids.json", {"passage_ids": ["101", "103", "109"]})
        config = {"dimension": 2, "precision": "float32"}
        write_json(embeddings / "configuration.json", config)
        write_json(
            shard / "manifest.json",
            {
                "status": "complete",
                "records": 3,
                "identity": config,
                "files": {
                    name: {"sha256": sha256_file(shard / name)}
                    for name in ("vectors.npy", "passage_ids.json")
                },
            },
        )
        write_json(
            embeddings / "coverage.json",
            {
                "status": "all embedding shards verified",
                "records": 3,
                "shard_count": 1,
                "dimension": 2,
                "configuration_sha256": sha256_file(embeddings / "configuration.json"),
                "shards": [
                    {
                        "name": shard.name,
                        "records": 3,
                        "manifest_sha256": sha256_file(shard / "manifest.json"),
                    }
                ],
            },
        )
        return embeddings

    def test_exact_results_and_reload(self):
        import faiss

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            output = root / "index"
            report = build(embeddings, output, 1)
            self.assertEqual(report["vectors"], 3)
            index = faiss.read_index(str(output / "exact.faiss"))
            scores, ids = index.search(np.array([[1, 0]], np.float32), 3)
            np.testing.assert_array_equal(ids, [[101, 103, 109]])
            np.testing.assert_allclose(scores, [[1, 0, -1]])
            with self.assertRaises(FileExistsError):
                build(embeddings, output, 1)

    def test_modified_embedding_rejected_without_published_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            (embeddings / "part-00000/passage_ids.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                build(embeddings, root / "index", 1)
            self.assertFalse((root / "index").exists())
