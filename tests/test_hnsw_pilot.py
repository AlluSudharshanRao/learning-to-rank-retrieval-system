"""HNSW persistence, external IDs, scores and audited-input rejection."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.hnsw_pilot import build_pilot
from ltr_system.project import write_json


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class HnswPilotTests(unittest.TestCase):
    def fixture(self, root):
        embeddings = root / "embeddings"
        shard = embeddings / "part-00000"
        shard.mkdir(parents=True)
        config = {"dimension": 3, "normalization": "L2", "precision": "float32"}
        write_json(embeddings / "configuration.json", config)
        np.save(shard / "vectors.npy", np.eye(3, dtype=np.float32))
        write_json(shard / "passage_ids.json", {"passage_ids": ["101", "503", "999"]})
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
                "ordered_source_id_alignment": True,
                "dimension": 3,
                "configuration_sha256": sha256_file(embeddings / "configuration.json"),
                "shards": [
                    {
                        "name": "part-00000",
                        "records": 3,
                        "manifest_sha256": sha256_file(shard / "manifest.json"),
                    }
                ],
            },
        )
        return embeddings

    def test_build_reload_mapping_and_scores(self):
        import faiss

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            output = root / "index"
            report = build_pilot(embeddings, output, 4, 20, 1)
            self.assertEqual(report["vectors"], 3)
            self.assertEqual(report["probe_neighbor_overlap"], 1.0)
            restored = faiss.read_index(str(output / "hnsw.faiss"))
            scores, ids = restored.search(np.array([[1, 0, 0]], np.float32), 1)
            np.testing.assert_array_equal(ids, [[101]])
            np.testing.assert_allclose(scores, [[1]])
            self.assertEqual(
                sha256_file(output / "hnsw.faiss"), report["files"]["hnsw.faiss"]["sha256"]
            )
            with self.assertRaises(FileExistsError):
                build_pilot(embeddings, output, 4, 20, 1)

    def test_changed_vector_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = self.fixture(root)
            np.save(embeddings / "part-00000/vectors.npy", np.zeros((3, 3), np.float32))
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_pilot(embeddings, root / "index", 4, 20, 1)
            self.assertFalse((root / "index").exists())
            self.assertFalse((embeddings / ".writer.lock").exists())

    def test_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for m, ef, threads in ((1, 200, 1), (32, 16, 1), (32, 200, 0)):
                with self.subTest(m=m, ef=ef, threads=threads), self.assertRaises(ValueError):
                    build_pilot(root / "absent", root / "index", m, ef, threads)
