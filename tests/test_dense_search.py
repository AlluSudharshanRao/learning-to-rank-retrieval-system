"""Exact search correctness and artifact/split alignment gates."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.dense_search import retrieve, run
from ltr_system.project import read_json, write_json


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class DenseSearchTests(unittest.TestCase):
    def fixture(self, root):
        import faiss

        index_root = root / "artifacts/indexes/e5-flat-ip-v1"
        query_root = root / "artifacts/queries/e5-validation-fp32"
        split = root / "data/processed/msmarco-passage-v1-splits"
        for directory in (index_root, query_root, split):
            directory.mkdir(parents=True)
        passages = np.array([[1, 0], [0, 1], [-1, 0]], np.float32)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        index.add_with_ids(passages, np.array([101, 103, 109], np.int64))
        faiss.write_index(index, str(index_root / "exact.faiss"))
        write_json(
            index_root / "manifest.json",
            {
                "status": "exact index built and reload checked",
                "index_type": "IndexIDMap2(IndexFlatIP)",
                "metric": "inner product",
                "vectors": 3,
                "dimension": 2,
                "embedding_configuration_sha256": "fixture",
                "files": {"exact.faiss": {"sha256": sha256_file(index_root / "exact.faiss")}},
            },
        )
        (split / "validation.queries.tsv").write_text("q2\tsecond\nq1\tfirst\n", encoding="utf-8")
        (split / "validation.qrels.tsv").write_text(
            "q2\t0\t103\t1\nq1\t0\t101\t1\n", encoding="utf-8"
        )
        write_json(
            split / "manifest.json",
            {
                "counts": {"actual_validation_queries": 2},
                "outputs": {
                    name: {"sha256": sha256_file(split / name)}
                    for name in ("validation.queries.tsv", "validation.qrels.tsv")
                },
            },
        )
        np.save(query_root / "vectors.npy", passages[[1, 0]])
        write_json(query_root / "query_ids.json", {"query_ids": ["q2", "q1"]})
        write_json(
            query_root / "manifest.json",
            {
                "status": "validation queries encoded",
                "queries": 2,
                "encoder": {"dimension": 2},
                "passage_configuration_sha256": "fixture",
                "prefix": "query: ",
                "precision": "float32",
                "pooling": "masked mean",
                "normalization": "L2",
                "source": {
                    "split": "validation",
                    "split_manifest_sha256": sha256_file(split / "manifest.json"),
                    "query_file_sha256": sha256_file(split / "validation.queries.tsv"),
                },
                "files": {
                    name: {"sha256": sha256_file(query_root / name)}
                    for name in ("vectors.npy", "query_ids.json")
                },
            },
        )
        return index, passages, query_root

    def test_complete_run_and_batch_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, passages, _ = self.fixture(root)
            for batch in (1, 2):
                report = run(root, f"batch{batch}", 3, batch, 1)
                self.assertEqual(report["metrics"]["mrr@10"], 1.0)
                self.assertEqual(report["missing_rankings"], 0)
            outputs = root / "artifacts/runs"
            self.assertEqual(
                (outputs / "batch1/ranking.tsv").read_bytes(),
                (outputs / "batch2/ranking.tsv").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                run(root, "batch1", 3, 1, 1)
            with self.assertRaises(ValueError):
                retrieve(index, passages, ["a", "b", "c"], root / "invalid.tsv", 4, 1)

    def test_changed_vectors_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, query_root = self.fixture(root)
            np.save(query_root / "vectors.npy", np.zeros((2, 2), np.float32))
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(root, "invalid", 3, 1, 1)
            self.assertFalse((root / "artifacts/runs/invalid").exists())

    def test_reordered_ids_rejected_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, query_root = self.fixture(root)
            write_json(query_root / "query_ids.json", {"query_ids": ["q1", "q2"]})
            report = read_json(query_root / "manifest.json")
            report["files"]["query_ids.json"]["sha256"] = sha256_file(query_root / "query_ids.json")
            write_json(query_root / "manifest.json", report)
            with self.assertRaisesRegex(ValueError, "order"):
                run(root, "invalid", 3, 1, 1)
            self.assertFalse((root / "artifacts/runs/invalid").exists())

    def test_unnormalized_vectors_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "normalized"):
                retrieve(index, np.array([[2, 0]], np.float32), ["q"], root / "bad.tsv", 1, 1)
