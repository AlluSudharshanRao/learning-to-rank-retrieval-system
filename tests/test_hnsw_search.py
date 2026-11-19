"""Neighbor agreement semantics and full HNSW validation integration."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.dense_search import run as exact_run
from ltr_system.hnsw_search import neighbor_agreement, run
from ltr_system.project import read_json, write_json


class AgreementTests(unittest.TestCase):
    def test_overlap_is_not_judged_relevance(self):
        result = neighbor_agreement(
            ["q"], {"q": ["b", "x", "a"]}, {"q": ["a", "b", "c"]}, (1, 2, 3)
        )
        self.assertEqual(
            result["metrics"],
            {"neighbor_recall@1": 0, "neighbor_recall@2": 0.5, "neighbor_recall@3": 2 / 3},
        )

    def test_population_depth_and_duplicates_rejected(self):
        for a, b, cutoffs in (
            ({"x": ["a"]}, {"q": ["a"]}, (1,)),
            ({"q": ["a"]}, {"q": ["a"]}, (2,)),
            ({"q": ["a", "a"]}, {"q": ["a", "b"]}, (1,)),
        ):
            with self.subTest(a=a), self.assertRaises(ValueError):
                neighbor_agreement(["q"], a, b, cutoffs)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class HnswSearchTests(unittest.TestCase):
    def fixture(self, root):
        import faiss

        flat_root = root / "artifacts/indexes/e5-flat-ip-v1"
        graph_root = root / "artifacts/indexes/e5-hnsw-m32-efc200-v1"
        query_root = root / "artifacts/queries/e5-validation-fp32"
        split = root / "data/processed/msmarco-passage-v1-splits"
        for path in (flat_root, graph_root, query_root, split):
            path.mkdir(parents=True)
        faiss.omp_set_num_threads(1)
        vectors = np.random.default_rng(42).normal(size=(1001, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = np.arange(1001, dtype=np.int64) + 100
        flat = faiss.IndexIDMap2(faiss.IndexFlatIP(16))
        flat.add_with_ids(vectors, ids)
        graph = faiss.IndexHNSWFlat(16, 32, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = 200
        mapped = faiss.IndexIDMap2(graph)
        mapped.add_with_ids(vectors, ids)
        for directory, index, filename, status, name in (
            (
                flat_root,
                flat,
                "exact.faiss",
                "exact index built and reload checked",
                "IndexIDMap2(IndexFlatIP)",
            ),
            (
                graph_root,
                mapped,
                "hnsw.faiss",
                "full HNSW index built and reload checked",
                "IndexIDMap2(IndexHNSWFlat)",
            ),
        ):
            faiss.write_index(index, str(directory / filename))
            write_json(
                directory / "manifest.json",
                {
                    "status": status,
                    "index_type": name,
                    "metric": "inner product",
                    "vectors": 1001,
                    "dimension": 16,
                    "coverage_sha256": "fixture-coverage",
                    "embedding_configuration_sha256": "fixture-configuration",
                    "hnsw": {"M": 32, "efConstruction": 200},
                    "files": {filename: {"sha256": sha256_file(directory / filename)}},
                },
            )
        (split / "validation.queries.tsv").write_text("q1\tfirst\nq2\tsecond\n", encoding="utf-8")
        (split / "validation.qrels.tsv").write_text(
            "q1\t0\t100\t1\nq2\t0\t102\t1\n", encoding="utf-8"
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
        np.save(query_root / "vectors.npy", vectors[[0, 2]])
        write_json(query_root / "query_ids.json", {"query_ids": ["q1", "q2"]})
        write_json(
            query_root / "manifest.json",
            {
                "status": "validation queries encoded",
                "queries": 2,
                "encoder": {"dimension": 16},
                "passage_configuration_sha256": "fixture-configuration",
                "prefix": "query: ",
                "precision": "float32",
                "pooling": "masked mean",
                "normalization": "L2",
                "source": {
                    "split": "validation",
                    "query_file_sha256": sha256_file(split / "validation.queries.tsv"),
                    "split_manifest_sha256": sha256_file(split / "manifest.json"),
                },
                "files": {
                    name: {"sha256": sha256_file(query_root / name)}
                    for name in ("vectors.npy", "query_ids.json")
                },
            },
        )
        exact_run(root, "e5-exact-validation-default", 1000, 2, 1)

    def test_saved_exact_baseline_to_hnsw_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = run(root, 1024, 2, 1)
            self.assertEqual(result["query_count"], 2)
            self.assertEqual(result["metrics"]["mrr@10"], 1.0)
            saved = root / "artifacts/runs/e5-hnsw-validation-ef1024-b2-t1"
            neighbors = read_json(saved / "neighbors.json")
            self.assertEqual(neighbors["queries"], 2)
            self.assertTrue(all(0 <= value <= 1 for value in neighbors["metrics"].values()))
            self.assertEqual(
                read_json(saved / "manifest.json")["counts"]["returned_passages"], 2000
            )
            with self.assertRaises(FileExistsError):
                run(root, 1024, 2, 1)

    def test_changed_exact_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            ranking = root / "artifacts/runs/e5-exact-validation-default/ranking.tsv"
            ranking.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(root, 1024, 2, 1)
            self.assertFalse((root / "artifacts/runs/e5-hnsw-validation-ef1024-b2-t1").exists())
