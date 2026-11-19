"""Reporting rejects inconsistent source summaries and mismatched artifacts."""

import tempfile
import unittest
from pathlib import Path

from ltr_system.benchmark_plots import LABELS, RUNS, evidence
from ltr_system.data_download import sha256_file
from ltr_system.project import read_json, write_json
from ltr_system.search_benchmark import schedule, summarize


class PlotEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        source = {"split": "validation", "query_file_sha256": "query-hash"}
        benchmark = {
            "status": "matched CPU search benchmark completed",
            "results": {},
            "batch_size": 32,
            "k": 1000,
            "threads": 4,
            "repeats": 4,
            "schedule": schedule(2, 1, 4, LABELS, 2026),
            "batch_queries": 2,
            "latency_unique_queries": 1,
            "query_ids_in_vector_order": ["q1", "q2"],
            "population": source,
            "final_queries_read": False,
            "seed": 2026,
            "metadata": {"cpu_model": "fixture", "faiss": "fixture", "numpy": "fixture"},
            "input_sha256": {
                "artifacts/queries/e5-validation-fp32/manifest.json": "queries",
                "artifacts/indexes/e5-flat-ip-v1/manifest.json": "flat",
                "artifacts/indexes/e5-hnsw-m32-efc200-v1/manifest.json": "graph",
            },
        }
        quality = root / "quality"
        for label, name in zip(LABELS, RUNS, strict=True):
            trials = [
                {"repeat": i, "batch_call_seconds": [2.0], "single_query_seconds": [0.1]}
                for i in range(4)
            ]
            benchmark["results"][label] = {"trials": trials, "summary": summarize(trials, 2)}
            directory = quality / name
            write_json(
                directory / "evaluation.json",
                {
                    "query_count": 2,
                    "per_query": {"q1": {}, "q2": {}},
                    "metrics": {"mrr@10": 0.5, "recall@1000": 0.9},
                    "inputs": {"queries": {"sha256": "query-hash"}},
                },
            )
            write_json(
                directory / "manifest.json",
                {
                    "evaluation_sha256": sha256_file(directory / "evaluation.json"),
                    "source": source,
                    "query_manifest_sha256": "queries",
                    "index_manifest_sha256": "flat" if label == "exact" else "graph",
                    "k": 1000,
                    "batch_size": 32,
                    "threads": 4,
                    "efSearch": 0 if label == "exact" else int(label.removeprefix("hnsw-ef")),
                },
            )
        path = root / "benchmark.json"
        write_json(path, benchmark)
        return path, quality

    def test_verified_summary_contains_no_query_text_or_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path, quality = self.fixture(Path(directory))
            result = evidence(path, quality)
            self.assertEqual(len(result["points"]), 4)
            self.assertEqual(result["points"][0]["search"]["median_batch_queries_per_second"], 1)
            self.assertNotIn("query_ids_in_vector_order", result)
            self.assertNotIn(str(Path(directory)), str(result))

    def test_inconsistent_saved_timing_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, quality = self.fixture(Path(directory))
            report = read_json(path)
            report["results"]["exact"]["summary"]["single_query_p95_ms"] = 999
            write_json(path, report)
            with self.assertRaises(AssertionError):
                evidence(path, quality)

    def test_quality_artifact_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, quality = self.fixture(Path(directory))
            manifest_path = quality / RUNS[1] / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["index_manifest_sha256"] = "another graph"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "different configurations"):
                evidence(path, quality)

    def test_missing_trial_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, quality = self.fixture(Path(directory))
            report = read_json(path)
            report["results"]["exact"]["trials"].pop()
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "trials"):
                evidence(path, quality)
