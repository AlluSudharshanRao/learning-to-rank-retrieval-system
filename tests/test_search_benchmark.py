"""Matched workload scheduling, timer boundaries, summary math and report integration."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import test_hnsw_search as search_fixtures

from ltr_system.project import read_json
from ltr_system.search_benchmark import measure, run, schedule, summarize


class BenchmarkTests(unittest.TestCase):
    def test_schedule_reproducibility_and_balanced_algorithm_positions(self):
        labels = ["exact", "a", "b", "c"]
        first = schedule(20, 5, 4, labels, 2026)
        self.assertEqual(first, schedule(20, 5, 4, labels, 2026))
        for trial in first:
            self.assertEqual(sorted(trial["batch_order"]), list(range(20)))
            self.assertEqual(set(trial["latency_order"]), set(first[0]["latency_order"]))
        for position in range(4):
            self.assertEqual({trial["algorithms"][position] for trial in first}, set(labels))

    def test_timer_batches_and_order(self):
        batches = []

        class Index:
            ntotal = 2

            def search(self, batch, k):
                batches.append(batch.copy())
                return np.zeros((len(batch), k), np.float32), np.tile(np.arange(k), (len(batch), 1))

        values = iter([10, 10.25, 20, 20.5])
        vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
        times = measure(Index(), vectors, [2, 0, 1], 2, 2, clock=lambda: next(values))
        self.assertEqual(times, [0.25, 0.5])
        np.testing.assert_array_equal(batches[0], vectors[[2, 0]])
        np.testing.assert_array_equal(batches[1], vectors[[1]])

    def test_invalid_results_fail(self):
        class Index:
            ntotal = 2

            def search(self, batch, k):
                return np.full((len(batch), k), np.nan), np.zeros((len(batch), k), np.int64)

        with self.assertRaisesRegex(ValueError, "results"):
            measure(Index(), np.ones((1, 2), np.float32), [0], 1, 1)

    def test_summary_uses_single_calls_not_amortized_batch_latency(self):
        result = summarize(
            [
                {"batch_call_seconds": [1, 1], "single_query_seconds": [0.001, 0.002]},
                {"batch_call_seconds": [2, 2], "single_query_seconds": [0.003, 0.004]},
            ],
            2000,
        )
        self.assertEqual(result["batch_queries_per_second_repetitions"], [1000, 500])
        self.assertEqual(result["median_batch_queries_per_second"], 750)
        self.assertAlmostEqual(result["single_query_p50_ms"], 2.5)
        self.assertAlmostEqual(result["single_query_p95_ms"], 3.85)
        self.assertEqual(result["single_query_observations"], 4)

    def test_invalid_schedule_rejected(self):
        for sample, repeats, labels in (
            (0, 1, ["a"]),
            (21, 1, ["a"]),
            (1, 0, ["a"]),
            (1, 1, ["a", "a"]),
        ):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                schedule(20, sample, repeats, labels, 2026)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class BenchmarkIntegrationTests(unittest.TestCase):
    def test_real_indexes_complete_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search_fixtures.HnswSearchTests().fixture(root)
            output = root / "benchmark.json"
            report = run(root, 1, 1, 2, 1, 2026, output)
            self.assertEqual(report["status"], "matched CPU search benchmark completed")
            self.assertEqual(
                set(report["results"]), {"exact", "hnsw-ef1024", "hnsw-ef2048", "hnsw-ef4096"}
            )
            self.assertEqual(read_json(output)["batch_queries"], 2)
            for result in report["results"].values():
                self.assertGreater(result["summary"]["median_batch_queries_per_second"], 0)
                self.assertEqual(result["summary"]["single_query_observations"], 1)
            with self.assertRaises(FileExistsError):
                run(root, 1, 1, 2, 1, 2026, output)

    def test_changed_vectors_do_not_publish_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search_fixtures.HnswSearchTests().fixture(root)
            np.save(
                root / "artifacts/queries/e5-validation-fp32/vectors.npy",
                np.zeros((2, 16), np.float32),
            )
            output = root / "benchmark.json"
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(root, 1, 1, 2, 1, 2026, output)
            self.assertFalse(output.exists())
