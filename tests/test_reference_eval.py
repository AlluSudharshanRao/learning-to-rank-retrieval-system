"""Reference adapter tests with fixed expected values; HPC verifies the real library."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ltr_system.evaluation import evaluate_files
from ltr_system.project import read_json, write_json
from ltr_system.reference_eval import crosscheck, independent_inputs


class ReferenceTests(unittest.TestCase):
    def fixture(self, root):
        queries, qrels, ranking, saved = [
            root / name for name in ("queries.tsv", "qrels.tsv", "ranking.tsv", "evaluation.json")
        ]
        queries.write_text("q1\tfirst\nq2\tsecond\nq3\tmissing\n", encoding="utf-8")
        qrels.write_text(
            "q1\t0\tpositive\t1\nq2\t0\tpositive\t1\nq3\t0\tpositive\t1\n", encoding="utf-8"
        )
        ranking.write_text(
            "q1\tnegative\t1\nq1\tpositive\t2\n"
            + "".join(f"q2\td{i}\t{i}\n" for i in range(1, 11))
            + "q2\tpositive\t11\n",
            encoding="utf-8",
        )
        evaluate_files(queries, qrels, ranking, saved, "fixture")
        return queries, qrels, ranking, saved

    def reference(self):
        calls = []

        class Evaluator:
            def __init__(self, judgments, measures):
                self.measures = measures
                if len(judgments) != 3:
                    raise AssertionError("Lost frozen query population")

            def evaluate(self, runs):
                calls.append((self.measures, runs))
                if self.measures == {"recip_rank"}:
                    return {"q1": {"recip_rank": 0.5}, "q2": {"recip_rank": 0.0}}
                return {
                    qid: {
                        "ndcg_cut_10": 1 / math.log2(3) if qid == "q1" else 0.0,
                        **{f"recall_{k}": 1.0 for k in (20, 50, 100, 1000)},
                    }
                    for qid in ("q1", "q2")
                }

        return SimpleNamespace(RelevanceEvaluator=Evaluator), calls

    def test_cutoffs_missing_population_and_preserved_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            reference, calls = self.reference()
            report = crosscheck(*paths, reference)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["queries"], 3)
            self.assertEqual(report["reference_metrics"]["mrr@10"], 0.5 / 3)
            self.assertEqual(list(calls[0][1]["q1"]), ["negative", "positive"])
            self.assertEqual(calls[0][1]["q2"]["positive"], -11.0)
            self.assertNotIn("positive", calls[1][1]["q2"])

    def test_per_query_mismatch_rejected_even_if_aggregate_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            saved = paths[-1]
            report = read_json(saved)
            report["per_query"]["q1"]["rr@10"] = 0.6
            write_json(saved, report)
            with self.assertRaisesRegex(ValueError, "Reference mismatch"):
                crosscheck(*paths, self.reference()[0])

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            paths[2].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input mismatch"):
                crosscheck(*paths, self.reference()[0])

    def test_duplicate_passage_rejected_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            paths[2].write_text("q1\tpositive\t1\nq1\tpositive\t2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate passage"):
                independent_inputs(*paths[:3])

    def test_aggregate_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            report = read_json(paths[-1])
            report["metrics"]["mrr@10"] = 1.0
            write_json(paths[-1], report)
            with self.assertRaisesRegex(ValueError, "Aggregate"):
                crosscheck(*paths, self.reference()[0])

    def test_query_population_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            report = read_json(paths[-1])
            report["per_query"].pop("q3")
            write_json(paths[-1], report)
            with self.assertRaisesRegex(ValueError, "population"):
                crosscheck(*paths, self.reference()[0])
