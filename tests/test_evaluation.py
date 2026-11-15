"""Evaluation file formats, source preservation, and CLI integration."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ltr_system.evaluation import evaluate_files, load_positives, load_rankings
from ltr_system.project import read_json


class EvaluationTests(unittest.TestCase):
    def test_file_report_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels, run = (root / name for name in ("queries.tsv", "qrels.tsv", "run.tsv"))
            queries.write_text("q1\tFirst query\nq2\tMissing results\n", encoding="utf-8")
            qrels.write_text("q1\t0\tA\t1\nq1\t0\tA\t1\nq2\t0\tB\t1\n", encoding="utf-8")
            run.write_text("q1\tA\t2\nq1\tX\t1\n", encoding="utf-8")
            output = root / "report.json"
            result = evaluate_files(queries, qrels, run, output, "fixture", (5,), 5)
            self.assertEqual(result["metrics"]["mrr@5"], 0.25)
            self.assertEqual(result["metrics"]["recall@5"], 0.5)
            self.assertEqual(result["identical_duplicate_qrel_rows"], 1)
            self.assertEqual(result["missing_rankings"], 1)
            self.assertEqual(read_json(output), result)
            with self.assertRaises(FileExistsError):
                evaluate_files(queries, qrels, run, output, "fixture")
            script = Path(__file__).resolve().parents[1] / "scripts/evaluate_retrieval.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(script),
                    "--project-root",
                    str(root),
                    "--queries",
                    "queries.tsv",
                    "--qrels",
                    "qrels.tsv",
                    "--run",
                    "run.tsv",
                    "--output",
                    "cli.json",
                    "--run-id",
                    "fixture",
                    "--recall-cutoffs",
                    "5",
                    "--rank-cutoff",
                    "5",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(read_json(root / "cli.json"), result)

    def test_invalid_rankings(self):
        for text in ("q\tA\t0\n", "q\tA\t2\n", "q\tA\t1\nq\tB\t1\n", "q\tA\tnan\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.tsv"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_rankings(path)

    def test_conflicting_and_graded_qrels_rejected(self):
        for text in ("q\t0\tA\t1\nq\t0\tA\t0\n", "q\t0\tA\t3\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qrels.tsv"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_positives(path)

    def test_failed_evaluation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels, run = (root / name for name in ("queries.tsv", "qrels.tsv", "run.tsv"))
            queries.write_text("q\tQuery\n", encoding="utf-8")
            qrels.write_text("q\t0\tA\t1\n", encoding="utf-8")
            run.write_text("q\tA\t1\nq\tA\t2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate IDs"):
                evaluate_files(queries, qrels, run, root / "report.json", "fixture")
            self.assertFalse((root / "report.json").exists())
