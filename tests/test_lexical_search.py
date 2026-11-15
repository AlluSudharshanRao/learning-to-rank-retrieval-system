"""Retrieval serialization and backend-result validation without a JVM."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ltr_system.evaluation import evaluate_files
from ltr_system.lexical_search import retrieve


class Searcher:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k):
        return self.hits.get(query, [])


class SearchTests(unittest.TestCase):
    def test_serialization_scoring_and_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.tsv"
            queries.write_text("q1\tfirst\nq2\tsecond\n", encoding="utf-8")
            searcher = Searcher({"first": [SimpleNamespace(docid="A", score=1.0)]})
            counts = retrieve(searcher, queries, root / "run.tsv", 1000)
            self.assertEqual(counts, {"queries": 2, "returned_passages": 1, "empty_queries": 1})
            qrels = root / "qrels.tsv"
            qrels.write_text("q1\t0\tA\t1\nq2\t0\tB\t1\n", encoding="utf-8")
            result = evaluate_files(queries, qrels, root / "run.tsv", root / "eval.json", "fixture")
            self.assertEqual(result["metrics"]["mrr@10"], 0.5)

    def test_rejects_bad_backend_results(self):
        cases = [
            [("A", 1.0), ("A", 0.5)],
            [("A", 1.0), ("B", 2.0)],
            [("A", float("nan"))],
            [("bad\tid", 1.0)],
        ]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                queries = root / "queries.tsv"
                queries.write_text("q\ttext\n", encoding="utf-8")
                searcher = Searcher(
                    {"text": [SimpleNamespace(docid=p, score=s) for p, s in entries]}
                )
                with self.assertRaises(ValueError):
                    retrieve(searcher, queries, root / "run.tsv", 1000)

    def test_preserves_backend_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.tsv"
            queries.write_text("q\ttext\n", encoding="utf-8")
            searcher = Searcher({"text": [SimpleNamespace(docid=p, score=1.0) for p in ("B", "A")]})
            retrieve(searcher, queries, root / "run.tsv", 2)
            self.assertEqual((root / "run.tsv").read_text(), "q\tB\t1\nq\tA\t2\n")
