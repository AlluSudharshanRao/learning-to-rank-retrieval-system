"""Hand-calculated relevance metrics and evaluation population checks."""

import unittest

from ltr_system.metrics import evaluate, query_metrics


class MetricTests(unittest.TestCase):
    def test_documented_example(self):
        result = query_metrics(["X", "A", "Y", "B", "Z"], {"A", "B", "C"}, 5)
        self.assertAlmostEqual(result["recall@5"], 2 / 3)
        self.assertEqual(result["rr@5"], 0.5)
        self.assertAlmostEqual(result["ndcg@5"], 0.49818925746641285)

    def test_perfect_and_short_rankings(self):
        result = query_metrics(["A", "B"], {"A", "B"}, 10)
        self.assertEqual(list(result.values()), [1.0, 1.0, 1.0])
        self.assertEqual(query_metrics(["A"], {"A", "B"}, 1)["recall@1"], 0.5)

    def test_empty_and_missed_positive(self):
        for ranking in ([], ["X", "Y"]):
            self.assertEqual(list(query_metrics(ranking, {"A"}, 10).values()), [0, 0, 0])

    def test_cutoff_boundary(self):
        self.assertEqual(query_metrics(["X", "A"], {"A"}, 1)["rr@1"], 0)
        self.assertEqual(query_metrics(["X", "A"], {"A"}, 2)["rr@2"], 0.5)

    def test_rejects_invalid_inputs(self):
        for ranking, relevant, k in (
            (["A", "A"], {"A"}, 1),
            (["A"], set(), 1),
            (["A"], ["A"], 1),
            ([1], {"A"}, 1),
            (["A"], {""}, 1),
            ("A", {"A"}, 1),
            (["A"], {"A"}, 0),
            (["A"], {"A"}, True),
            (["A"], {"A"}, 1.5),
        ):
            with self.subTest(ranking=ranking, relevant=relevant, k=k):
                with self.assertRaises(ValueError):
                    query_metrics(ranking, relevant, k)

    def test_macro_average_includes_missing_queries(self):
        result = evaluate(["q1", "q2"], {"q1": ["A"]}, {"q1": {"A"}, "q2": {"B"}})
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["missing_rankings"], 1)
        self.assertEqual(result["empty_rankings"], 1)
        self.assertTrue(all(value == 0.5 for value in result["metrics"].values()))
        self.assertEqual(result["per_query"]["q2"]["rr@10"], 0)

    def test_explicit_empty_is_not_missing(self):
        result = evaluate(["q"], {"q": []}, {"q": {"A"}})
        self.assertEqual(result["missing_rankings"], 0)
        self.assertEqual(result["empty_rankings"], 1)

    def test_rejects_population_errors(self):
        for ids, rankings, positives in (
            ([], {}, {}),
            (["q", "q"], {}, {"q": {"A"}}),
            (["q"], {"other": []}, {"q": {"A"}}),
            (["q"], {}, {}),
        ):
            with self.subTest(ids=ids, rankings=rankings):
                with self.assertRaises(ValueError):
                    evaluate(ids, rankings, positives)
