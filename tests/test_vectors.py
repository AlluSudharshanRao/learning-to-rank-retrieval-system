"""Vector integrity and tie semantics, independent of model quality."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ltr_system.vectors import check_exact_results, validate_vectors


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.query = np.array([[1.0, 0.0]], dtype=np.float32)
        self.passages = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.ids = np.array([10, 20], dtype=np.int64)

    def test_rejects_nonfinite_or_unnormalized(self):
        for query in [np.array([[np.nan, 0.0]]), np.array([[2.0, 0.0]])]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_vectors(query, self.passages, self.ids)

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            validate_vectors(np.array([[1.0, 0.0, 0.0]]), self.passages, self.ids)

    def test_rejects_invalid_ids(self):
        for ids in [
            np.array([10, 10], dtype=np.int64),
            np.array([-1, 20], dtype=np.int64),
            np.array([10.0, 20.0]),
        ]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_vectors(self.query, self.passages, ids)

    def test_exact_ranking(self):
        check_exact_results(
            self.query, self.passages, self.ids, np.array([[1.0, 0.0]]), np.array([[10, 20]])
        )

    def test_rejects_scores_associated_with_wrong_ids(self):
        with self.assertRaises(AssertionError):
            check_exact_results(
                self.query, self.passages, self.ids, np.array([[1.0, 0.0]]), np.array([[20, 10]])
            )

    def test_rejects_unsorted_results(self):
        with self.assertRaises(AssertionError):
            check_exact_results(
                self.query, self.passages, self.ids, np.array([[0.0, 1.0]]), np.array([[20, 10]])
            )

    def test_tied_scores_allow_different_id_orders(self):
        passages = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        for ids in [np.array([[10, 20]]), np.array([[20, 10]])]:
            check_exact_results(self.query, passages, self.ids, np.array([[1.0, 1.0]]), ids)

    def test_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            check_exact_results(
                self.query, self.passages, self.ids, np.array([[1.0, 0.0]]), np.array([[10, 99]])
            )

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
    def test_faiss_ids_and_serialization(self):
        import faiss

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        index.add_with_ids(self.passages, self.ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.faiss"
            faiss.write_index(index, str(path))
            restored = faiss.read_index(str(path))
            scores, found = restored.search(self.query, 2)
        check_exact_results(self.query, self.passages, self.ids, scores, found)


if __name__ == "__main__":
    unittest.main()
