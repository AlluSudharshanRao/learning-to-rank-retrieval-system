"""Validation membership, prefixes, and hash checks without model inference."""

import tempfile
import unittest
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.project import write_json
from ltr_system.query_encode import validation_inputs


class QueryInputTests(unittest.TestCase):
    def fixture(self, root, text="q2\tsecond query\nq1\tfirst query\n"):
        path = root / "validation.queries.tsv"
        path.write_text(text, encoding="utf-8")
        write_json(
            root / "manifest.json",
            {
                "outputs": {path.name: {"sha256": sha256_file(path)}},
                "counts": {"actual_validation_queries": 2},
            },
        )

    def test_order_and_query_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            ids, texts, identity = validation_inputs(root)
            self.assertEqual(ids, ["q2", "q1"])
            self.assertEqual(texts, ["query: second query", "query: first query"])
            self.assertEqual(identity["split"], "validation")
            # No judgment or final query file is present or needed.

    def test_modified_or_duplicate_queries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "validation.queries.tsv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                validation_inputs(root)
            self.fixture(root, "q\tfirst\nq\tduplicate\n")
            with self.assertRaisesRegex(ValueError, "uniqueness"):
                validation_inputs(root)
