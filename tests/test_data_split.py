"""Query groups, deterministic selection, and final-judgment isolation."""

import tempfile
import unittest
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.data_split import INPUTS, assign, normalize, split
from ltr_system.project import read_json, write_json


class SplitTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize("  ＧＰＵ\tMemory  "), "gpu memory")

    def test_groups_exclusion_and_order_independence(self):
        queries = {str(i): f"query {i}" for i in range(20)}
        queries.update({"a": " QUERY  1 ", "held": "FINAL QUERY", "unjudged": "unused"})
        judged = set(queries) - {"unjudged"}
        result, counts = assign(queries, judged, {"f": "final query"}, 2, 8)
        reversed_result, _ = assign(
            dict(reversed(list(queries.items()))), judged, {"f": "final query"}, 2, 8
        )
        self.assertEqual(result, reversed_result)
        for ids in result.values():
            self.assertEqual("1" in ids, "a" in ids)
            self.assertNotIn("held", ids)
            self.assertNotIn("unjudged", ids)
        self.assertEqual(counts["excluded_judged_queries_matching_final_id_or_text"], 1)
        larger, _ = assign(queries, judged, {"f": "final query"}, 2, 12)
        self.assertEqual(result["validation"], larger["validation"])
        self.assertEqual(result["train"], larger["train"][: len(result["train"])])

    def test_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            assign({"1": "one", "2": "two"}, {"1", "2"}, {}, 1, 10)

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        (source / "queries.train.tsv").write_text(
            "".join(f"{i}\tquery {i}\n" for i in range(10)), encoding="utf-8"
        )
        (source / "queries.dev.small.tsv").write_text("final\tfinal query\n", encoding="utf-8")
        (source / "qrels.train.tsv").write_text(
            "".join(f"{i}\t0\t100\t1\n" * 2 for i in range(10)), encoding="utf-8"
        )
        write_json(
            source / "integrity.json",
            {
                "status": "records validated; splits not generated",
                "source_sha256": "fixture",
                "files": {name: {"sha256": sha256_file(source / name)} for name in INPUTS},
            },
        )
        return source

    def test_published_files_deduplicate_and_do_not_read_final_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            output = root / "output"
            # There is deliberately no final judgment file in this fixture.
            split(source, output, 2, 4)
            report = read_json(output / "manifest.json")
            self.assertFalse(report["final_judgments_read"])
            self.assertEqual(report["unique_judgment_counts"], {"train": 4, "validation": 2})
            for name, identity in report["outputs"].items():
                self.assertEqual(identity["sha256"], sha256_file(output / name))
            with self.assertRaises(FileExistsError):
                split(source, output, 2, 4)
            other = root / "other"
            split(source, other, 2, 4)
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), (other / path.name).read_bytes())

    def test_modified_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            (source / "qrels.train.tsv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                split(source, root / "output", 2, 4)
            self.assertFalse((root / "output").exists())
