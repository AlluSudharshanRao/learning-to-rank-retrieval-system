"""Record integrity checks use tiny synthetic data, never benchmark passages."""

import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltr_system.data_download import ARCHIVE_NAME, inventory, sha256_file
from ltr_system.data_prepare import prepare, validate
from ltr_system.project import read_json, write_json


class PrepareTests(unittest.TestCase):
    def test_extraction_publication_and_failed_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            fixture = root / "fixture"
            fixture.mkdir()
            self.fixture(fixture)
            archive = raw / ARCHIVE_NAME
            with tarfile.open(archive, "w:gz") as bundle:
                for path in sorted(fixture.iterdir()):
                    bundle.add(path, arcname=path.name)
            write_json(
                raw / "source.json",
                {
                    "dataset": "msmarco-passage-v1",
                    "sha256": sha256_file(archive),
                    "bytes": archive.stat().st_size,
                    "members": inventory(archive),
                },
            )
            output = root / "processed" / "records"
            with self.assertRaisesRegex(ValueError, "corpus record count"):
                prepare(raw, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.iterdir()), [])
            with patch("ltr_system.data_prepare.EXPECTED_PASSAGES", 2):
                prepare(raw, output)
            self.assertEqual(
                read_json(output / "integrity.json")["record_counts"]["collection.tsv"], 2
            )
            self.assertEqual(
                (output / "collection.tsv").read_bytes(), (fixture / "collection.tsv").read_bytes()
            )

    def fixture(self, directory):
        root = Path(directory)
        files = {
            "collection.tsv": "0\tFirst passage\n1\tSecond passage\n",
            "queries.train.tsv": "10\tTraining query\n",
            "queries.dev.small.tsv": "20\tFinal query\n",
            "qrels.train.tsv": "10\t0\t0\t1\n",
            "qrels.dev.small.tsv": "20\t0\t1\t1\n",
        }
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        return root

    def test_valid_records_and_id_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            result = validate(root, root / "ids.sqlite")
            self.assertEqual(result["record_counts"]["collection.tsv"], 2)
            self.assertEqual(result["missing_judgment_references"], 0)
            self.assertEqual(result["train_final_query_id_overlap"], 0)

    def test_invalid_records(self):
        cases = [
            ("collection.tsv", "0\tFirst\n0\tDuplicate\n", "duplicate ID"),
            ("collection.tsv", "0\t\n", "nonempty"),
            ("collection.tsv", "0\ttext\textra\n", "nonempty"),
            ("collection.tsv", "-1\ttext\n", "identifier"),
            ("qrels.train.tsv", "10\t0\t99\t1\n", "missing"),
            ("qrels.train.tsv", "99\t0\t0\t1\n", "missing"),
            ("qrels.train.tsv", "10\t0\t0\t1\n10\t0\t0\t0\n", "conflicting judgment"),
            ("qrels.train.tsv", "10\t0\t0\t3\n", "binary judgment"),
        ]
        for name, content, message in cases:
            with self.subTest(name=name, content=content):
                with tempfile.TemporaryDirectory() as directory:
                    root = self.fixture(directory)
                    (root / name).write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        validate(root, root / "ids.sqlite")

    def test_identical_duplicates_are_counted_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            for name in ("qrels.train.tsv", "qrels.dev.small.tsv"):
                path = root / name
                path.write_bytes(path.read_bytes() * 3)
            original = (root / "qrels.train.tsv").read_bytes()
            result = validate(root, root / "ids.sqlite")
            self.assertEqual(result["duplicate_judgments"], 4)
            for name in ("qrels.train.tsv", "qrels.dev.small.tsv"):
                self.assertEqual(
                    result["judgment_counts"][name],
                    {
                        "source_rows": 3,
                        "unique_judgments": 1,
                        "identical_duplicate_rows": 2,
                        "conflicting_judgments": 0,
                    },
                )
                self.assertEqual(result["record_counts"][name], 3)
            self.assertEqual((root / "qrels.train.tsv").read_bytes(), original)

    def test_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "collection.tsv").write_bytes(b"0\t\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                validate(root, root / "ids.sqlite")

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                prepare(root / "missing", root)
