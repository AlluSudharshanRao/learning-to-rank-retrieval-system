"""Shard boundaries, byte identity, and safe failure on synthetic corpus inputs."""

import json
import tempfile
import unittest
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.lexical_corpus import convert
from ltr_system.project import write_json


class LexicalCorpusTests(unittest.TestCase):
    def fixture(self, root, text='0\tCafé "quoted" \\ text\n1\tSecond\n2\tThird\n', count=3):
        source = root / "source"
        source.mkdir()
        collection = source / "collection.tsv"
        collection.write_text(text, encoding="utf-8")
        write_json(
            source / "integrity.json",
            {
                "status": "records validated; splits not generated",
                "source_sha256": "synthetic",
                "record_counts": {"collection.tsv": count},
                "files": {
                    "collection.tsv": {
                        "sha256": sha256_file(collection),
                        "bytes": collection.stat().st_size,
                    }
                },
            },
        )
        return source

    def test_roundtrip_boundaries_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            first = convert(source, root / "one", 2)
            second = convert(source, root / "two", 2)
            self.assertEqual(first, second)
            self.assertEqual([s["records"] for s in first["shards"]], [2, 1])
            documents = []
            for shard in first["shards"]:
                path = root / "one/shards" / shard["name"]
                self.assertEqual(sha256_file(path), shard["sha256"])
                documents.extend(
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                )
            self.assertEqual([d["id"] for d in documents], ["0", "1", "2"])
            self.assertEqual(documents[0]["contents"], 'Café "quoted" \\ text')
            with self.assertRaises(FileExistsError):
                convert(source, root / "one", 2)

    def test_failed_conversion_never_publishes(self):
        for text, count in (("0\tfirst\nmalformed\n", 2), ("0\tfirst\n", 2)):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self.fixture(root, text, count)
                with self.assertRaises(ValueError):
                    convert(source, root / "result", 1)
                self.assertFalse((root / "result").exists())
                self.assertEqual(list(root.glob(".lexical-*")), [])

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            (source / "collection.tsv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                convert(source, root / "result")

    def test_exact_boundary_has_no_empty_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root, "0\tfirst\n1\tsecond\n", 2)
            result = convert(source, root / "result", 2)
            self.assertEqual(len(result["shards"]), 1)
            for size in (0, -1, True):
                with self.assertRaises(ValueError):
                    convert(source, root / "unused", size)
