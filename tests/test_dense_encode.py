"""Shard publication and restart integrity using a synthetic encoder callback."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ltr_system.data_download import sha256_file
from ltr_system.dense_encode import encode_shard, verify_completed, writer_lock
from ltr_system.project import read_json


class DenseEncodeTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "part-00000.jsonl"
        source.write_text(
            "".join(json.dumps({"id": str(i), "contents": f"text {i}"}) + "\n" for i in range(3)),
            encoding="utf-8",
        )
        entry = {
            "name": source.name,
            "records": 3,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        config = {"dimension": 2, "batch_size": 2, "revision": "fixture"}
        return source, entry, config

    def test_publish_mapping_and_skip_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, entry, config = self.fixture(root)
            encoder = Mock(
                side_effect=lambda texts: np.tile(np.array([[1, 0]], np.float32), (len(texts), 1))
            )
            destination = root / "output"
            report = encode_shard(source, entry, destination, config, encoder)
            self.assertEqual(
                encoder.call_args_list[0].args[0], ["passage: text 0", "passage: text 1"]
            )
            self.assertEqual(
                read_json(destination / "passage_ids.json")["passage_ids"], ["0", "1", "2"]
            )
            encoder.reset_mock()
            self.assertEqual(encode_shard(source, entry, destination, config, encoder), report)
            encoder.assert_not_called()

    def test_interrupted_shard_is_not_published_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, entry, config = self.fixture(root)
            encoder = Mock(
                side_effect=[np.array([[1, 0], [1, 0]], np.float32), RuntimeError("interrupted")]
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                encode_shard(source, entry, root / "output", config, encoder)
            self.assertFalse((root / "output").exists())
            self.assertEqual(list(root.glob(".encoding-*")), [])
            encode_shard(
                source,
                entry,
                root / "output",
                config,
                lambda texts: np.tile(np.array([[1, 0]], np.float32), (len(texts), 1)),
            )

    def test_tampered_completed_shard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, entry, config = self.fixture(root)
            destination = root / "output"
            report = encode_shard(
                source,
                entry,
                destination,
                config,
                lambda texts: np.tile(np.array([[1, 0]], np.float32), (len(texts), 1)),
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                verify_completed(destination, {**report["identity"], "revision": "different"}, 3, 2)
            (destination / "passage_ids.json").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                encode_shard(source, entry, destination, config, Mock())

    def test_lock_prevents_second_writer_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with writer_lock(root):
                with self.assertRaises(FileExistsError):
                    with writer_lock(root):
                        pass
            self.assertFalse((root / ".writer.lock").exists())
