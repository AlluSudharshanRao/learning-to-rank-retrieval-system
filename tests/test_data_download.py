"""Source download failures must not publish partial archives."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltr_system.data_download import REQUIRED, download, inventory, sha256_file


def archive_bytes(names):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 3
            archive.addfile(member, io.BytesIO(b"abc"))
    return output.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, content, expected=None):
        super().__init__(content)
        self.headers = {"Content-Length": str(expected if expected is not None else len(content))}

    def geturl(self):
        return "https://example.test/data"


class DownloadTests(unittest.TestCase):
    def test_success_records_hash_and_members(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.tar.gz"
            with patch(
                "ltr_system.data_download.urlopen", return_value=Response(archive_bytes(REQUIRED))
            ):
                result = download("https://example.test/data", destination)
            self.assertEqual(result["sha256"], sha256_file(destination))
            self.assertEqual({m["name"] for m in result["members"]}, REQUIRED)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_incomplete_transfer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.tar.gz"
            destination.write_bytes(b"original")
            with patch("ltr_system.data_download.urlopen", return_value=Response(b"short", 100)):
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    download("https://example.test/data", destination)
            self.assertEqual(destination.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_invalid_archives(self):
        cases = [
            ["collection.tsv"],
            [*REQUIRED, "../escape"],
            [*REQUIRED, "/absolute"],
            [*REQUIRED, "collection.tsv"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar.gz"
            for names in cases:
                with self.subTest(names=names):
                    path.write_bytes(archive_bytes(names))
                    with self.assertRaises(ValueError):
                        inventory(path)

    def test_corrupt_gzip_does_not_publish(self):
        content = bytearray(archive_bytes(REQUIRED))
        content[-8] ^= 255
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.tar.gz"
            with patch("ltr_system.data_download.urlopen", return_value=Response(content)):
                with self.assertRaises((OSError, tarfile.TarError)):
                    download("https://example.test/data", destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])
