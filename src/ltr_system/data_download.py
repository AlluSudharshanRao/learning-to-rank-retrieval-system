"""Download and inventory the official passage-v1 source archive without extraction."""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from ltr_system.project import initialize, parser, read_json, write_json

SOURCE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
ARCHIVE_NAME = "collectionandqueries.tar.gz"
REQUIRED = {
    "collection.tsv",
    "queries.train.tsv",
    "queries.dev.small.tsv",
    "qrels.train.tsv",
    "qrels.dev.small.tsv",
}
LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> list[dict]:
    """Read the entire gzip stream; reject unsafe or duplicate archive members."""
    members = []
    names = set()
    with (
        gzip.open(path, "rb") as compressed,
        tarfile.open(fileobj=compressed, mode="r|") as archive,
    ):
        for member in archive:
            name = member.name
            normalized = PurePosixPath(name)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or "\\" in name
                or ":" in name
                or not member.isfile()
                or len(normalized.parts) != 1
            ):
                raise ValueError(f"Unsupported archive member: {name!r}")
            name = normalized.name
            if name in names:
                raise ValueError(f"Duplicate archive member: {name!r}")
            names.add(name)
            members.append({"name": name, "bytes": member.size})
        # Consume the trailer too, so gzip CRC/truncation failures are observed.
        while compressed.read(1024 * 1024):
            pass
    if missing := REQUIRED - names:
        raise ValueError(f"Missing required members: {sorted(missing)}")
    return members


def download(url: str, destination: Path) -> dict:
    """Publish only a complete, inventoried archive; retries restart from byte zero."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f"Expected HTTP 200, received {response.status}")
                expected = response.headers.get("Content-Length")
                received = 0
                next_log = 256 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    received += len(chunk)
                    if received >= next_log:
                        LOGGER.info("Downloaded %.0f MiB", received / 1024**2)
                        next_log += 256 * 1024 * 1024
                if expected is not None and received != int(expected):
                    raise ValueError(f"Incomplete download: {received} of {expected} bytes")
                final_url = response.geturl()
            output.flush()
            os.fsync(output.fileno())
        LOGGER.info("Checking archive structure and gzip integrity")
        members = inventory(temporary)
        result = {
            "schema_version": 1,
            "dataset": "msmarco-passage-v1",
            "source_url": url,
            "resolved_url": final_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "bytes": received,
            "sha256": sha256_file(temporary),
            "members": members,
            "status": "archive validated; records not parsed",
        }
        os.replace(temporary, destination)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    command = parser("Download and inventory the official MS MARCO Passage v1 archive.")
    args = command.parse_args()
    root = initialize(args)
    directory = root / "data/raw/msmarco-passage-v1"
    archive = directory / ARCHIVE_NAME
    manifest = directory / "source.json"
    if archive.exists():
        if not manifest.exists():
            raise ValueError(f"Archive has no source manifest; inspect before reuse: {archive}")
        recorded = read_json(manifest)
        if recorded.get("source_url") != SOURCE_URL:
            raise ValueError("Existing source URL differs from the configured source")
        if archive.stat().st_size != recorded["bytes"]:
            raise ValueError("Existing archive size differs from its manifest")
        if sha256_file(archive) != recorded["sha256"]:
            raise ValueError("Existing archive checksum differs from its manifest")
        LOGGER.info("Verified existing archive: %s", archive)
    else:
        if manifest.exists():
            raise ValueError("Source manifest exists without its archive; inspect before retrying")
        write_json(manifest, download(SOURCE_URL, archive))
        LOGGER.info("Saved source manifest: %s", manifest)


if __name__ == "__main__":
    main()
