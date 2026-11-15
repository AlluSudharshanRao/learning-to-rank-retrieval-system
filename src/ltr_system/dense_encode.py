"""Shard-level resumable FP32 E5 corpus encoding with verified artifact reuse."""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from ltr_system.commands import _cuda
from ltr_system.data_download import sha256_file
from ltr_system.encoding import masked_mean_pool
from ltr_system.project import (
    configure_hub,
    encoder_config,
    initialize,
    parser,
    read_json,
    write_json,
)

LOGGER = logging.getLogger(__name__)


@contextmanager
def writer_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    # A stale lock after forced termination requires inspection, never automatic deletion.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
            },
            stream,
        )
    try:
        yield
    finally:
        lock.unlink()


def check_vectors(vectors, shape: tuple[int, int]) -> None:
    import numpy as np

    if vectors.shape != shape or vectors.dtype != np.float32:
        raise ValueError("Unexpected vector shape or dtype")
    for start in range(0, shape[0], 4096):
        batch = vectors[start : start + 4096]
        if not np.isfinite(batch).all():
            raise ValueError("Nonfinite corpus vectors")
        if not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5, rtol=1e-5):
            raise ValueError("Corpus vectors are not unit normalized")


def verify_completed(destination: Path, identity: dict, rows: int, dimension: int) -> dict:
    import numpy as np

    report = read_json(destination / "manifest.json")
    if report.get("status") != "complete" or report["identity"] != identity:
        raise ValueError("Completed shard has incompatible encoding identity")
    for name in ("vectors.npy", "passage_ids.json"):
        expected = report["files"][name]
        path = destination / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Completed shard file changed: {name}")
    ids = read_json(destination / "passage_ids.json")["passage_ids"]
    if len(ids) != rows or len(set(ids)) != rows:
        raise ValueError("Completed shard ID count/uniqueness mismatch")
    vectors = np.load(destination / "vectors.npy", mmap_mode="r", allow_pickle=False)
    try:
        check_vectors(vectors, (rows, dimension))
    finally:
        vectors._mmap.close()
        del vectors
    return report


def encode_shard(source: Path, entry: dict, destination: Path, config: dict, encode) -> dict:
    """The callback receives prefixed text batches and returns FP32 unit vectors."""
    import numpy as np

    if source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]:
        raise ValueError("Source shard identity mismatch")
    identity = {**config, "source_shard_sha256": entry["sha256"], "source_shard": entry["name"]}
    count = entry["records"]
    if destination.exists():
        report = verify_completed(destination, identity, count, config["dimension"])
        LOGGER.info("Verified completed shard; skipping encoder: %s", destination.name)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".encoding-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        vectors = np.lib.format.open_memmap(
            stage / "vectors.npy", mode="w+", dtype=np.float32, shape=(count, config["dimension"])
        )
        ids = []
        texts = []
        offset = 0
        started = time.perf_counter()

        def flush(array):
            nonlocal offset
            encoded = encode(list(texts))
            check_vectors(encoded, (len(texts), config["dimension"]))
            if offset + len(texts) > count:
                raise ValueError("Source exceeds declared record count")
            array[offset : offset + len(texts)] = encoded
            offset += len(texts)
            texts.clear()

        try:
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    pid, text = record["id"], record["contents"]
                    if not isinstance(pid, str) or not pid.isascii() or not pid.isdecimal():
                        raise ValueError("Invalid passage ID")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Invalid passage text")
                    ids.append(pid)
                    texts.append("passage: " + text)
                    if len(texts) == config["batch_size"]:
                        flush(vectors)
                        if offset % 10_000 == 0:
                            LOGGER.info("Encoded %s: %s/%s", entry["name"], offset, count)
            if texts:
                flush(vectors)
            if offset != count or len(set(ids)) != count:
                raise ValueError("Source record count or ID uniqueness mismatch")
            vectors.flush()
            check_vectors(vectors, (count, config["dimension"]))
        finally:
            vectors._mmap.close()
            del vectors
        if sha256_file(source) != entry["sha256"]:
            raise ValueError("Source shard changed during encoding")
        write_json(stage / "passage_ids.json", {"passage_ids": ids})
        report = {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "records": count,
            "shard_processing_seconds": time.perf_counter() - started,
            "timing_scope": "parsing, tokenization, encoding, vector writes/checks, source rehash, IDs",
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "files": {
                name: {"bytes": (stage / name).stat().st_size, "sha256": sha256_file(stage / name)}
                for name in ("vectors.npy", "passage_ids.json")
            },
        }
        write_json(stage / "manifest.json", report)
        stage.rename(destination)
    LOGGER.info("Published encoded shard: %s", destination)
    return report


def main() -> None:
    command = parser("Encode corpus shards with verified shard-level restart support.")
    command.add_argument("--start-shard", type=int, default=0)
    command.add_argument(
        "--stop-shard",
        type=int,
        default=1,
        help="Exclusive end; default encodes only the first shard",
    )
    command.add_argument("--batch-size", type=int, default=64)
    args = command.parse_args()
    root = initialize(args)
    source = root / "data/processed/msmarco-passage-v1-jsonl"
    corpus_hash = sha256_file(source / "manifest.json")
    corpus = read_json(source / "manifest.json")
    if corpus.get("status") != "corpus converted; Lucene index not built":
        raise ValueError("Expected verified converted corpus manifest")
    if not 0 <= args.start_shard < args.stop_shard <= len(corpus["shards"]) or args.batch_size <= 0:
        raise ValueError("Invalid shard range or batch size")
    config = encoder_config()
    checkpoint = read_json(root / "artifacts/models/e5-small-v2.json")
    if any(checkpoint[key] != config[key] for key in ("repo_id", "revision")):
        raise ValueError("Checkpoint does not match pinned configuration")
    configure_hub(root, offline=True)
    torch = _cuda()
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    identity = {
        **config,
        "batch_size": args.batch_size,
        "precision": "float32",
        "pooling": "masked mean",
        "normalization": "L2",
        "prefix": "passage: ",
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_build": torch.version.cuda,
        "source_corpus_manifest_sha256": corpus_hash,
    }
    output = root / "artifacts/embeddings/e5-small-v2-fp32"
    with writer_lock(output):
        if (output / "configuration.json").exists():
            if read_json(output / "configuration.json") != identity:
                raise ValueError("Existing embedding configuration differs; do not mix runs")
        else:
            write_json(output / "configuration.json", identity)
        LOGGER.info("Loading encoder on %s", torch.cuda.get_device_name(0))
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint["snapshot_path"], local_files_only=True, trust_remote_code=False
        )
        model = (
            AutoModel.from_pretrained(
                checkpoint["snapshot_path"],
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .float()
            .to("cuda:0")
            .eval()
        )

        def encode(texts):
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                vectors = functional.normalize(
                    masked_mean_pool(hidden, tokens["attention_mask"]), p=2, dim=1
                )
            return vectors.cpu().numpy()

        for entry in corpus["shards"][args.start_shard : args.stop_shard]:
            name = entry["name"]
            if Path(name).name != name:
                raise ValueError("Invalid shard filename")
            encode_shard(
                source / "shards" / name, entry, output / Path(name).stem, identity, encode
            )
        if sha256_file(source / "manifest.json") != corpus_hash:
            raise ValueError("Corpus manifest changed")
    LOGGER.info("Requested shard range complete: [%s, %s)", args.start_shard, args.stop_shard)


if __name__ == "__main__":
    main()
