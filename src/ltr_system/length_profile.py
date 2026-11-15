"""Uniform corpus sampling and untruncated E5 token-length profiling on CPU."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.lexical_index import verify_corpus
from ltr_system.project import (
    configure_hub,
    encoder_config,
    initialize,
    parser,
    read_json,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def sample_records(source: Path, manifest: dict, count: int, seed: int) -> list[dict]:
    if not 0 < count <= manifest["records"]:
        raise ValueError("Sample size must be positive and no larger than corpus")
    targets = set(random.Random(seed).sample(range(manifest["records"]), count))
    selected = []
    offset = 0
    for entry in manifest["shards"]:
        local_count = 0
        with (source / "shards" / entry["name"]).open("rb") as stream:
            for position, line in enumerate(stream):
                if offset + position in targets:
                    record = json.loads(line)
                    if not isinstance(record["id"], str) or not isinstance(record["contents"], str):
                        raise ValueError("Invalid sampled passage")
                    selected.append(record)
                local_count += 1
        if local_count != entry["records"]:
            raise ValueError("Shard line count differs from manifest")
        offset += local_count
    if len(selected) != count or len({r["id"] for r in selected}) != count:
        raise ValueError("Missing sample records or duplicate sampled IDs")
    return selected


def summarize(lengths: list[int], limit: int) -> dict:
    if not lengths or limit <= 0 or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("Expected positive lengths and token limit")
    ordered = sorted(lengths)
    return {
        "count": len(lengths),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": sum(lengths) / len(lengths),
        "percentiles_nearest_rank": {
            str(p): ordered[math.ceil(p / 100 * len(ordered)) - 1] for p in (50, 90, 95, 99)
        },
        "above_limit": sum(n > limit for n in lengths),
        "fraction_above_limit": sum(n > limit for n in lengths) / len(lengths),
    }


def main() -> None:
    command = parser("Profile a uniform full-corpus sample with the pinned E5 tokenizer.")
    command.add_argument("--sample-size", type=int, default=10_000)
    command.add_argument("--seed", type=int, default=2026)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run corpus scanning/tokenization in a CPU Slurm allocation")
    destination = root / f"artifacts/profiles/e5-n{args.sample_size}-seed{args.seed}"
    if destination.exists():
        raise FileExistsError(f"Profile exists: {destination}")
    source = root / "data/processed/msmarco-passage-v1-jsonl"
    source_hash = sha256_file(source / "manifest.json")
    LOGGER.info("Verifying shards and selecting a uniform sample")
    manifest = verify_corpus(source)
    records = sample_records(source, manifest, args.sample_size, args.seed)
    config = encoder_config()
    checkpoint = read_json(root / "artifacts/models/e5-small-v2.json")
    if any(checkpoint[key] != config[key] for key in ("repo_id", "revision")):
        raise ValueError("Tokenizer checkpoint differs from pinned configuration")
    configure_hub(root, offline=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    lengths = []
    for start in range(0, len(records), 256):
        texts = ["passage: " + record["contents"] for record in records[start : start + 256]]
        tokens = tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(int(n) for n in tokens["length"])
    if len(lengths) != len(records):
        raise ValueError("Tokenizer length count mismatch")
    if sha256_file(source / "manifest.json") != source_hash:
        raise ValueError("Source manifest changed during profiling")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".profile-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        with (stage / "sample.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(
            stage / "token_lengths.json",
            {
                "passage_ids": [r["id"] for r in records],
                "lengths": lengths,
            },
        )
        report = {
            "schema_version": 1,
            "scope": "sampled passage token lengths; no model inference",
            "sampling": "uniform row sample without replacement; source order retained",
            "seed": args.seed,
            "python": sys.version.split()[0],
            "source_manifest_sha256": source_hash,
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "max_length": config["max_length"],
            "lengths_include": "passage prefix and tokenizer special tokens; no truncation",
            "statistics": summarize(lengths, config["max_length"]),
            "sample_sha256": sha256_file(stage / "sample.jsonl"),
            "token_lengths_sha256": sha256_file(stage / "token_lengths.json"),
        }
        write_json(stage / "manifest.json", report)
        stage.rename(destination)
    LOGGER.info("Length profile: %s", report["statistics"])
    LOGGER.info("Saved profile: %s", destination)


if __name__ == "__main__":
    main()
