"""Small, explicit experiment commands with lazy optional imports."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import ModuleType

from ltr_system.encoding import masked_mean_pool
from ltr_system.project import (
    DEFAULTS_ROOT,
    configure_hub,
    encoder_config,
    initialize,
    parser,
    read_json,
    validate_revision,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def download_main() -> None:
    args = parser("Download the configured revision-pinned E5 checkpoint.").parse_args()
    root = initialize(args)
    config = encoder_config()
    configure_hub(root, offline=False)
    from huggingface_hub import snapshot_download

    manifest_path = root / "artifacts/models/e5-small-v2.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if (existing["repo_id"], existing["revision"]) != (config["repo_id"], config["revision"]):
            raise ValueError(
                "Manifest differs from versioned config; resolve explicitly before downloading."
            )
    snapshot = snapshot_download(
        repo_id=config["repo_id"],
        revision=config["revision"],
        cache_dir=str(root / "cache/huggingface/hub"),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md"],
        token=False,
    )
    if not list(Path(snapshot).glob("*.safetensors")):
        raise RuntimeError("No root Safetensors weights downloaded.")
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "snapshot_path": snapshot,
            "status": "downloaded; model execution checked separately",
        },
    )
    LOGGER.info("Downloaded %s at %s", config["repo_id"], config["revision"])
    print("Manifest:", manifest_path)


def _cuda() -> ModuleType:
    import torch

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this command inside a Slurm GPU allocation.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this PyTorch process.")
    return torch


def gpu_main() -> None:
    args = parser("Check CUDA forward and backward execution; not a benchmark.").parse_args()
    initialize(args)
    torch = _cuda()
    import socket
    import sys

    print("Node:", socket.gethostname())
    print("Job:", os.environ["SLURM_JOB_ID"])
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Build CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    n = 256
    a = torch.ones((n, n), device="cuda:0", requires_grad=True)
    b = torch.ones((n, n), device="cuda:0")
    output = a @ b
    output.mean().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, float(n)))
    torch.testing.assert_close(a.grad, torch.full_like(a, 1.0 / n))
    print("CUDA forward/backward check: PASS")


def encoder_main() -> None:
    args = parser("Inspect offline E5 embeddings on illustrative text.").parse_args()
    root = initialize(args)
    configure_hub(root, offline=True)
    config = encoder_config()
    manifest = read_json(root / "artifacts/models/e5-small-v2.json")
    validate_revision(manifest["revision"])
    if (manifest["repo_id"], manifest["revision"]) != (config["repo_id"], config["revision"]):
        raise ValueError("Model manifest does not match the configured checkpoint.")
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Model snapshot missing: {snapshot}; run download_encoder.py.")
    torch = _cuda()
    import numpy as np
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    fixture = read_json(DEFAULTS_ROOT / "smoke_inputs.json")
    ids = np.array([item["id"] for item in fixture["passages"]], dtype=np.int64)
    texts = ["query: " + fixture["query"]] + [
        "passage: " + item["text"] for item in fixture["passages"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        .to("cuda:0")
        .eval()
    )
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        pooled = masked_mean_pool(hidden, batch["attention_mask"])
        embeddings = functional.normalize(pooled, p=2, dim=1)
    torch.cuda.synchronize()
    if embeddings.shape != (len(texts), config["dimension"]):
        raise ValueError("Unexpected embedding shape.")
    from ltr_system.vectors import validate_vectors

    query, passages, ids = validate_vectors(
        embeddings[:1].cpu().numpy(), embeddings[1:].cpu().numpy(), ids
    )
    output_dir = root / "artifacts/smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "encoder_vectors.npz", query=query, passages=passages, passage_ids=ids)
    write_json(
        output_dir / "encoder_provenance.json",
        {
            "schema_version": 1,
            "revision": config["revision"],
            "repo_id": config["repo_id"],
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "job_id": os.environ["SLURM_JOB_ID"],
            "max_length": config["max_length"],
            "passage_ids": ids.tolist(),
            "texts": texts,
            "purpose": "illustrative correctness check, not retrieval evaluation",
        },
    )
    print("Token batch:", tuple(batch["input_ids"].shape))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Revision:", config["revision"])
    print("Token counts:", batch["attention_mask"].sum(dim=1).tolist())
    print("Hidden states:", tuple(hidden.shape))
    print("Embeddings:", tuple(embeddings.shape))
    print("L2 norms:", embeddings.norm(dim=1).tolist())
    for pid, score in zip(ids, (query @ passages.T)[0], strict=True):
        print(f"Passage {pid} cosine: {score:.6f}")
    print("Encoder check: PASS")
    print("Saved:", output_dir / "encoder_vectors.npz")


def faiss_main() -> None:
    args = parser("Check exact CPU search and saved index integrity.").parse_args()
    root = initialize(args)
    import faiss
    import numpy as np

    from ltr_system.vectors import check_exact_results, validate_vectors

    output_dir = root / "artifacts/smoke"
    faiss.omp_set_num_threads(1)
    with np.load(output_dir / "encoder_vectors.npz", allow_pickle=False) as data:
        query, passages, ids = validate_vectors(
            data["query"], data["passages"], data["passage_ids"]
        )
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(passages.shape[1]))
    index.add_with_ids(passages, ids)
    scores, found = index.search(query, len(ids))
    check_exact_results(query, passages, ids, scores, found)
    index_path = output_dir / "exact.faiss"
    faiss.write_index(index, str(index_path))
    restored = faiss.read_index(str(index_path))
    restored_scores, restored_ids = restored.search(query, len(ids))
    check_exact_results(query, passages, ids, restored_scores, restored_ids)
    np.testing.assert_allclose(restored_scores, scores, atol=1e-6, rtol=1e-5)
    print("Vectors:", index.ntotal, "Dimensions:", index.d)
    for rank, (pid, score) in enumerate(zip(found[0], scores[0], strict=True), 1):
        print(f"Rank {rank}: passage_id={pid}, cosine={score:.6f}")
    print("Direct dot-product agreement: PASS")
    print("Index save/reload agreement: PASS")
