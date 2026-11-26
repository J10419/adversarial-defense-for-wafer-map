"""Verify published model bytes and tensor hashes using the export manifest.

Run ``python -m project.verify_models`` from the repository root. No dataset or
GPU is needed. The manifest and checkpoints must come from a trusted source:
matching a hash establishes integrity relative to that manifest, not trust in
an arbitrary downloaded manifest. Original checkpoints are not read by this
command; their export-time tensor comparison remains historical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickletools
import re
import sys
import zipfile

import torch


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_METADATA = re.compile(
    r"(?i)(?:[A-Za-z]:[\\/]+(?:Users|Program Files|Windows)|"
    r"file:/{2,}|/home/[^/]+|/Users/[^/]+|/mnt/[a-z]/)"
)


def tensor_state_hash(state):
    """Match the manifest's length-prefixed metadata/raw tensor hash format."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state must map string names to tensors")
        tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def inspect_pickle_strings(container):
    """Inspect pickle opcodes as data; this step does not unpickle objects."""
    with zipfile.ZipFile(io.BytesIO(container)) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one checkpoint data.pkl member")
        for _, value, _ in pickletools.genops(archive.read(candidates[0])):
            if isinstance(value, str) and PROHIBITED_METADATA.search(value):
                raise ValueError("Checkpoint contains prohibited local or unrelated metadata")


def verify_models(models_dir):
    models_dir = Path(models_dir).resolve()
    manifest_path = models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest schema")
    records = manifest["models"]
    expected = {(method, seed) for method in ("clean", "noise", "pgd") for seed in (77, 78, 79)}
    if len(records) != 9 or {(row["method"], row["seed"]) for row in records} != expected:
        raise ValueError("Manifest must contain the nine distinct method/seed checkpoints")
    verified = []
    for record in records:
        model_id = f'{record["method"]}_seed{record["seed"]}'
        relative = PurePosixPath(record["path"])
        if relative.parts != ("models", model_id + ".pt") or record["source_checkpoint_id"] != model_id:
            raise ValueError("Unexpected checkpoint identifier or path")
        checkpoint_path = (models_dir / relative.name).resolve()
        if checkpoint_path.parent != models_dir:
            raise ValueError("Checkpoint path escapes model directory")
        container = checkpoint_path.read_bytes()
        container_hash = hashlib.sha256(container).hexdigest()
        if container_hash != record["new_container_sha256"]:
            raise ValueError(f"Container SHA-256 mismatch: {model_id}")
        inspect_pickle_strings(container)
        # These are trusted repository checkpoints; hash verification precedes
        # loading, and the exact verified bytes are passed to the CPU loader.
        checkpoint = torch.load(io.BytesIO(container), map_location="cpu", weights_only=False)
        if checkpoint["config"].get("data_dir") != "data/processed":
            raise ValueError(f"Nonportable data directory: {model_id}")
        if checkpoint["config"].get("training_mode") != record["method"] or checkpoint["config"].get("seed") != record["seed"]:
            raise ValueError(f"Checkpoint method or seed mismatch: {model_id}")
        state = checkpoint["model_state_dict"]
        if len(state) != record["tensor_count"]:
            raise ValueError(f"Tensor count mismatch: {model_id}")
        state_hash = tensor_state_hash(state)
        if state_hash != record["tensor_state_sha256"]:
            raise ValueError(f"Tensor state SHA-256 mismatch: {model_id}")
        verified.append({
            "model_id": model_id, "container_sha256": container_hash,
            "tensor_state_sha256": state_hash, "tensor_count": len(state),
            "historical_export_check": {
                "tensor_equality_verified": record.get("tensor_equality_verified") is True,
                "source_checkpoint_unchanged": record.get("source_checkpoint_unchanged") is True,
                "rechecked_against_original_files": False,
            },
        })
    return {
        "status": "PASS", "models_verified": len(verified),
        "verification_scope": "Published checkpoint container hashes, tensor hashes, portable metadata and model identities",
        "original_checkpoint_files_read": False,
        "historical_evidence_note": "Export-time equality flags are reported from the manifest; this command does not repeat comparison against the original checkpoints.",
        "models": verified,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    try:
        result = verify_models(args.models_dir)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
