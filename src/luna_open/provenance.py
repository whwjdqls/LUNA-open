"""Content fingerprints for inputs that must stay fixed across a run."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_sources(root: Path, manifest: dict) -> None:
    """Check training inputs once at job startup, before loading cached features."""
    if manifest.get("schema_version") != 2 or not manifest.get("source_sha256"):
        raise ValueError(
            "Training/feature caching requires a version-2 content-fingerprinted manifest"
        )
    for relative, expected in manifest["source_sha256"].items():
        path = root / relative
        if file_sha256(path) != expected:
            raise ValueError(f"Dataset content changed after manifest preparation: {relative}")


def validate_body_asset(checkpoint: dict, current_sha256: str) -> None:
    if checkpoint.get("smpl_asset_sha256") != current_sha256:
        raise ValueError(
            "SMPL asset differs from checkpoint; query/teacher correspondence is not valid"
        )


def validate_identity_transfer(checkpoint: dict, config: dict, feature_metadata: dict) -> None:
    """Preserve sampled point IDs and identity inputs when starting the animator."""
    if checkpoint["config"].get("smpl_pose_blend_shapes", True) != config.get(
        "smpl_pose_blend_shapes", True
    ):
        raise ValueError("Identity checkpoint teacher pose-corrective convention differs")
    for key in ("seed", "num_queries", "model", "image_size"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Identity checkpoint configuration differs: {key}")
    for kind in ("body", "face"):
        if checkpoint["feature_metadata"].get(kind) != feature_metadata.get(kind):
            raise ValueError(f"Identity checkpoint feature metadata differs: {kind}")
