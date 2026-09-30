import json

import pytest

from luna_open.provenance import (
    file_sha256,
    validate_body_asset,
    validate_identity_transfer,
    validate_inference_assets,
    verify_sources,
)


def test_detect_changed_pose_asset_and_dataset_input(tmp_path):
    path = tmp_path / "pose.txt"
    path.write_text("pose-v1")
    original = file_sha256(path)
    manifest = dict(schema_version=2, source_sha256={"pose.txt": original})
    verify_sources(tmp_path, manifest)
    validate_body_asset({"smpl_asset_sha256": original}, original)
    path.write_text("pose-v2")
    with pytest.raises(ValueError, match="content changed"):
        verify_sources(tmp_path, manifest)
    with pytest.raises(ValueError, match="SMPL asset differs"):
        validate_body_asset({"smpl_asset_sha256": original}, file_sha256(path))


def test_reject_unfingerprinted_training_manifest(tmp_path):
    with pytest.raises(ValueError, match="version-2"):
        verify_sources(tmp_path, {"schema_version": 1})


def test_identity_transfer_preserves_point_correspondence_and_inputs():
    config = dict(seed=2026, num_queries=8192, model={"width": 1024}, image_size=512)
    metadata = dict(body={"revision": "body-v1"}, face={"revision": "face-v1"})
    checkpoint = dict(config=config, feature_metadata=metadata)
    validate_identity_transfer(checkpoint, config, {**metadata, "motion": {}})
    # Same body asset and network shapes do not imply the same sampled point IDs.
    with pytest.raises(ValueError, match="seed"):
        validate_identity_transfer(checkpoint, {**config, "seed": 42}, metadata)
    with pytest.raises(ValueError, match="face"):
        validate_identity_transfer(checkpoint, config, {**metadata, "face": {"revision": "v2"}})
    with pytest.raises(ValueError, match="pose-corrective"):
        validate_identity_transfer(
            checkpoint, {**config, "smpl_pose_blend_shapes": False}, metadata
        )


def test_inference_rejects_encoder_revision_changes(tmp_path):
    metadata = {}
    for kind, key in (("body", "sapiens_body"), ("face", "dino_face"), ("motion", "dino_motion")):
        asset = dict(repo=f"test/{key}", revision="expected", files=["weights.bin"])
        metadata[kind] = dict(asset=asset)
        (tmp_path / key).mkdir()
        (tmp_path / key / "weights.bin").write_bytes(b"test fixture")
        (tmp_path / f"{key}-receipt.json").write_text(
            json.dumps(dict(status="downloaded", **asset))
        )
    validate_inference_assets(metadata, tmp_path)
    path = tmp_path / "dino_motion-receipt.json"
    changed = json.loads(path.read_text())
    changed["revision"] = "another-model-revision"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="motion"):
        validate_inference_assets(metadata, tmp_path)
