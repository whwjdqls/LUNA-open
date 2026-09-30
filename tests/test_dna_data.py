"""Format/geometry tests with generated data, never licensed release samples."""

import json
import pickle

import cv2
import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from luna_open.data import dataset_from_manifest
from luna_open.data.dna import DNADataset, observation_name, prepare
from luna_open.data.dna_audit import inspect_sequence
from luna_open.data.dna_smc import SMCFiles, SMCSequence, numeric_key
from luna_open.provenance import file_sha256


@pytest.fixture
def dna_fixture(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    K = np.array([[35.0, 0, 18.25], [0, 37.0, 14.75], [0, 0, 1.0]])
    D = np.array([0.025, -0.003, 0.001, -0.002, 0.0])
    rgb_path, annotation_path = root / "rgb.smc", root / "annotations.smc"
    cameras = [0, 1, 2, 3, 4, 48]
    with h5py.File(rgb_path, "w") as rgb, h5py.File(annotation_path, "w") as annot:
        for camera in cameras:
            colors = rgb.require_group(
                f"{'Camera_5mp' if camera < 48 else 'Camera_12mp'}/{camera}/color"
            )
            masks = annot.require_group(f"Mask/{camera}/mask")
            calibration = annot.require_group(f"Camera_Parameter/{camera:02d}")
            c2w = np.eye(4)
            c2w[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.01 * camera, 0.0]))[0]
            c2w[:3, 3] = [100.0, 50.0, 0.0]  # stored millimeters, deliberately not meters
            for key, value in dict(K=K, D=D, RT=c2w, Color_Calibration=np.eye(3)).items():
                calibration[key] = value
            for frame in [0, 1]:
                bgr = np.zeros((32, 40, 3), np.uint8)
                bgr[:] = [255, 0, 0]  # blue background must not bleed into red foreground
                bgr[5:27, 7:31] = [0, 0, 255]
                mask = np.zeros((32, 40), np.uint8)
                mask[5:27, 7:31] = 255
                mask[5, 7:31] = 128
                colors[str(frame)] = cv2.imencode(".png", bgr)[1].reshape(-1)
                masks[str(frame)] = cv2.imencode(".png", mask)[1].reshape(-1)
        fits = annot.create_group("SMPLx")
        fits["betas"] = np.zeros((2, 10))
        fits["expression"] = np.zeros((2, 10))
        fits["fullpose"] = np.zeros((2, 55, 3))
        fits["transl"] = np.zeros((2, 3))
        fits["scale"] = np.ones(1)
    pool = [[c, 0] for c in cameras[:5]]
    plan = dict(
        protocol="synthetic format test, explicit camera/frame split",
        world_unit_to_meter=0.001,
        world_unit_evidence="generated millimeter camera translations",
        scenes={
            "synthetic": dict(
                actor_id="synthetic_actor",
                rgb="rgb.smc",
                annotations="annotations.smc",
                reference_pool=pool,
                references=pool[:4],
                splits=dict(train=[[0, 1], [1, 1]], val=[[2, 1]], test=[[48, 1]]),
            )
        },
    )
    plan_path, manifest = root / "plan.json", root / "manifest.json"
    plan_path.write_text(json.dumps(plan))
    prepare(root, plan_path, manifest)
    return root, plan_path, manifest, K, D


def test_decode_camera_groups_polarity_native_fit_and_closure(dna_fixture):
    root, _, _, _, _ = dna_fixture
    with SMCFiles(maximum=2) as files:
        sequence = SMCSequence(root / "rgb.smc", root / "annotations.smc", files)
        for camera in [0, 48]:
            np.testing.assert_array_equal(sequence.image(camera, 0)[10, 10], [255, 0, 0])
            assert sequence.mask(camera, 0)[10, 10] == 1
            assert sequence.mask(camera, 0)[0, 0] == 0
            np.testing.assert_allclose(sequence.calibration(camera)["RT"][:3, 3], [100, 50, 0])
        assert sequence.smplx(1)["fullpose"].shape == (55, 3)
        with pytest.raises(ValueError, match="outside"):
            sequence.smplx(2)
        handle = files.open(root / "rgb.smc")
    assert not handle.id.valid


def test_pixel_projection_units_and_soft_alpha(dna_fixture):
    root, _, manifest, K, D = dna_fixture
    data = DNADataset(root, manifest, split="test", size=64)
    item = data[0]
    assert item["reference_images"].shape == (4, 3, 64, 64)
    assert item["rgb"].dtype == item["mask"].dtype == torch.float32
    assert (item["mask"] > 0).any() and (item["mask"] < 1).any()
    assert ((item["mask"] > 0) & (item["mask"] < 1)).any()
    # Red foreground on white: all red pixels stay 1 despite blue raw background.
    torch.testing.assert_close(item["rgb"][0], torch.ones((64, 64)), atol=1e-6, rtol=0)
    torch.testing.assert_close(item["rgb"][1], 1 - item["mask"][0], atol=1e-6, rtol=0)
    world_point = np.array([0.15, 0.025, 2.0, 1.0])
    camera_point = item["world_to_camera"].numpy() @ world_point
    expected_rotation = cv2.Rodrigues(np.array([0.0, 0.48, 0.0]))[0].T
    np.testing.assert_allclose(
        camera_point[:3], expected_rotation @ (world_point[:3] - [0.1, 0.05, 0]), atol=1e-7
    )
    # Independently distort a point, then remove distortion with OpenCV.
    distorted, _ = cv2.projectPoints(camera_point[None, :3], np.zeros(3), np.zeros(3), K, D)
    original = cv2.undistortPoints(distorted, K, D, P=K)[0, 0]
    row = data.lookup[item["scene"]][item["frame"]]
    left, top, right, _ = row["crop_xyxy"]
    expected = (original + 0.5 - [left, top]) * (64 / (right - left))
    projected = item["K"].numpy() @ camera_point[:3]
    np.testing.assert_allclose(projected[:2] / projected[2], expected, atol=1e-5)
    assert "pose" not in item  # Raw SMPL-X never silently enters the SMPL teacher.
    data.close()


def test_reference_selection_and_dataset_factory(dna_fixture):
    root, _, manifest, _, _ = dna_fixture
    data = dataset_from_manifest(root, manifest, size=32, random_references=True)
    assert isinstance(data, DNADataset)
    for i in range(4):
        item = data[i % len(data)]
        assert len(set(item["reference_names"])) == 4
        assert item["frame"] not in item["reference_names"]
        assert set(item["reference_names"]) <= set(data.metadata["synthetic"]["reference_pool"])
    data.close()


def test_worker_spawn_and_pickle_do_not_share_hdf5_handles(dna_fixture):
    root, _, manifest, _, _ = dna_fixture
    data = DNADataset(root, manifest, size=32)
    expected = data[0]["rgb"]
    clone = pickle.loads(pickle.dumps(data))
    assert not clone.files._files
    torch.testing.assert_close(clone[0]["rgb"], expected)
    loader = DataLoader(data, batch_size=1, num_workers=2, multiprocessing_context="spawn")
    batches = list(loader)
    assert len(batches) == len(data)
    torch.testing.assert_close(batches[0]["rgb"][0], expected)
    clone.close()
    data.close()


@pytest.mark.parametrize("failure", ["leak", "overlap", "actor", "path"])
def test_invalid_manifest_plans_fail(dna_fixture, failure):
    root, plan_path, _, _, _ = dna_fixture
    plan = json.loads(plan_path.read_text())
    scene = plan["scenes"]["synthetic"]
    if failure == "leak":
        scene["reference_pool"].append(scene["splits"]["test"][0])
    elif failure == "overlap":
        scene["splits"]["val"] = scene["splits"]["train"]
    elif failure == "actor":
        plan["identity_disjoint"] = True
    else:
        scene["rgb"] = "../escape.smc"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        prepare(root, plan_path, root / "invalid.json")


def test_source_mutation_detected(dna_fixture):
    root, _, manifest, _, _ = dna_fixture
    with h5py.File(root / "annotations.smc", "r+") as handle:
        handle["SMPLx/betas"][0, 0] = 0.1
    with pytest.raises(ValueError, match="content changed"):
        DNADataset(root, manifest)


@pytest.mark.parametrize("field", ["name", "K", "extrinsics", "identity", "units"])
def test_changed_manifest_cannot_silently_change_geometry_or_splits(dna_fixture, field):
    root, _, manifest, _, _ = dna_fixture
    document = json.loads(manifest.read_text())
    row = document["scenes"]["synthetic"]["frames"][0]
    if field == "name":
        row["frame_id"] += 1
    elif field == "K":
        row["K_opencv"][0][2] += 2
    elif field == "extrinsics":
        row["world_to_camera"][0][3] += 1
    elif field == "identity":
        document["identity_disjoint"] = True
    else:
        document["world_unit_to_meter"] = 0
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        data = DNADataset(root, manifest)
        try:
            data.load_frame("synthetic", row["name"])
        finally:
            data.close()


def test_ambiguous_keys_and_missing_fits_fail(dna_fixture):
    root, _, manifest, _, _ = dna_fixture
    with pytest.raises(ValueError, match="validated SMPL fits"):
        DNADataset(root, manifest, require_smpl=True)
    with h5py.File(root / "rgb.smc", "r+") as handle:
        group = handle["Camera_5mp"]
        group.create_group("00")
        with pytest.raises(ValueError, match="ambiguous"):
            numeric_key(group, 0)


def test_smpl_sidecar_requires_provenance_and_matches_teacher_frame(dna_fixture):
    root, plan_path, _, _, _ = dna_fixture
    transforms = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    transforms[:, :3, 3] = [0, 0, 3]
    parameters = root / "smpl.npz"
    np.savez(
        parameters,
        frame_ids=np.arange(2),
        pose=np.zeros((2, 72)),
        betas=np.zeros((1, 10)),
        body_to_world=transforms,
    )
    receipt_path = root / "smpl-receipt.json"
    receipt = dict(
        model="smpl",
        units="meters",
        pose_blend_shapes=True,
        smpl_asset_sha256="synthetic-asset",
        source_annotations_sha256=file_sha256(root / "annotations.smc"),
        parameters_sha256=file_sha256(parameters),
        validation=dict(passed=True, kind="synthetic_fixture"),
    )
    receipt_path.write_text(json.dumps(receipt))
    plan = json.loads(plan_path.read_text())
    plan["scenes"]["synthetic"]["smpl"] = dict(
        parameters=parameters.name, receipt=receipt_path.name
    )
    plan_path.write_text(json.dumps(plan))
    manifest = root / "smpl-manifest.json"
    prepare(root, plan_path, manifest)
    data = DNADataset(root, manifest, require_smpl=True, size=32)
    data.validate_smpl_asset("synthetic-asset", True)
    item = data[0]
    np.testing.assert_allclose(
        item["body_to_camera"], item["world_to_camera"].numpy() @ transforms[1], atol=1e-7
    )
    assert item["pose"].shape == (72,) and item["betas"].shape == (10,)
    with pytest.raises(ValueError, match="different SMPL asset"):
        data.validate_smpl_asset("wrong-asset", True)
    data.close()
    receipt["validation"]["passed"] = False
    receipt_path.write_text(json.dumps(receipt))
    manifest = root / "bad-smpl-manifest.json"
    prepare(root, plan_path, manifest)
    with pytest.raises(ValueError, match="successful validation"):
        DNADataset(root, manifest, require_smpl=True)


def test_observation_ids_cannot_collide():
    assert observation_name([0, 48]) != observation_name([48, 0])
    for invalid in [[0.0, 1], [60, 0], [0, -1], [False, 0]]:
        with pytest.raises(ValueError):
            observation_name(invalid)


def test_audit_full_camera_coverage_and_missing_mask_frame(dna_fixture):
    root, _, _, _, _ = dna_fixture
    with (
        h5py.File(root / "rgb.smc", "r+") as rgb,
        h5py.File(root / "annotations.smc", "r+") as annot,
    ):
        for camera in range(60):
            group = rgb["Camera_5mp" if camera < 48 else "Camera_12mp"]
            if str(camera) not in group:
                group[str(camera)] = rgb["Camera_5mp/0"]
                annot["Mask"][str(camera)] = annot["Mask/0"]
                annot["Camera_Parameter"][f"{camera:02d}"] = annot["Camera_Parameter/00"]
        annot.create_group("Keypoints_3D")["keypoints3d"] = np.zeros((2, 25, 4))
        annot.create_group("Keypoints_2D")["00"] = np.zeros((2, 25, 3))
    report = inspect_sequence(root, "rgb.smc", "annotations.smc")
    assert report["format_checks_passed"] and not report["training_ready"]
    assert len(report["cameras"]) == 60 and report["coverage"]["synchronized_frame_ids"]
    assert report["sources"]["rgb"]["sha256"] == file_sha256(root / "rgb.smc")
    assert report["annotation_schema"]["Keypoints_3D/keypoints3d"]["shape"] == [2, 25, 4]
    assert report["cameras"]["48"]["samples"][0]["height"] == 32
    assert report["body_fields"]["scale"]["sampled_min"] == 1
    with h5py.File(root / "annotations.smc", "r+") as annot:
        del annot["Mask/48/mask/1"]
    report = inspect_sequence(root, "rgb.smc", "annotations.smc")
    assert not report["format_checks_passed"]
    assert any("Camera 48: RGB/mask frame coverage differs" in x for x in report["errors"])
