import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_evaluation_records",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_evaluation_records.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def evaluation():
    manifest = {
        "scenes": {
            "a": {"splits": {"val": ["v1", "v2"], "test": ["t1", "t2"]}},
            "b": {"splits": {"val": ["v3"], "test": ["t3"]}},
        }
    }
    report = {
        "manifest_sha256": "manifest-digest",
        "checkpoint": "/example/identity/best.pt",
        "protocol": audit.PROTOCOL,
        "frames": [
            {"scene": scene, "frame": frame, "metrics": dict.fromkeys(audit.METRICS, score)}
            for scene, frame, score in [("a", "v1", 0.1), ("a", "v2", 0.2), ("b", "v3", 0.4)]
        ],
        "per_scene": {
            "a": dict.fromkeys(audit.METRICS, 0.15),
            "b": dict.fromkeys(audit.METRICS, 0.4),
        },
        "mean_over_scenes": dict.fromkeys(audit.METRICS, 0.275),
    }
    return report, manifest


def inspect(report, manifest, split="val"):
    return audit.inspect_evaluation(
        report, manifest, split, "/example/identity/best.pt", "manifest-digest"
    )


@pytest.mark.parametrize("split", ["val", "test"])
def test_exact_membership_and_macro_aggregation(evaluation, split):
    report, manifest = evaluation
    if split == "test":
        for row in report["frames"]:
            row["frame"] = row["frame"].replace("v", "t")
    before = deepcopy(report)
    result = inspect(report, manifest, split)
    assert result["frames"] == 3
    assert result["frames_per_scene"] == {"a": 2, "b": 1}
    assert result["mean_over_scenes"]["lpips"] == 0.275
    assert result["aggregation_max_abs_difference"] < 1e-12
    assert report == before


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing",
        "wrong_frame",
        "wrong_scene",
        "wrong_manifest",
        "wrong_checkpoint",
        "wrong_protocol",
        "missing_metric",
        "extra_metric",
        "bool_metric",
        "nan_frame",
        "inf_aggregate",
        "wrong_scene_mean",
        "wrong_macro_mean",
        "frame_weighted_mean",
        "missing_scene",
        "duplicate_manifest",
    ],
)
def test_rejects_corrupted_evaluation(evaluation, corruption):
    report, manifest = evaluation
    if corruption == "duplicate":
        report["frames"].append(deepcopy(report["frames"][0]))
    elif corruption == "missing":
        report["frames"].pop()
    elif corruption == "wrong_frame":
        report["frames"][0]["frame"] = "t1"
    elif corruption == "wrong_scene":
        report["frames"][0]["scene"] = "unknown"
    elif corruption == "wrong_manifest":
        report["manifest_sha256"] = "different-manifest"
    elif corruption == "wrong_checkpoint":
        report["checkpoint"] = "/example/identity/latest.pt"
    elif corruption == "wrong_protocol":
        report["protocol"] = "unrelated protocol"
    elif corruption == "missing_metric":
        del report["frames"][0]["metrics"]["lpips"]
    elif corruption == "extra_metric":
        report["frames"][0]["metrics"]["extra"] = 0.1
    elif corruption == "bool_metric":
        report["frames"][0]["metrics"]["lpips"] = True
    elif corruption == "nan_frame":
        report["frames"][0]["metrics"]["lpips"] = float("nan")
    elif corruption == "inf_aggregate":
        report["mean_over_scenes"]["lpips"] = float("inf")
    elif corruption == "wrong_scene_mean":
        report["per_scene"]["a"]["lpips"] += 0.01
    elif corruption == "wrong_macro_mean":
        report["mean_over_scenes"]["lpips"] += 0.01
    elif corruption == "frame_weighted_mean":
        report["mean_over_scenes"]["lpips"] = 0.7 / 3
    elif corruption == "missing_scene":
        del report["per_scene"]["b"]
    elif corruption == "duplicate_manifest":
        manifest["scenes"]["a"]["splits"]["val"].append("v1")
    with pytest.raises(ValueError):
        inspect(report, manifest)


def test_validation_frames_cannot_pass_as_test(evaluation):
    report, manifest = evaluation
    with pytest.raises(ValueError, match="frame membership"):
        inspect(report, manifest, "test")


def test_file_snapshots(tmp_path):
    path = tmp_path / "records.json"
    path.write_bytes(b"{}\n")
    text, digest = audit.stable_text(path)
    assert text == "{}\n"
    assert digest == audit.stable_digest(path)


def test_requires_compute_node_before_reading_config(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["audit", "--config", "/missing-config", "--output", "/missing-output"]
    )
    with pytest.raises(RuntimeError, match="compute node"):
        audit.main()
