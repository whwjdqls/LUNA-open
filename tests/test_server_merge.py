"""Exercise interfaces shared by the Yonsei experiments and PARCC DNA work."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from luna_open.provenance import file_sha256

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backbone", ["dinov2", "sapiens"])
def test_face_cache_audit_supports_both_server_protocols(tmp_path, backbone):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenes": {"actor": {"frames": [{"name": "0.png"}]}}}))
    folder = tmp_path / "features/face"
    (folder / "actor").mkdir(parents=True)
    catalog = yaml.safe_load((REPO / "configs/assets.yaml").read_text())
    sapiens = backbone == "sapiens"
    metadata = dict(
        manifest_sha256=file_sha256(manifest),
        kind="face",
        asset=catalog["sapiens_body" if sapiens else "dino_face"],
        preprocessing_version=2 if sapiens else 1,
        dtype="float16",
    )
    if sapiens:
        metadata["face_backbone"] = backbone
    (folder / "metadata.json").write_text(json.dumps(metadata))
    shape = (1024, 1536) if sapiens else (4, 1024, 1024)
    features = torch.zeros(shape, dtype=torch.float16)
    features.reshape(-1)[::2] = 1
    path = folder / "actor/0.pt"
    torch.save(dict(features=features, crop_source="fixture"), path)
    command = [
        sys.executable,
        str(REPO / "scripts/verify_features.py"),
        "--manifest",
        str(manifest),
        "--features",
        str(folder.parent),
        "--kinds",
        "face",
        "--output",
        str(tmp_path / "audit.json"),
    ]
    subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)
    assert json.loads((tmp_path / "audit.json").read_text())["kinds"]["face"]["shape"] == list(
        shape
    )
    torch.save(dict(features=features.reshape(-1)[:10], crop_source="fixture"), path)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0 and "Wrong feature shape/dtype" in result.stderr


def test_yonsei_runtime_copy_retains_links_and_rejects_changed_target(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "runtime", REPO / "scripts/prepare_lhm_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "compute-fixture")
    (tmp_path / "references/LHM").mkdir(parents=True)
    (tmp_path / "references/LHM/source.py").write_text("# native fixture\n")
    (tmp_path / "assets/lhm-priors/pretrained_models").mkdir(parents=True)
    (tmp_path / "assets/lhm-priors/gfpgan").mkdir()
    weights = tmp_path / "assets/sapiens_body/sapiens_1b_epoch_173_torchscript.pt2"
    weights.parent.mkdir()
    weights.write_bytes(b"fixture")
    runtime = tmp_path / "runtime/source"
    module.prepare_source(tmp_path, runtime)
    module.prepare_source(tmp_path, runtime)
    assert (
        runtime / "pretrained_models"
    ).resolve() == tmp_path / "assets/lhm-priors/pretrained_models"
    assert (runtime / "source.py").read_text() == "# native fixture\n"
    link = runtime / "gfpgan"
    link.unlink()
    link.symlink_to(weights.parent)
    with pytest.raises(ValueError, match="another target"):
        module.prepare_source(tmp_path, runtime)
