"""Render the frozen identity checkpoint with the diagnostic common canonical cameras."""

import json
import os
import socket
from pathlib import Path

import torch
from diagnose_native_avatar import orbit
from evaluate_lhm_neuman import save_tensor
from render_qualitative import load_snapshot

from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    torch.set_num_threads(3)
    work = Path("/scratch2/whwjdqls99/LUNA-open")
    base = work / "baselines/lhm-20260928"
    out = work / "diagnostics/canonical-ours-20260928"
    out.mkdir(exist_ok=False)
    state, checksum = load_snapshot(base / "identity-snapshots/identity-014750.pt")
    cfg = state["config"]
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg["smpl_pose_blend_shapes"],
    ).cuda()
    identity = IdentityEncoder(
        teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"])
    )
    identity.load_state_dict(state["identity"], strict=True)
    identity = identity.cuda().eval()
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    assert features.metadata == state["feature_metadata"]
    protocol = json.loads((base / "protocol-test/protocol.json").read_text())
    camera = json.loads((base / "ours-identity-14750-test/canonical-camera.json").read_text())
    K = torch.tensor(camera["K"], device="cuda")[None]
    for scene, info in protocol["scenes"].items():
        (out / scene).mkdir()
        body, face = features.references(dict(scene=scene, reference_names=info["references"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        for angle in range(0, 360, 30):
            pred = render(canonical.gaussians, K, (512, 512), orbit(camera, angle)[None])
            save_tensor(pred["rgb"][0], out / scene / f"{angle:03}.png")
        print(scene, flush=True)
    (out / "method.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256=checksum,
                update=state["update"],
                host=socket.gethostname(),
                job=os.environ["SLURM_JOB_ID"],
                camera=camera,
                note="Frozen original reference set, no canonical realignment required for SMPL coordinates.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
