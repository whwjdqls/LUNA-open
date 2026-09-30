"""Compute-node preflight and checkpoint audit for the new identity run."""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import yaml

from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.provenance import file_sha256
from luna_open.training import FeatureStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU compute node")
    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    report = dict(
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
        config_sha256=file_sha256(args.config),
    )
    if not args.checkpoint:
        features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
        assert features.metadata["face"]["face_backbone"] == "sapiens"
        assert features.metadata["face"]["asset"] == features.metadata["body"]["asset"]
        inventory = []
        for scene, info in manifest["scenes"].items():
            splits = info["splits"]
            assert set(splits["train"]).isdisjoint(splits["val"] + splits["test"])
            assert set(splits["val"]).isdisjoint(splits["test"])
            assert len(info["references"]) == len(set(info["references"])) == 4
            assert set(info["references"]).issubset(splits["train"])
            for frame in info["frames"]:
                path = features.root / "face" / scene / (Path(frame["name"]).stem + ".pt")
                tensor = torch.load(path, weights_only=True, map_location="cpu")["features"]
                assert tensor.shape == (1024, 1536) and tensor.dtype == torch.float16
                assert tensor.isfinite().all() and tensor.float().std() > 0
                inventory.append(dict(scene=scene, frame=frame["name"], sha256=file_sha256(path)))
        assert len(inventory) == 429
        report.update(
            face_feature_count=len(inventory),
            feature_metadata=features.metadata,
            face_feature_inventory=inventory,
        )
        if args.legacy_checkpoint:
            legacy = torch.load(args.legacy_checkpoint, weights_only=False, map_location="cpu")
            old = IdentityEncoder(
                legacy["identity"]["anchors"],
                legacy["identity"]["semantic_labels"],
                ModelConfig(**legacy["config"]["model"]),
            )
            old.load_state_dict(legacy["identity"], strict=True)
            assert not any(block.qk_norm for block in old.blocks)
            report["legacy_checkpoint_strict_load"] = True
            report["legacy_checkpoint_sha256"] = file_sha256(args.legacy_checkpoint)
            del old, legacy
    else:
        state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
        assert state["config"] == cfg and state["trainer"] == "identity_multitarget_v2"
        assert state["stage"] == "identity"
        assert state["update"] > 0
        finite_parameters = all(v.isfinite().all() for v in state["identity"].values())
        assert finite_parameters
        adam_steps = [int(value["step"]) for value in state["optimizer"]["state"].values()]
        assert adam_steps and min(adam_steps) == max(adam_steps) == state["update"]
        for sample in state["last_sampled"]:
            assert len(sample["references"]) == 4
            assert len(sample["targets"]) == cfg["training"]["targets_per_identity"]
            assert set(sample["references"]).isdisjoint(sample["targets"])
            assert set(sample["references"] + sample["targets"]).issubset(
                manifest["scenes"][sample["scene"]]["splits"]["train"]
            )
        identity = IdentityEncoder(
            state["identity"]["anchors"],
            state["identity"]["semantic_labels"],
            ModelConfig(**cfg["model"]),
        )
        identity.load_state_dict(state["identity"], strict=True)
        identity.cuda().eval().requires_grad_(False)
        assert all(block.qk_norm for block in identity.blocks)
        assert identity.fp32_decoder and identity.face_encoder == "sapiens"
        assert not hasattr(identity, "face_fusion")
        features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
        assert features.metadata == state["feature_metadata"]
        scene = next(iter(manifest["scenes"]))
        refs = dict(scene=scene, reference_names=manifest["scenes"][scene]["references"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(*features.references(refs))
        canonical.gaussians.validate()
        assert canonical.gaussians.means.dtype == torch.float32
        assert canonical.tokens.shape == (1, cfg["num_queries"], cfg["model"]["width"])
        report.update(
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=file_sha256(args.checkpoint),
            update=state["update"],
            finite_parameters=bool(finite_parameters),
            adam_states=len(adam_steps),
            adam_step_min=min(adam_steps),
            adam_step_max=max(adam_steps),
            inference_gaussians_valid=True,
            canonical_tokens_shape=list(canonical.tokens.shape),
            reference_target_disjoint=True,
            identity_qk_norm=True,
            face_encoder="sapiens",
            fp32_decoder=True,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"face_feature_inventory", "feature_metadata"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
