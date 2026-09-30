"""Render frozen old/new identities on the same official validation frames."""

import gc
import json
import os
import socket
import time
from pathlib import Path

import torch
from render_identity_animation import panel, virtual_camera
from render_qualitative import load_snapshot, rgb_image, stack_panels

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.metrics import aggregate_records, image_metrics
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, prepare_item, seed_all

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
OUT = WORK / "reports/luna-progress-20260928"
PRIVATE = WORK / "reports/luna-progress-20260928-work"


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use an allocated GPU compute node")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.25)
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("Need 8 GiB free beside the running training process")
    started = time.perf_counter()
    paths = json.loads((PRIVATE / "snapshots.json").read_text())
    metadata = json.loads((OUT / "evidence/report-snapshot.json").read_text())
    destination = OUT / "qualitative/10_same_frame_identity_comparison"
    destination.mkdir(exist_ok=False)
    cfg = metadata["models"]["retrained_identity"]["config"]
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    seed_all(cfg["seed"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg["smpl_pose_blend_shapes"],
    ).cuda()
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], split="val", size=cfg["image_size"])
    perceptual = build_lpips("alex", device="cuda")
    report = dict(
        job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),
        split="val",
        reference_protocol="same four fixed training references; all official 44 validation frames",
        panels={},
        models={},
        script_sha256=file_sha256(__file__),
    )
    # Use identical virtual cameras for both models, derived from fixed template
    # anchors; canonical views have no canonical ground-truth images.
    template = Gaussians(
        teacher.anchors[None], None, torch.full_like(teacher.anchors[None], 0.008), None, None
    )
    cameras = {
        name: virtual_camera(template, 512, angle)
        for name, angle in [("front", 0), ("side", 90), ("back", 180)]
    }
    pictures = {}
    for label in ("original_identity", "retrained_identity"):
        state, checksum = load_snapshot(Path(paths[label]))
        config = state["config"]
        assert checksum == metadata["models"][label]["snapshot_sha256"]
        assert state["manifest_sha256"] == file_sha256(config["manifest"])
        validate_body_asset(state, file_sha256(config["smpl_model"]))
        assert torch.equal(teacher.anchors.cpu(), state["identity"]["anchors"])
        identity = (
            IdentityEncoder(
                teacher.anchors, teacher.semantic_labels, ModelConfig(**config["model"])
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        identity.load_state_dict(state["identity"], strict=True)
        features = FeatureStore(config["features"], config["manifest"], ("body", "face"))
        assert features.metadata == state["feature_metadata"]
        records = []
        pictures[label] = {}
        for scene, info in manifest["scenes"].items():
            refs = info["references"]
            assert set(refs).issubset(info["splits"]["train"])
            assert set(refs).isdisjoint(info["splits"]["val"])
            scene_dir = destination / scene
            scene_dir.mkdir(exist_ok=True)
            body, face = features.references(dict(scene=scene, reference_names=refs))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                canonical = identity(body, face)
            canonical.gaussians.validate()
            images = {}
            for view, (K, viewmat) in cameras.items():
                pred = render(canonical.gaussians, K, (512, 512), viewmat)
                images[view] = rgb_image(pred["rgb"][0])
                images[view].save(scene_dir / f"{label}-canonical-{view}.png")
            panel(
                list(images.values()),
                tuple(images),
                f"{scene} | {label} | step {state['update']:,}",
                "Canonical render; shared virtual cameras; no canonical GT",
                tile=512,
            ).save(scene_dir / f"{label}-canonical-views.png")
            pictures[label][scene] = dict(canonical=images, frames={})
            for name in info["splits"]["val"]:
                item = data.load_frame(scene, name)
                target = prepare_item(item)
                posed = teacher(
                    canonical.gaussians, target["pose"], target["betas"], target["body_to_camera"]
                )
                pred = render(posed, target["K"], (512, 512))
                image = rgb_image(pred["rgb"][0])
                image.save(scene_dir / f"{label}-{name}")
                pictures[label][scene]["frames"][name] = image
                if label == "original_identity":
                    rgb_image(target["rgb"][0]).save(scene_dir / f"gt-{name}")
                metrics = {
                    key: float(value[0])
                    for key, value in image_metrics(
                        pred, target["rgb"], target["mask"], perceptual
                    ).items()
                }
                error = (pred["rgb"] - target["rgb"]).abs()
                mask = target["mask"].expand_as(error)
                metrics["foreground_l1"] = float((error * mask).sum() / mask.sum().clamp_min(1))
                records.append(dict(scene=scene, frame=name, metrics=metrics))
            references = [rgb_image(data.load_frame(scene, name)["rgb"]) for name in refs]
            panel(
                references,
                refs,
                f"{scene} | Shared reference frames",
                "All four inputs belong to the official training split",
                tile=256,
            ).save(scene_dir / "reference-inputs.png")
        result = aggregate_records(records)
        result.update(
            stage="identity",
            split="val",
            update=state["update"],
            snapshot_sha256=checksum,
            model=label,
        )
        (OUT / "quantitative" / f"{label}-same-frame-val.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        report["models"][label] = dict(
            update=state["update"], snapshot_sha256=checksum, metrics=result["mean_over_scenes"]
        )
        print(
            json.dumps(
                dict(model=label, update=state["update"], metrics=result["mean_over_scenes"])
            ),
            flush=True,
        )
        del identity, state, features, canonical, body, face, pred, posed, target
        gc.collect()
        torch.cuda.empty_cache()
    from PIL import Image

    overview = []
    for scene, info in manifest["scenes"].items():
        scene_dir = destination / scene
        names = info["splits"]["val"]
        middle = names[len(names) // 2]
        comparisons = []
        canonical_gifs = []
        for name in names:
            truth = Image.open(scene_dir / f"gt-{name}").convert("RGB")
            old = pictures["original_identity"][scene]["frames"][name]
            new = pictures["retrained_identity"][scene]["frames"][name]
            comparison = panel(
                [truth, old, new],
                [
                    "Ground truth",
                    "Original identity 10k",
                    "Retrained identity " + str(report["models"]["retrained_identity"]["update"]),
                ],
                f"{scene} | validation {name}",
                "Same references, target camera, crop and fitted SMPL/LBS pose",
                tile=512,
            )
            comparison.save(scene_dir / f"comparison-{Path(name).stem}.png")
            comparisons.append(comparison.resize((1056, 414)))
            sequence = panel(
                [truth, pictures["retrained_identity"][scene]["canonical"]["front"], new],
                ["Ground truth", "Canonical identity", "Fitted SMPL pose (LBS)"],
                f"{scene} | retrained identity | validation {name}",
                "Canonical middle column is fixed; right column uses annotated pose",
                tile=384,
            )
            canonical_gifs.append(sequence)
            if name == middle:
                comparison.save(destination / f"{scene}-comparison.png")
                sequence.save(destination / f"{scene}-gt-canonical-lbs.png")
                overview.append(comparison)
                report["panels"][scene] = dict(
                    frame=name,
                    selection="middle frame of the official validation list",
                    references=info["references"],
                )
        comparisons[0].save(
            destination / f"{scene}-old-vs-new.gif",
            save_all=True,
            append_images=comparisons[1:],
            duration=300,
            loop=0,
        )
        canonical_gifs[0].save(
            destination / f"{scene}-gt-canonical-lbs.gif",
            save_all=True,
            append_images=canonical_gifs[1:],
            duration=300,
            loop=0,
        )
    stack_panels(overview[:3]).save(destination / "overview-1.png")
    stack_panels(overview[3:]).save(destination / "overview-2.png")
    report.update(
        seconds=time.perf_counter() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        frames_per_model=44,
    )
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(completed=True, seconds=report["seconds"], peak_gib=report["peak_allocated_gib"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
