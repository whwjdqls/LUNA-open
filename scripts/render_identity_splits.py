"""Visualize the latest identity checkpoint on complete NeuMan train/test splits.

Canonical identity Gaussians are shown directly through virtual cameras. For
frame reconstruction, the existing fixed SMPL teacher applies each annotation's
pose/betas/body-to-camera transform, as in identity-stage evaluation. This is
identity visualization, not the neural animator. Four fixed training references
are shared across both splits. Training targets that are references are labeled.
"""

import argparse
import gc
import html
import json
import os
import socket
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from render_identity_animation import panel, virtual_camera
from render_qualitative import load_snapshot, rgb_image, stack_panels

from luna_open.data.neuman import NeuManDataset
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, prepare_item, seed_all


def write_gallery(path, title, introduction, body):
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1320px;margin:32px auto;"
        "padding:0 20px;background:#f4f6f9;color:#17212b}img{max-width:100%;height:auto}"
        "figure{margin:24px 0}figcaption,p{line-height:1.6}a{color:#174ba0}"
        ".frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}"
        "</style>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(introduction)}</p>{body}</html>\n"
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Use a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for rendering")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.25)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free in the existing allocation")
    started = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    state, checkpoint_hash = load_snapshot(args.checkpoint)
    manifest_hash = file_sha256(cfg["manifest"])
    if state["stage"] != "identity" or state["config"] != cfg:
        raise ValueError("Identity checkpoint/config mismatch")
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest mismatch")
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    log_path = Path(cfg["output"]) / "identity/train.jsonl"
    last_update = json.loads(log_path.read_text().splitlines()[-1])["update"]
    if state["update"] != last_update:
        raise ValueError("Checkpoint is not at the latest logged identity update")
    if args.checkpoint.resolve() != (Path(cfg["output"]) / "identity/latest.pt").resolve():
        raise ValueError("Use the latest identity checkpoint")
    update = state["update"]
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    if features.metadata != state["feature_metadata"]:
        raise ValueError("Feature metadata mismatch")
    seed_all(cfg["seed"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    if not torch.equal(teacher.anchors.cpu(), state["identity"]["anchors"]) or not torch.equal(
        teacher.semantic_labels.cpu(), state["identity"]["semantic_labels"]
    ):
        raise ValueError("SMPL query correspondence mismatch")
    identity = (
        IdentityEncoder(teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"]))
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    identity.load_state_dict(state["identity"])
    args.output.mkdir(parents=True)
    snapshot_keys = (
        "stage",
        "update",
        "config",
        "identity",
        "manifest_sha256",
        "smpl_asset_sha256",
        "feature_metadata",
    )
    torch.save({key: state[key] for key in snapshot_keys}, args.output / "identity-snapshot.pt")
    del state
    gc.collect()
    data = {
        split: NeuManDataset(cfg["data_root"], cfg["manifest"], split=split, size=cfg["image_size"])
        for split in ("train", "test")
    }
    scenes = list(dict.fromkeys(scene for scene, _ in data["train"].items))
    split_records = {split: {} for split in data}
    selected = {split: {} for split in data}
    canonical_records, processed = {}, {split: [] for split in data}
    index_sections, combined_panels = [], []
    for scene in scenes:
        metadata = manifest["scenes"][scene]
        refs = metadata["references"]
        if (
            len(refs) != 4
            or len(set(refs)) != 4
            or not set(refs).issubset(metadata["splits"]["train"])
        ):
            raise ValueError("Expected four distinct training references")
        if set(metadata["splits"]["train"]) & set(metadata["splits"]["test"]):
            raise ValueError("Train/test overlap")
        destination = args.output / "canonical" / scene
        destination.mkdir(parents=True)
        body, face = features.references({"scene": scene, "reference_names": refs})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        canonical.gaussians.validate()
        torch.save(
            {
                "gaussians": {
                    f.name: getattr(canonical.gaussians, f.name).cpu()
                    for f in fields(canonical.gaussians)
                },
                "tokens": canonical.tokens.cpu(),
                "references": refs,
                "coordinates": "SMPL canonical meters, quaternion wxyz, batch dimension retained",
            },
            destination / "output.pt",
        )
        views, camera_records = [], {}
        for name, angle in (("front", 0), ("side", 90), ("back", 180)):
            intrinsic, view = virtual_camera(canonical.gaussians, cfg["image_size"], angle)
            rendered = render(
                canonical.gaussians, intrinsic, (cfg["image_size"], cfg["image_size"]), view
            )
            picture = rgb_image(rendered["rgb"][0])
            picture.save(destination / f"{name}.png")
            views.append(picture)
            camera_records[name] = {
                "K": intrinsic.cpu().tolist(),
                "world_to_camera": view.cpu().tolist(),
            }
        panel(
            views,
            ("Front", "Side", "Back"),
            f"{scene} | Canonical identity | step {update:,}",
            "Four fixed training references; direct encoder output through virtual cameras",
        ).save(destination / "views.jpg", quality=95)
        reference_images = [
            rgb_image(data["train"].load_frame(scene, name)["rgb"]) for name in refs
        ]
        panel(
            reference_images,
            tuple(refs),
            f"{scene} | Four training reference images",
            "These same inputs define the canonical identity for both train and test galleries",
            tile=256,
        ).save(destination / "references.jpg", quality=95)
        canonical_records[scene] = {
            "references": refs,
            "cameras": camera_records,
            "output_sha256": file_sha256(destination / "output.pt"),
        }
        for split, dataset in data.items():
            directory = args.output / split / scene
            directory.mkdir(parents=True)
            names = metadata["splits"][split]
            eligible = [name for name in names if name not in refs]
            chosen = [eligible[i] for i in sorted({0, len(eligible) // 2, len(eligible) - 1})]
            frame_panels, chosen_panels, records, figures = [], [], [], []
            for name in names:
                item = dataset.load_frame(scene, name)
                target = prepare_item(item)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    posed = teacher(
                        canonical.gaussians,
                        target["pose"],
                        target["betas"],
                        target["body_to_camera"],
                        detach=True,
                    )
                prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
                truth, reconstruction = rgb_image(item["rgb"]), rgb_image(prediction["rgb"][0])
                truth.save(directory / f"target-{name}")
                reconstruction.save(directory / f"identity-{name}")
                rgb_image(prediction["alpha"][0].expand(3, -1, -1)).save(
                    directory / f"alpha-{name}"
                )
                is_reference = name in refs
                tag = " | reference input" if is_reference else ""
                comparison = panel(
                    (truth, reconstruction),
                    ("Ground truth", "Identity + fitted SMPL pose"),
                    f"{scene} | {split} {name} | identity {update:,}{tag}",
                    "Same fixed training references; annotated foreground crop; 512 px renders",
                )
                comparison_name = f"comparison-{Path(name).stem}.jpg"
                comparison.save(directory / comparison_name, quality=94)
                frame_panels.append(comparison.resize((608, 368)))
                if name in chosen:
                    chosen_panels.append(comparison)
                if name == eligible[len(eligible) // 2]:
                    selected[split][scene] = {
                        "frame": name,
                        "truth": truth,
                        "prediction": reconstruction,
                        "panel": comparison,
                    }
                records.append({"frame": name, "reference_input": is_reference})
                processed[split].append((scene, name))
                figures.append(
                    f'<figure><a href="{comparison_name}"><img loading="lazy" src="{comparison_name}" alt="{html.escape(scene + " " + split + " " + name)}"></a><figcaption>{html.escape(name + tag)}</figcaption></figure>'
                )
            stack_panels(chosen_panels).save(directory / "first-middle-last.jpg", quality=95)
            selected[split][scene]["panel"].save(directory / "overview.jpg", quality=95)
            frame_panels[0].save(
                directory / "sequence.gif",
                save_all=True,
                append_images=frame_panels[1:],
                duration=500,
                loop=0,
            )
            write_gallery(
                directory / "index.html",
                f"{scene}: {split} — identity {update:,}",
                f"{len(names)} frames. Left: ground truth. Right: canonical identity posed with fitted SMPL. Reference inputs are labeled. GIF display is 2 fps, not capture timing.",
                '<p><a href="../../index.html">All subjects</a> · <a href="sequence.gif">Sequence GIF</a> · <a href="first-middle-last.jpg">First/middle/last</a></p><div class="frames">'
                + "".join(figures)
                + "</div>",
            )
            split_records[split][scene] = {
                "count": len(names),
                "references": refs,
                "selected_frames": chosen,
                "records": records,
            }
            print(
                json.dumps(
                    {
                        "event": "split_scene_complete",
                        "split": split,
                        "scene": scene,
                        "frames": len(names),
                    }
                ),
                flush=True,
            )
        train, test = selected["train"][scene], selected["test"][scene]
        combined = panel(
            (train["truth"], train["prediction"], test["truth"], test["prediction"]),
            (
                "Train: ground truth",
                "Train: identity output",
                "Test: ground truth",
                "Test: identity output",
            ),
            f"{scene} | identity {update:,} | train {train['frame']} / test {test['frame']}",
            "Identity outputs use fitted SMPL poses; same four training references; both shown targets excluded from references",
            tile=320,
        )
        combined_panels.append(combined)
        combined.save(args.output / f"{scene}-train-test.jpg", quality=95)
        index_sections.append(
            f'<h2>{html.escape(scene)}</h2><p><a href="train/{scene}/index.html">Train: {len(metadata["splits"]["train"])} frames</a> · <a href="test/{scene}/index.html">Test: {len(metadata["splits"]["test"])} frames</a> · <a href="canonical/{scene}/views.jpg">Canonical 3D views</a> · <a href="canonical/{scene}/references.jpg">Reference images</a></p><img loading="lazy" src="{scene}-train-test.jpg" alt="{scene} train and test reconstruction">'
        )
        del canonical, body, face, posed, prediction, target, rendered
    for split, dataset in data.items():
        if processed[split] != dataset.items:
            raise ValueError(f"Incomplete or reordered {split} membership")
        for offset in range(0, len(scenes), 3):
            stack_panels(
                [selected[split][scene]["panel"] for scene in scenes[offset : offset + 3]]
            ).save(args.output / split / f"overview-{offset // 3 + 1}.jpg", quality=95)
    for offset in range(0, len(combined_panels), 3):
        stack_panels(combined_panels[offset : offset + 3]).save(
            args.output / f"train-test-{offset // 3 + 1}.jpg", quality=95
        )
    write_gallery(
        args.output / "index.html",
        f"Identity encoder {update:,}: train and test",
        "All 344 train frames and 41 test frames from the same six NeuMan subjects. Both splits use four fixed training reference images per subject. The canonical identity is rendered with fitted SMPL poses for comparison. This evaluates held-out frames of seen subjects, not unseen identities.",
        "".join(index_sections),
    )
    torch.cuda.synchronize()
    artifact_paths = sorted(path for path in args.output.rglob("*") if path.is_file())
    artifact_inventory = {
        str(path.relative_to(args.output)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }
    if any(record["bytes"] == 0 for record in artifact_inventory.values()):
        raise ValueError("Empty output file")
    root = Path(__file__).resolve().parents[1]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "gpu": torch.cuda.get_device_name(),
        "seconds": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "identity_update": update,
        "latest_logged_identity_update": last_update,
        "snapshot_sha256": file_sha256(args.output / "identity-snapshot.pt"),
        "manifest_sha256": manifest_hash,
        "config_sha256": file_sha256(args.config),
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                root / "scripts/render_identity_animation.py",
                root / "scripts/render_qualitative.py",
                root / "src/luna_open/model.py",
                root / "src/luna_open/smpl.py",
                root / "src/luna_open/rendering.py",
            )
        },
        "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "initial_free_gpu_bytes": free_bytes,
        "total_gpu_bytes": total_bytes,
        "counts": {split: len(rows) for split, rows in processed.items()},
        "complete_manifest_membership": True,
        "canonical": canonical_records,
        "splits": split_records,
        "protocol": "Seen subjects; official frame splits; fixed four training references shared across splits; reference train targets explicitly labeled",
        "reconstruction": "Identity canonical Gaussians deformed by fitted SMPL pose/betas/body_to_camera, then rendered with annotated crop K; no NeuralAnimator",
        "selection": "All frames rendered. Overview middle non-reference target; per-scene sheet first/middle/last non-reference targets, independent of scores",
        "playback": "All split frames at 2 fps display, no interpolation, capture timing unverified",
        "artifact_inventory": artifact_inventory,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "counts": report["counts"],
                "seconds": report["seconds"],
                "peak_bytes": report["max_allocated_gpu_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
