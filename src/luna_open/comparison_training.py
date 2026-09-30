"""Single-GPU controlled retraining with the same corpus and update contract.

This is a controlled photometric experiment. LUNA's paper supervision remains
available through luna_open.training; shared-photometric LUNA is an explicitly
labeled objective ablation. Released reconstruction checkpoints are excluded.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from .comparison import (
    SharedSampler,
    digest_document,
    learning_rate_factor,
    load_contract,
    optimizer_groups,
    validate_motion_receipt,
)
from .data import dataset_from_manifest
from .metrics import aggregate_records, image_metrics
from .model import IdentityEncoder, ModelConfig, NeuralAnimator
from .native_baselines import NativeReconstruction, blank_motion
from .perceptual import build_lpips
from .provenance import file_sha256, verify_sources
from .rendering import render
from .smpl import SMPLTeacher
from .training import (
    FeatureStore,
    atomic_checkpoint,
    prepare_item,
    restore_rng,
    rng_state,
    seed_all,
    translation_statistics,
)
from .upstream_objective import ReleasedPhotometricObjective


def path(value):
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError("Unresolved server variable in path")
    return Path(expanded).expanduser().resolve()


def runtime_environment():
    """Execution evidence; devices may differ while the comparison contract matches."""
    import importlib.metadata

    packages = {}
    for name in ("torch", "torchvision", "numpy", "gsplat", "xformers", "flash-attn", "lpips"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        python=platform.python_version(),
        packages=packages,
        torch_cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        host=platform.node(),
        job_id=os.environ.get("SLURM_JOB_ID"),
    )


class Corpus:
    def __init__(self, configuration, contract, method, verify=True):
        self.contract, self.method = contract, method
        self.datasets, self.manifests, self.fit_documents, self.sources = {}, {}, {}, {}
        self.configuration = configuration
        weights = {}
        for name, entry in sorted(configuration["datasets"].items()):
            root, manifest_path = path(entry["root"]), path(entry["manifest"])
            document = json.loads(manifest_path.read_text())
            if verify:
                verify_sources(root, document)
            self.manifests[name] = document
            self.datasets[name] = dataset_from_manifest(
                root,
                manifest_path,
                size=contract["image_size"],
                require_smpl=True,
                verify=False,
            )
            weights[name] = entry["weight"]
            self.sources[name] = dict(manifest_sha256=file_sha256(manifest_path))
            # All methods require the same admitted corpus, including validated
            # native labels, before beginning a controlled comparison.
            if not entry.get("native_motion"):
                raise ValueError(f"Validated native SMPL-X motion receipt missing: {name}")
            fit_path = path(entry["native_motion"])
            fits = json.loads(fit_path.read_text())
            validate_motion_receipt(document, self.sources[name]["manifest_sha256"], fits)
            if entry["native_coordinate_frame"] not in {"body", "world"}:
                raise ValueError("Native fit coordinate frame must be explicit")
            self.fit_documents[name] = fits
            self.sources[name]["native_motion_sha256"] = file_sha256(fit_path)
        self.sampler = SharedSampler(contract, self.manifests, weights)
        self.sha256 = digest_document(
            dict(
                sources=self.sources,
                membership=self.sampler.membership_sha256,
                coordinate_frames={
                    name: entry["native_coordinate_frame"]
                    for name, entry in configuration["datasets"].items()
                },
            )
        )
        self.face_crop = None

    def item(self, choice):
        name, scene, frame = choice["dataset"], choice["scene"], choice["frame"]
        data = self.datasets[name]
        item = data.load_frame(scene, frame)
        item.update(
            scene=scene,
            frame=frame,
            dataset=name,
            reference_names=choice["references"],
            reference_images=torch.stack(
                [data.load_frame(scene, ref)["rgb"] for ref in choice["references"]]
            ),
        )
        return item

    def native_inputs(self, item):
        if self.face_crop is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
            from cache_features import face_image

            self.face_crop = face_image
        name, scene, frame = item["dataset"], item["scene"], item["frame"]
        data = self.datasets[name]
        fitted = self.fit_documents[name]["scenes"][scene][frame]
        motion = blank_motion(torch.device("cuda"))
        # Omitted face/eye/expression fields explicitly mean zero in the admitted
        # NeuMan conversion; native DNA labels need their own validated mapping.
        for key in motion:
            if key not in fitted and key not in {"jaw_pose", "leye_pose", "reye_pose", "expr"}:
                raise ValueError(f"Native body field missing: {name}/{scene}/{frame}/{key}")
            if key in fitted:
                tensor = torch.tensor(fitted[key], dtype=torch.float32, device="cuda")
                if tensor.shape != motion[key].shape[1:] and tensor.shape != motion[key].shape[2:]:
                    raise ValueError(f"Native body shape differs: {key}")
                motion[key] = tensor.reshape(motion[key].shape)
                if not torch.isfinite(motion[key]).all():
                    raise ValueError(f"Nonfinite native body field: {key}")
        coordinate = self.configuration["datasets"][name]["native_coordinate_frame"]
        w2c = item["body_to_camera"] if coordinate == "body" else item["world_to_camera"]
        faces = torch.stack(
            [self.face_crop(data, scene, ref, size=112)[0] for ref in item["reference_names"]]
        )
        return (
            item["reference_images"][None].cuda(),
            faces[None].cuda(),
            motion,
            w2c[None].cuda(),
            item["K"][None].cuda(),
            (self.contract["image_size"],) * 2,
        )

    def evaluation_choices(self, split):
        for name, document in sorted(self.manifests.items()):
            for scene, info in sorted(document["scenes"].items()):
                refs = info["references"][: self.contract["reference_count"]]
                if len(refs) != self.contract["reference_count"]:
                    raise ValueError("Too few fixed evaluation references")
                allowed = set(info.get("reference_pool", info["splits"]["train"]))
                if len(set(refs)) != len(refs) or not set(refs) <= allowed:
                    raise ValueError("Evaluation references are outside the admitted training pool")
                for frame in info["splits"][split]:
                    if frame in refs:
                        raise ValueError("Evaluation target also appears as a reference")
                    yield dict(dataset=name, scene=scene, frame=frame, references=refs)


class LUNAComparison(nn.Module):
    def __init__(self, configuration, corpus, phase):
        super().__init__()
        cfg = yaml.safe_load(path(configuration["config"]).read_text())
        self.teacher = SMPLTeacher(
            path(cfg["smpl_model"]),
            cfg["num_queries"],
            seed=corpus.contract["seed"],
            pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
        ).cuda()
        self.identity = IdentityEncoder(
            self.teacher.anchors, self.teacher.semantic_labels, ModelConfig(**cfg["model"])
        ).cuda()
        self.features = {
            name: FeatureStore(
                str(path(entry["features"])),
                str(path(entry["manifest"])),
                ("body", "face") if phase == "reconstruction" else ("body", "face", "motion"),
            )
            for name, entry in corpus.configuration["datasets"].items()
        }
        if any(
            store.metadata["face"].get("face_backbone", "dinov2") != self.identity.face_encoder
            for store in self.features.values()
        ):
            raise ValueError("Cached face encoder differs from LUNA architecture")
        self.phase = phase
        if phase == "animation":
            # Equal observation weight, retaining the original project's explicit
            # translation-statistics convention across the admitted corpus.
            values = []
            for data in corpus.datasets.values():
                mean, std = translation_statistics(self.teacher, data)
                values.append((mean, std, len(data)))
            total = sum(count for _, _, count in values)
            mean = sum(mean * count for mean, _, count in values) / total
            variance = (
                sum((std.square() + (m - mean).square()) * count for m, std, count in values)
                / total
            )
            self.animator = NeuralAnimator(
                cfg["num_queries"],
                ModelConfig(**cfg["model"]),
                mean,
                variance.sqrt().clamp_min(1e-4),
            ).cuda()
            self.identity.requires_grad_(False)
        self.provenance = dict(
            method="luna",
            phase=phase,
            architecture=cfg["model"],
            num_queries=cfg["num_queries"],
            query_sampling_seed=corpus.contract["seed"],
            smpl_asset_sha256=file_sha256(path(cfg["smpl_model"])),
            pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
            feature_metadata={name: store.metadata for name, store in self.features.items()},
            objective_deviation="LUNA supervision ablation: paper distillation/global losses omitted",
        )
        for data in corpus.datasets.values():
            if hasattr(data, "validate_smpl_asset"):
                data.validate_smpl_asset(
                    self.provenance["smpl_asset_sha256"], self.provenance["pose_blend_shapes"]
                )

    def forward(self, item):
        features = self.features[item["dataset"]]
        body, face = features.references(item)
        target = prepare_item(item)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.phase == "reconstruction"):
            canonical = self.identity(body, face)
        if self.phase == "reconstruction":
            gaussians = self.teacher(
                canonical.gaussians,
                target["pose"],
                target["betas"],
                target["body_to_camera"],
                detach=False,
            )
        else:
            motion = features.get("motion", item["scene"], item["frame"])[None]
            gaussians = self.animator(canonical, motion).gaussians
        return render(gaussians, target["K"], tuple(item["rgb"].shape[-2:]))


@torch.no_grad()
def evaluate(model, corpus, method, objective, lpips, split):
    model.eval()
    records = []
    for choice in corpus.evaluation_choices(split):
        item = corpus.item(choice)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(item) if method == "luna" else model(*corpus.native_inputs(item))
        scores = image_metrics(
            prediction, item["rgb"][None].cuda(), item["mask"][None].cuda(), lpips
        )
        records.append(
            dict(
                scene=choice["dataset"] + "/" + choice["scene"],
                frame=choice["frame"],
                metrics={key: float(value[0]) for key, value in scores.items()},
            )
        )
    if not records:
        raise ValueError("Requested evaluation split is empty")
    return aggregate_records(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lhm", "lhmpp", "luna"), required=True)
    parser.add_argument(
        "--phase", choices=("reconstruction", "animation"), default="reconstruction"
    )
    for name in ("contract", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reconstruction-checkpoint", type=Path)
    parser.add_argument("--evaluate", choices=("val", "test"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one allocated GPU")
    contract = load_contract(args.contract, args.method)
    configuration = yaml.safe_load(args.corpus.read_text())
    seed_all(contract["seed"])
    if torch.cuda.get_device_capability() == (10, 0) and args.method != "luna":
        from xformers.ops import fmha

        fmha._set_use_fa3(False)
    corpus = Corpus(configuration, contract, args.method)
    if args.method == "luna":
        model = LUNAComparison(configuration["models"]["luna"], corpus, args.phase)
    else:
        options = configuration["models"][args.method]
        model = NativeReconstruction(
            args.method,
            path(options["source"]),
            path(options["runtime"]),
            path(options["architecture"]),
        ).cuda()
    lpips = build_lpips()
    objective = ReleasedPhotometricObjective(
        path(configuration["training_source"]), contract["loss_weights"], lpips
    )
    provenance = dict(
        contract_sha256=digest_document(contract),
        corpus_sha256=corpus.sha256,
        method=args.method,
        phase=args.phase,
        model=model.provenance,
        objective=objective.provenance,
        runner_source_sha256={
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "comparison.py",
                "comparison_training.py",
                "native_baselines.py",
                "upstream_objective.py",
            )
        },
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, contract["weight_decay"]),
        lr=contract["learning_rate"],
        betas=tuple(contract["adam_betas"]),
    )
    start, best = 0, float("inf")
    if args.phase == "animation" and not args.resume:
        if args.reconstruction_checkpoint is None:
            raise ValueError("Animation phase needs the same method's reconstruction checkpoint")
        previous = torch.load(
            args.reconstruction_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            any(
                previous["provenance"][key] != provenance[key]
                for key in ("contract_sha256", "corpus_sha256", "method")
            )
            or previous["provenance"]["phase"] != "reconstruction"
        ):
            raise ValueError("Reconstruction transfer has a different contract, corpus or method")
        if args.method == "luna":
            prefix = "identity."
            model.identity.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in previous["model"].items()
                    if key.startswith(prefix)
                }
            )
        else:
            model.load_state_dict(previous["model"], strict=True)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["provenance"] != provenance:
            raise ValueError("Resume configuration, dataset, model or objective changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start, best = state["update"], state["best"]
    if args.evaluate:
        if not args.resume:
            raise ValueError("Evaluation requires a trained comparison checkpoint")
        args.output.mkdir(parents=True, exist_ok=True)
        result = evaluate(model, corpus, args.method, objective, lpips, args.evaluate)
        result["provenance"] = provenance
        result["split"] = args.evaluate
        result["checkpoint_sha256"] = file_sha256(args.resume)
        result["environment"] = runtime_environment()
        (args.output / f"{args.evaluate}-metrics.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError("Choose a fresh run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps(
            dict(contract=contract, provenance=provenance, environment=runtime_environment()),
            indent=2,
        )
        + "\n"
    )
    for update in range(start, contract["updates"]):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rate = contract["learning_rate"] * learning_rate_factor(update, contract)
        for group in optimizer.param_groups:
            group["lr"] = rate
        if args.method != "luna":
            model.set_update(update)
        totals, samples = {}, []
        for microstep in range(contract["effective_batch"]):
            choice = corpus.sampler.sample(update, microstep)
            samples.append(choice)
            item = corpus.item(choice)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = (
                    model(item) if args.method == "luna" else model(*corpus.native_inputs(item))
                )
                loss, components = objective(
                    prediction, item["rgb"][None].cuda(), item["mask"][None].cuda()
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite shared training objective")
            (loss / contract["effective_batch"]).backward()
            for key, value in components.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach()) / contract["effective_batch"]
                )
        norm = nn.utils.clip_grad_norm_(
            model.parameters(), contract["gradient_clip"], error_if_nonfinite=True
        )
        if norm <= 0:
            raise RuntimeError("No gradient reached the trainable model")
        optimizer.step()
        torch.cuda.synchronize()
        completed = update + 1
        record = dict(
            update=completed,
            learning_rate=rate,
            training_seconds=time.perf_counter() - started,
            gradient_norm=float(norm),
            samples=samples,
            **totals,
        )
        improved = False
        if completed % contract["validate_every"] == 0 or completed == contract["updates"]:
            result = evaluate(model, corpus, args.method, objective, lpips, "val")
            score = result["mean_over_scenes"]["lpips"]
            improved, best = score < best, min(score, best)
            record["validation_lpips"] = score
            (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        with (args.output / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if (
            completed % contract["checkpoint_every"] == 0
            or completed == contract["updates"]
            or improved
        ):
            state = dict(
                update=completed,
                best=best,
                provenance=provenance,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
            )
            atomic_checkpoint(args.output / "latest.pt", state)
            if improved:
                atomic_checkpoint(args.output / "best.pt", state)
        print(
            json.dumps({key: value for key, value in record.items() if key != "samples"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
