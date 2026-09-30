"""Shared sampling and optimization contract for controlled model comparisons.

The contract contains no server paths or model architecture. Dataset roots and
native body fits are supplied separately. A sample is determined by its update
and accumulation index, so all methods consume the same observations and resume
without relying on model-specific RNG consumption.
"""

import hashlib
import json
import math
import random
from pathlib import Path

import torch


def digest_document(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_motion_receipt(manifest: dict, manifest_sha256: str, fits: dict) -> None:
    """Require complete, numerically accepted body fits for the exact corpus."""
    if not fits.get("all_passed") or fits.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Native motion receipt does not validate this manifest")
    thresholds = fits.get("thresholds", {})
    if not {"mean_mm", "p95_mm", "mean_pixels", "p95_pixels"} <= thresholds.keys() or any(
        not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0
        for limit in thresholds.values()
    ):
        raise ValueError("Native motion needs finite positive 3D and projection thresholds")
    for scene, info in manifest["scenes"].items():
        selected = set().union(*map(set, info["splits"].values()))
        for frame in sorted(selected):
            row = fits.get("scenes", {}).get(scene, {}).get(frame)
            if row is None:
                raise ValueError(f"Selected native body fit missing: {scene}/{frame}")
            quality = row.get("quality", {})
            if not quality.get("passed") or any(
                not isinstance(quality.get(key), (int, float))
                or not math.isfinite(quality[key])
                or not 0 <= quality[key] <= limit
                for key, limit in thresholds.items()
            ):
                raise ValueError(
                    f"Native body fit lacks passing numerical validation: {scene}/{frame}"
                )


def validate_contract(contract: dict, method: str) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported comparison contract")
    if not isinstance(contract["seed"], int) or contract["seed"] < 0:
        raise ValueError("Invalid sampling seed")
    if method not in {"lhm", "lhmpp", "luna"}:
        raise ValueError("Unknown comparison method")
    if contract["reference_count"] not in {1, 4}:
        raise ValueError("Reference count must be one or four")
    if method == "lhm" and contract["reference_count"] != 1:
        raise ValueError(
            "Native LHM consumes one reference; four-view LHM requires a distinct MV-LHM model"
        )
    if contract["initialization"] != "fresh_reconstruction":
        raise ValueError("Controlled retraining requires fresh reconstruction weights")
    if contract["objective"] != "shared_photometric":
        raise ValueError("The controlled comparison uses the shared photometric objective")
    for key in ("updates", "effective_batch", "image_size", "validate_every", "checkpoint_every"):
        if not isinstance(contract[key], int) or contract[key] < 1:
            raise ValueError(f"Invalid comparison setting: {key}")
    if not 0 <= contract["warmup_updates"] < contract["updates"]:
        raise ValueError("Warmup must be shorter than training")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(contract[key]) or contract[key] <= 0:
            raise ValueError(f"Invalid comparison setting: {key}")
    if not math.isfinite(contract["weight_decay"]) or contract["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if len(contract["adam_betas"]) != 2 or any(not 0 <= x < 1 for x in contract["adam_betas"]):
        raise ValueError("Invalid Adam betas")
    if contract["precision"] != "bfloat16" or contract["background"] != "white":
        raise ValueError("Unsupported precision or background")
    if contract["loss_weights"].keys() != {"rgb", "mask", "lpips"} or any(
        not math.isfinite(x) or x <= 0 for x in contract["loss_weights"].values()
    ):
        raise ValueError("All three shared photometric losses require positive weights")


def learning_rate_factor(update: int, contract: dict) -> float:
    """LR for the update about to run (zero based), including the first warmup step."""
    warmup, total = contract["warmup_updates"], contract["updates"]
    if not 0 <= update < total:
        raise ValueError("Update outside the agreed training budget")
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup - 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_groups(module, weight_decay):
    """Identical decay policy: omit biases, one-dimensional parameters and LayerNorm."""
    no_decay_ids = {
        id(parameter)
        for layer in module.modules()
        if isinstance(layer, torch.nn.LayerNorm)
        for parameter in layer.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            group = (
                no_decay
                if parameter.ndim <= 1 or name.endswith(".bias") or id(parameter) in no_decay_ids
                else decay
            )
            group.append(parameter)
    if not decay and not no_decay:
        raise ValueError("No trainable model parameters")
    return [
        dict(params=decay, weight_decay=weight_decay),
        dict(params=no_decay, weight_decay=0.0),
    ]


class SharedSampler:
    """Explicit corpus membership; dataset mixture, then uniform actor and frame."""

    def __init__(self, contract, manifests: dict[str, dict], weights: dict[str, float]):
        self.contract = contract
        if set(manifests) != set(weights) or not manifests:
            raise ValueError("Dataset membership and mixture weights differ")
        if any(not math.isfinite(x) or x <= 0 for x in weights.values()):
            raise ValueError("Dataset mixture weights must be finite and positive")
        self.names = sorted(manifests)
        self.weights = [weights[name] for name in self.names]
        self.frames, self.pools, self.actors = {}, {}, {}
        for name in self.names:
            document = manifests[name]
            if document.get("schema_version") != 2 or not document.get("source_sha256"):
                raise ValueError("Every corpus member needs a fingerprinted version-2 manifest")
            groups = {}
            for scene, info in document["scenes"].items():
                known = {row["name"] for row in info["frames"]}
                train = list(info["splits"]["train"])
                val, test = info["splits"]["val"], info["splits"]["test"]
                held_out = set(val) | set(test)
                pool = sorted(info.get("reference_pool", train))
                if (
                    not set(train) <= known
                    or not held_out <= known
                    or set(val) & set(test)
                    or len(val) != len(set(val))
                    or len(test) != len(set(test))
                    or not set(pool) <= known
                    or (set(train) | set(pool)) & held_out
                    or len(train) != len(set(train))
                    or len(pool) != len(set(pool))
                ):
                    raise ValueError(
                        f"Training/reference leakage or duplicate membership: {name}/{scene}"
                    )
                if not train:
                    continue
                if any(len(set(pool) - {target}) < contract["reference_count"] for target in train):
                    raise ValueError(
                        f"Too few distinct references excluding the target: {name}/{scene}"
                    )
                key = name, scene
                self.frames[key], self.pools[key] = sorted(train), pool
                actor = info.get("actor_id", scene)
                groups.setdefault(actor, []).append(scene)
            if not groups:
                raise ValueError(f"Dataset has no training actors: {name}")
            self.actors[name] = {actor: sorted(scenes) for actor, scenes in sorted(groups.items())}
        self.membership_sha256 = digest_document(dict(manifests=manifests, weights=weights))

    def sample(self, update: int, microstep: int) -> dict:
        if (
            not 0 <= update < self.contract["updates"]
            or not 0 <= microstep < self.contract["effective_batch"]
        ):
            raise ValueError("Sample index outside agreed update/batch budget")
        seed = digest_document(dict(seed=self.contract["seed"], update=update, microstep=microstep))
        rng = random.Random(int(seed, 16))
        dataset = rng.choices(self.names, weights=self.weights, k=1)[0]
        actor = rng.choice(list(self.actors[dataset]))
        scene = rng.choice(self.actors[dataset][actor])
        target = rng.choice(self.frames[dataset, scene])
        references = rng.sample(
            [name for name in self.pools[dataset, scene] if name != target],
            self.contract["reference_count"],
        )
        return dict(dataset=dataset, actor=actor, scene=scene, frame=target, references=references)


def load_contract(path: Path, method: str) -> dict:
    import yaml

    contract = yaml.safe_load(path.read_text())
    validate_contract(contract, method)
    return contract
