import json
from pathlib import Path

import pytest
import torch
import yaml

from luna_open.comparison import (
    SharedSampler,
    learning_rate_factor,
    optimizer_groups,
    validate_contract,
    validate_motion_receipt,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def contract():
    return yaml.safe_load((REPO / "configs/comparison.yaml").read_text())


def manifest():
    return dict(
        schema_version=2,
        source_sha256={"file": "fixture"},
        scenes={
            "scene": dict(
                actor_id="actor",
                frames=[dict(name=f"{i}.png") for i in range(7)],
                splits=dict(train=[f"{i}.png" for i in range(5)], val=["5.png"], test=["6.png"]),
            )
        },
    )


def test_shared_samples_survive_model_rng_consumption_and_resume(contract):
    source = {"first": manifest(), "second": manifest()}
    sampler = SharedSampler(contract, source, {"first": 1.0, "second": 3.0})
    stream = [sampler.sample(update, micro) for update in range(10) for micro in range(16)]
    torch.rand(2000)  # a different architecture can consume a different amount of RNG
    restored = SharedSampler(
        contract, json.loads(json.dumps(source)), {"second": 3.0, "first": 1.0}
    )
    assert stream[5 * 16 :] == [
        restored.sample(update, micro) for update in range(5, 10) for micro in range(16)
    ]
    for sample in stream:
        assert sample["frame"] not in sample["references"]
        assert "5.png" not in sample["references"] and "6.png" not in sample["references"]
        assert len(sample["references"]) == 1
    changed = SharedSampler(contract, source, {"first": 3.0, "second": 1.0})
    assert changed.membership_sha256 != restored.membership_sha256


def test_reject_reference_leakage_and_unequal_native_information(contract):
    bad = manifest()
    bad["scenes"]["scene"]["reference_pool"] = ["0.png", "5.png"]
    with pytest.raises(ValueError, match="leakage"):
        SharedSampler(contract, {"data": bad}, {"data": 1.0})
    for method in ("lhm", "lhmpp", "luna"):
        validate_contract(contract, method)
    with pytest.raises(ValueError, match="MV-LHM"):
        validate_contract({**contract, "reference_count": 4}, "lhm")
    reserved = manifest()
    reserved["scenes"]["scene"]["splits"]["train"] = ["4.png"]
    reserved["scenes"]["scene"]["reference_pool"] = [f"{i}.png" for i in range(4)]
    sampler = SharedSampler({**contract, "reference_count": 4}, {"dna": reserved}, {"dna": 1.0})
    assert len(set(sampler.sample(0, 0)["references"])) == 4


def test_lr_budget_and_identical_decay_policy(contract):
    assert learning_rate_factor(0, contract) == 0.001
    assert learning_rate_factor(999, contract) == 1
    assert learning_rate_factor(1000, contract) == 1
    assert learning_rate_factor(29999, contract) == 0
    with pytest.raises(ValueError, match="budget"):
        learning_rate_factor(30000, contract)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    model[0].bias.requires_grad_(False)
    decay, no_decay = optimizer_groups(model, 0.1)
    assert {id(p) for p in decay["params"]} == {id(model[0].weight)}
    assert {id(p) for p in no_decay["params"]} == {id(model[1].weight), id(model[1].bias)}
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0


def test_released_training_objective_reaches_rgb_and_alpha_gradients():
    from luna_open.upstream_objective import ReleasedPhotometricObjective

    source = Path("/vast/projects/lingjie6/impossible/jungbinc/references/LHM-plusplus")
    if not source.exists():
        pytest.skip("Pinned external LHM++ source unavailable")

    class PerceptualFixture(torch.nn.Module):
        def forward(self, x, y):
            return (x - y).square().mean((1, 2, 3), keepdim=True)

    objective = ReleasedPhotometricObjective(
        source, dict(rgb=1.0, mask=2.0, lpips=0.5), PerceptualFixture()
    )
    rgb = torch.full((1, 3, 4, 4), 0.25, requires_grad=True)
    alpha = torch.full((1, 1, 4, 4), 0.4, requires_grad=True)
    loss, parts = objective(
        dict(rgb=rgb, alpha=alpha), torch.zeros_like(rgb), torch.ones_like(alpha)
    )
    torch.testing.assert_close(loss, torch.tensor(1.575))
    loss.backward()
    assert rgb.grad.abs().sum() > 0 and alpha.grad.abs().sum() > 0
    assert set(parts) == {"rgb", "mask", "lpips"}
    assert objective.handle.model is None


def test_admission_requires_training_fits_and_numeric_evidence():
    document = manifest()
    thresholds = dict(mean_mm=10, p95_mm=25, mean_pixels=3, p95_pixels=8)
    fits = dict(
        manifest_sha256="same-manifest",
        all_passed=True,
        thresholds=thresholds,
        scenes={
            "scene": {
                f"{i}.png": dict(quality=dict(passed=True, **{key: 1.0 for key in thresholds}))
                for i in range(7)
            }
        },
    )
    validate_motion_receipt(document, "same-manifest", fits)
    # Evaluation-only conversions cannot authorize full-corpus training.
    missing = json.loads(json.dumps(fits))
    del missing["scenes"]["scene"]["0.png"]
    with pytest.raises(ValueError, match="fit missing"):
        validate_motion_receipt(document, "same-manifest", missing)
    fits["scenes"]["scene"]["0.png"]["quality"]["p95_pixels"] = 9.0
    with pytest.raises(ValueError, match="numerical validation"):
        validate_motion_receipt(document, "same-manifest", fits)
    with pytest.raises(ValueError, match="manifest"):
        validate_motion_receipt(document, "another-manifest", fits)


def test_result_comparator_rejects_unequal_protocol_and_targets(tmp_path):
    import runpy

    compare = runpy.run_path(str(REPO / "scripts/compare_controlled_runs.py"))["compare"]
    metrics = dict(psnr=20.0, l1=0.1, ssim=0.8, mask_iou=0.9, lpips=0.1)
    paths = [tmp_path / f"{method}.json" for method in ("lhm", "lhmpp", "luna")]
    for method, target in zip(("lhm", "lhmpp", "luna"), paths, strict=True):
        target.write_text(
            json.dumps(
                dict(
                    split="test",
                    provenance=dict(
                        method=method,
                        phase="reconstruction",
                        contract_sha256="shared",
                        corpus_sha256="shared",
                        objective={"sha": "shared"},
                        runner_source_sha256={"sha": "shared"},
                    ),
                    mean_over_scenes=metrics,
                    frames=[dict(scene="scene", frame="6.png", metrics=metrics)],
                )
            )
        )
    assert compare(paths)["frames"] == 1
    changed = json.loads(paths[2].read_text())
    changed["provenance"]["contract_sha256"] = "other"
    paths[2].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="contract_sha256"):
        compare(paths)
    changed["provenance"]["contract_sha256"] = "shared"
    changed["frames"][0]["frame"] = "5.png"
    paths[2].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="targets"):
        compare(paths)
