import random

import numpy as np
import torch

from luna_open.metrics import aggregate_records, image_metrics
from luna_open.training import atomic_checkpoint, restore_rng, rng_state


def test_metrics_identity_and_scene_weighting():
    rgb = torch.rand(2, 3, 24, 24)
    mask = torch.ones(2, 1, 24, 24)
    result = image_metrics(dict(rgb=rgb, alpha=mask), rgb, mask)
    torch.testing.assert_close(result["ssim"], torch.ones(2))
    assert result["mask_iou"].min() == 1
    rows = [
        dict(scene="a", metrics={"score": 1}),
        dict(scene="a", metrics={"score": 1}),
        dict(scene="b", metrics={"score": 4}),
    ]
    assert aggregate_records(rows)["mean_over_scenes"]["score"] == 2.5


def test_checkpoint_rng_and_optimizer_round_trip(tmp_path):
    torch.manual_seed(99)
    param = torch.nn.Parameter(torch.randn(3))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.8)

    def step(p, opt, sched):
        opt.zero_grad()
        (p - torch.randn(3)).square().sum().backward()
        opt.step()
        sched.step()

    step(param, optimizer, scheduler)
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(
        path,
        dict(
            param=param,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=rng_state(),
        ),
    )
    py_expected, np_expected = random.random(), np.random.rand()
    step(param, optimizer, scheduler)
    target = param.detach().clone()
    state = torch.load(path, weights_only=False)
    other = torch.nn.Parameter(state["param"].detach().clone())
    opt = torch.optim.AdamW([other], lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, 0.8)
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    assert random.random() == py_expected and np.random.rand() == np_expected
    step(other, opt, sched)
    torch.testing.assert_close(other, target, atol=0, rtol=0)
    assert sched.get_last_lr() == scheduler.get_last_lr()
