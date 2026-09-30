"""Audit native body buffers and full dense-point skinning against independent FK.

The published checkpoint can overwrite body buffers constructed from assets.
Check these explicitly before trusting a joint-only comparison. No RGB fitting.
Native source is imported read-only; independent FK below uses SciPy rotations
and homogeneous transforms, not the native or smplx batch_rigid_transform.
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import smplx
import torch
from safetensors import safe_open
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)), max=float(x.max()))


def independent_fk(model, motion):
    beta = motion["betas"].cpu().numpy()[0].astype(np.float64)
    shape = model.v_template.cpu().numpy().astype(np.float64) + np.einsum(
        "vci,i->vc", model.shapedirs.cpu().numpy().astype(np.float64), beta
    )
    joints = model.J_regressor.cpu().numpy().astype(np.float64) @ shape
    aa = np.concatenate([motion[k].cpu().numpy().reshape(-1, 3) for k in
                         ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose")])
    rotations = Rotation.from_rotvec(aa).as_matrix()
    global_mats = np.tile(np.eye(4), (55, 1, 1))
    parents = model.parents.cpu().numpy()
    for j, parent in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = rotations[j]
        local[:3, 3] = joints[j] if j == 0 else joints[j] - joints[parent]
        global_mats[j] = local if j == 0 else global_mats[parent] @ local
    result = global_mats.copy()
    result[:, :3, 3] -= np.einsum("nij,nj->ni", global_mats[:, :3, :3], joints)
    return torch.tensor(result, device="cuda", dtype=torch.float32), joints, shape


def pose(fit=None):
    sizes = dict(root_pose=(3,), body_pose=(21, 3), jaw_pose=(3,), leye_pose=(3,), reye_pose=(3,),
                 lhand_pose=(15, 3), rhand_pose=(15, 3), expr=(100,), trans=(3,), betas=(10,))
    return {k: torch.tensor(fit[k], device="cuda", dtype=torch.float32)[None]
            if fit is not None and k in fit else torch.zeros(1, *v, device="cuda")
            for k, v in sizes.items()}


def project(x, annotation):
    a = np.asarray(annotation["body_to_camera"])
    k = np.asarray(annotation["K"])
    p = (x @ a[:3, :3].T + a[:3, 3]) @ k.T
    return p[:, :2] / p[:, 2:]


@torch.no_grad()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute allocation required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lhm", "lhmpp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name()
    source = WORK / f"baselines/{args.method}-20260928/source"
    checkpoint = WORK / "assets" / ("lhm_500m" if args.method == "lhm" else "lhmpp_700m")
    cfg = json.loads((checkpoint / "config.json").read_text())
    os.chdir(source)
    sys.path.insert(0, str(source))
    # BasicSR imported by the pinned LHM package uses the pre-0.17 alias.
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from accelerate import Accelerator
    Accelerator()
    if args.method == "lhm":
        from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel as Native
    else:
        from core.models.rendering.skinnings.smplx_diffused_voxel_skinning import SMPLXDiffusedVoxelSkinning as Native
    native = Native(cfg["human_model_path"], "neutral", cfg["smplx_subdivide_num"],
                    expr_param_dim=cfg["expr_param_dim"], shape_param_dim=cfg["shape_param_dim"],
                    cano_pose_type=cfg["cano_pose_type"], dense_sample_points=cfg["dense_sample_pts"],
                    apply_pose_blendshape=False).cuda().eval()
    record = dict(method=args.method, host=socket.gethostname(), job=os.environ["SLURM_JOB_ID"],
                  gpu=torch.cuda.get_device_name(), buffers={}, assets={}, frames=[])
    state = native.state_dict()
    prefix = "renderer.smplx_model."
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            old = state[name]
            new = f.get_tensor(key)
            if old.shape != new.shape:
                raise RuntimeError(f"Buffer shape mismatch: {name}")
            a, b = old.reshape(-1), new.reshape(-1)
            maximum, different = 0., 0
            for start in range(0, a.numel(), 2**20):
                delta = (a[start:start+2**20].double() - b[start:start+2**20].cuda().double()).abs()
                maximum = max(maximum, float(delta.max()))
                different += int((delta > 1e-6).sum())
            record["buffers"][name] = dict(max_abs=maximum, different_over_1e_minus6=different, count=a.numel())
            old.copy_(new)
            print(f"{name}: max={maximum:.6g} changed={different}/{a.numel()}", flush=True)
            (out / "audit.json").write_text(json.dumps(record, indent=2))
    fits = json.loads((BASE / "fit-test/fits.json").read_text())
    protocol = json.loads((BASE / "protocol-test/protocol.json").read_text())
    standard = smplx.SMPLX(fits["files"]["smplx"]["path"], num_betas=10, use_pca=False,
                          flat_hand_mean=True, num_expression_coeffs=10).cuda().eval()
    for k in ("v_template", "shapedirs", "J_regressor", "lbs_weights", "parents", "faces_tensor", "pose_mean", "posedirs"):
        a, b = getattr(native.smplx_layer, k), getattr(standard, k)
        record["assets"][k] = dict(shape=list(a.shape), max_abs=float((a.double()-b.double()).abs().max()))
    dense = native.dense_pts.cuda()
    nearest = cKDTree(standard.v_template.cpu().numpy()).query(dense.cpu().numpy())[1]
    nearest = torch.tensor(nearest, device="cuda")
    record["dense_lookup"] = dict(
        skinning_max_abs=float((native.skinning_weight-standard.lbs_weights[nearest]).abs().max()),
        shape_dirs_max_abs=float((native.shape_dirs-standard.shapedirs[nearest]).abs().max()),
        nearest_vertex_distance_mm=stats((dense-standard.v_template[nearest]).norm(dim=-1).cpu().numpy()*1000),
        weight_sum_error=float((native.skinning_weight.sum(-1)-1).abs().max()),
    )
    if args.method == "lhm":
        query, _, bind = native.get_query_points(pose(), torch.device("cuda"))
    else:
        points = native.get_query_points(True, pose(), torch.device("cuda"))
        query, bind = points["neutral_coords"], points["transform_mat_to_null_pose"]
    mask = (native.is_rhand | native.is_lhand | native.is_face)[None]
    weights = native.skinning_weight[None] if args.method == "lhm" else native.query_voxel_skinning_weights(query)
    weights[:, mask[0]] = native.skinning_weight[mask[0]]
    record["effective_weight_sum_error"] = float((weights.sum(-1)-1).abs().max())

    def native_forward(motion):
        if args.method == "lhm":
            return native.transform_to_posed_verts_from_neutral_pose(query, motion, query, bind, torch.device("cuda"))[0][0]
        p = dict(points)
        p["neutral_coords"] = query.clone()
        return native.transform_to_posed_verts_from_neutral_pose(p, motion, torch.device("cuda"))["posed_coords"][0]

    zero = native_forward(pose())
    record["canonical_roundtrip_mm"] = stats((zero-dense).norm(dim=-1).cpu().numpy()*1000)
    # Full chain prediction from native zero-pose points and independent FK.
    # This isolates transform order/translation from native inverse-bind approximation.
    cases = [(s, n, fits["scenes"][s][n]) for s, info in protocol["scenes"].items()
             for n in [info["references"][0]] + info["targets"]]
    for scene, name, fitted in cases:
        motion = pose(fitted)
        actual = native_forward(motion)
        A, _, _ = independent_fk(standard, motion)
        transforms = (weights[0] @ A.reshape(55, 16)).reshape(-1, 4, 4)
        shape_delta = (native.shape_dirs * motion["betas"][:, None, None]).sum(-1)[0]
        predicted = (transforms[:, :3, :3] @ (zero+shape_delta)[..., None]).squeeze(-1) + transforms[:, :3, 3] + motion["trans"]
        direct = (transforms[:, :3, :3] @ (dense+shape_delta)[..., None]).squeeze(-1) + transforms[:, :3, 3] + motion["trans"]
        actual_np, direct_np = actual.cpu().numpy(), direct.cpu().numpy()
        ann = protocol["scenes"][scene]["annotations"][name]
        row = dict(scene=scene, frame=name,
                   independent_full_chain_mm=stats((actual-predicted).norm(dim=-1).cpu().numpy()*1000),
                   direct_dense_lbs_mm=stats((actual-direct).norm(dim=-1).cpu().numpy()*1000),
                   direct_dense_lbs_pixels=stats(np.linalg.norm(project(actual_np, ann)-project(direct_np, ann), axis=-1)))
        record["frames"].append(row)
        print(scene, name, row["independent_full_chain_mm"], flush=True)
    record["complete"] = True
    (out / "audit.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ("frames","buffers")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
