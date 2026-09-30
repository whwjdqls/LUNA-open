"""Fit SMPL-X to NeuMan's SMPL meshes using official vertex correspondences.

Independent implementation of the correspondence/edge/vertex fitting procedure
described in vchoutas/smplx transfer_model/docs/transfer.md (also vendored in
LHM @ 4f88aaeb). Uses PyTorch LBFGS rather than its trust-region optimizer.
Pose blend shapes are disabled to match our NeuMan SMPL teacher and native LHM
animation. No held-out RGB, masks, or reconstruction losses enter this fitting.
Body-model and correspondence assets remain subject to their own licenses.
"""

import argparse
import hashlib
import json
import os
import pickle
import socket
import time
from pathlib import Path

import numpy as np
import smplx
import torch
from PIL import Image, ImageDraw
from scipy import sparse


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_transfer(path):
    before = path.stat()
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Correspondence file is still being uploaded")
    if "mtx" in data:
        matrix = sparse.csr_matrix(data["mtx"])
        matrix = matrix[:, : matrix.shape[1] // 2]  # upstream's use_normals=False convention
    elif "matrix" in data:
        matrix = data["matrix"]
    else:
        raise ValueError(f"Unknown transfer format: {list(data)}")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape != (10475, 6890):
        raise ValueError(f"Need SMPL->SMPL-X (10475,6890), got {matrix.shape}")
    return matrix


def project(vertices, body_to_camera, K):
    camera = vertices @ body_to_camera[:3, :3].T + body_to_camera[:3, 3]
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:].clamp_min(1e-6)


def fit_frame(model, target, initial_pose, valid, edges, edge_steps, vertex_steps):
    device = target.device
    root = torch.nn.Parameter(initial_pose[:3][None].clone())
    body = torch.nn.Parameter(initial_pose[3:66][None].clone())
    left = torch.nn.Parameter(torch.zeros(1, 45, device=device))
    right = torch.nn.Parameter(torch.zeros(1, 45, device=device))
    betas = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    trans = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    variables = [root, body, left, right, betas, trans]

    def vertices():
        return model(
            global_orient=root,
            body_pose=body,
            left_hand_pose=left,
            right_hand_pose=right,
            betas=betas,
            transl=trans,
        ).vertices[0]

    with torch.no_grad():
        initial = vertices()
        trans.copy_((target[valid] - initial[valid]).mean(0)[None])
        initial_error = float((vertices()[valid] - target[valid]).norm(dim=-1).mean())
    target_edges = target[edges[:, 0]] - target[edges[:, 1]]
    for stage, steps in (("edges", edge_steps), ("vertices", vertex_steps)):
        optimizer = torch.optim.LBFGS(
            variables,
            lr=1.0,
            max_iter=steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            pred = vertices()
            if stage == "edges":
                error = (pred[edges[:, 0]] - pred[edges[:, 1]] - target_edges).square().mean()
            else:
                error = (pred[valid] - target[valid]).square().mean()
            # Weak regularizers stabilize degrees of freedom poorly represented
            # in SMPL; these coefficients are project fitting choices.
            loss = (
                error
                + 1e-8 * betas.square().mean()
                + 1e-8 * (left.square().mean() + right.square().mean())
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        if stage == "edges":
            with torch.no_grad():
                trans.add_((target[valid] - vertices()[valid]).mean(0)[None])
    with torch.no_grad():
        fitted = vertices().detach()
        error = (fitted[valid] - target[valid]).norm(dim=-1)
        params = dict(
            root_pose=root[0].tolist(),
            body_pose=body[0].view(21, 3).tolist(),
            lhand_pose=left[0].view(15, 3).tolist(),
            rhand_pose=right[0].view(15, 3).tolist(),
            betas=betas[0].tolist(),
            trans=trans[0].tolist(),
        )
        quality = dict(
            initial_mean_mm=1000 * initial_error,
            mean_mm=float(error.mean()) * 1000,
            p95_mm=float(error.quantile(0.95)) * 1000,
            max_mm=float(error.max()) * 1000,
        )
    return params, fitted, quality


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smpl", type=Path, required=True)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--edge-steps", type=int, default=100)
    parser.add_argument("--vertex-steps", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    device = "cuda"
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    transfer = load_transfer(args.transfer)
    mask_ids = np.load(args.mask, allow_pickle=False)
    if mask_ids.dtype == bool:
        mask_ids = np.flatnonzero(mask_ids)
    if mask_ids.ndim != 1 or mask_ids.min() < 0 or mask_ids.max() >= 10475:
        raise ValueError("Unexpected SMPL-X correspondence mask")
    valid = torch.tensor(mask_ids, dtype=torch.long, device=device)
    source = smplx.SMPL(str(args.smpl), num_betas=10).to(device).requires_grad_(False)
    target_model = (
        smplx.SMPLX(
            str(args.smplx),
            gender="neutral",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
            use_face_contour=False,
            num_expression_coeffs=10,
        )
        .to(device)
        .requires_grad_(False)
    )
    source.posedirs.zero_()
    target_model.posedirs.zero_()
    faces = target_model.faces_tensor
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = torch.unique(edges.sort(dim=1).values, dim=0)
    keep = torch.zeros(10475, dtype=torch.bool, device=device)
    keep[valid] = True
    edges = edges[keep[edges].all(dim=1)]
    args.output.mkdir(parents=True)
    result = dict(
        manifest_sha256=protocol["manifest_sha256"],
        split=protocol["split"],
        scenes={},
        method="official SMPL-to-SMPL-X correspondences; edge then vertex LBFGS fit; pose blend shapes disabled",
        thresholds=dict(mean_mm=10, p95_mm=25, mean_pixels=3, p95_pixels=8),
        files={
            key: dict(path=str(getattr(args, key)), sha256=sha256(getattr(args, key)))
            for key in ("smpl", "smplx", "transfer", "mask")
        },
        settings=dict(
            edge_steps=args.edge_steps,
            vertex_steps=args.vertex_steps,
            shape_penalty=1e-8,
            hand_penalty=1e-8,
        ),
        annotation_inputs="SMPL pose/shape meshes only; no RGB/mask optimization",
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        frames=[],
    )
    started = time.perf_counter()
    for scene, info in protocol["scenes"].items():
        if args.scene and scene != args.scene:
            continue
        result["scenes"][scene] = {}
        folder = args.output / scene
        folder.mkdir()
        # Fit the first training reference plus held-out targets. Other refs are
        # irrelevant for the released one-image LHM model.
        names = [info["references"][0]] + info["targets"]
        if args.limit:
            names = names[: args.limit]
        for name in names:
            annotation = info["annotations"][name]
            pose = torch.tensor(annotation["pose"], device=device).float().flatten()
            betas = torch.tensor(annotation["betas"], device=device).float().reshape(1, 10)
            with torch.no_grad():
                mesh = source(
                    global_orient=pose[:3][None], body_pose=pose[3:][None], betas=betas
                ).vertices[0]
                corresponding = torch.from_numpy(transfer @ mesh.cpu().numpy()).to(device)
            params, fitted, quality = fit_frame(
                target_model, corresponding, pose, valid, edges, args.edge_steps, args.vertex_steps
            )
            with torch.no_grad():
                body_to_camera = torch.tensor(annotation["body_to_camera"], device=device)
                K = torch.tensor(annotation["K"], device=device)
                target_pixels = project(corresponding, body_to_camera, K)
                fitted_pixels = project(fitted, body_to_camera, K)
                pixel_error = (fitted_pixels[valid] - target_pixels[valid]).norm(dim=-1)
                quality.update(
                    mean_pixels=float(pixel_error.mean()),
                    p95_pixels=float(pixel_error.quantile(0.95)),
                )
            passed = all(quality[key] <= value for key, value in result["thresholds"].items())
            params["quality"] = dict(**quality, passed=passed)
            result["scenes"][scene][name] = params
            result["frames"].append(dict(scene=scene, frame=name, **quality, passed=passed))
            image = Image.open(args.protocol / scene / "rgb" / name).convert("RGB")
            draw = ImageDraw.Draw(image)
            for pixels, color in (
                (target_pixels[valid][::8], "#00cc44"),
                (fitted_pixels[valid][::8], "#ff5500"),
            ):
                for x, y in pixels.cpu().tolist():
                    if 0 <= x < 512 and 0 <= y < 512:
                        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            image.save(folder / name, format="PNG")
            (args.output / "fits.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["frames"][-1]), flush=True)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["all_passed"] = all(row["passed"] for row in result["frames"])
    (args.output / "fits.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_passed"]:
        raise RuntimeError(
            "Some conversion fits exceed the predefined quality thresholds; inspect before scoring"
        )


if __name__ == "__main__":
    main()
