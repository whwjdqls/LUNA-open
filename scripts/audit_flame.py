"""Check converted FLAME arrays, native LHM consumers and standard CPU geometry.

Run with the isolated baseline Python. Original files are read by our narrow
compatibility reader; native constructors read the converted files normally.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import smplx as standard_smplx
import torch
from smplx.lbs import lbs

from luna_open.body_assets import array_metadata, read_numeric_flame, read_numeric_smpl
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--lhm-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-source", type=Path)
    parser.add_argument("--smpl-numeric", type=Path)
    args = parser.parse_args()
    if bool(args.smpl_source) != bool(args.smpl_numeric):
        parser.error("Provide both SMPL paths or neither")
    if args.output.exists():
        raise FileExistsError("Preserve the existing audit; choose a new output path")
    sys.path.insert(0, str(args.lhm_reference.resolve()))
    from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLX_Mesh
    from LHM.models.rendering.smplx import smplx as native_smplx

    torch.manual_seed(7)
    original_root = args.assets / "lhm_prior_official/pretrained_models/human_model_files/flame"
    numeric_root = args.assets / "flame_numeric"
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        numpy=np.__version__,
        torch=torch.__version__,
        models={},
        limitation=(
            "Native constructors/expression consumers and standard FLAME forward only. "
            "Vendored standalone FLAME.forward has a documented LBS argument mismatch. "
            "Full LHM construction and GPU execution remain unverified."
        ),
    )
    for name in ("FLAME_NEUTRAL.pkl", "2019/generic_model.pkl"):
        source, _ = read_numeric_flame(original_root / name)
        converted_path = numeric_root / name
        with converted_path.open("rb") as stream:
            converted = pickle.load(stream, encoding="latin1")
        receipt = json.loads(converted_path.with_suffix(".receipt.json").read_text())
        assert file_sha256(original_root / name) == receipt["source_sha256"]
        assert file_sha256(converted_path) == receipt["output_sha256"]
        assert array_metadata(source) == array_metadata(converted) == receipt["arrays"]
        assert source["bs_type"] == converted["bs_type"]
        assert source["bs_style"] == converted["bs_style"]

        # Native construction requires its standard basename and landmarks.
        with tempfile.TemporaryDirectory(prefix="luna-flame-") as temporary:
            flame_dir = Path(temporary) / "flame"
            flame_dir.mkdir()
            (flame_dir / "FLAME_NEUTRAL.pkl").symlink_to(converted_path.resolve())
            for landmark in ("flame_static_embedding.pkl", "flame_dynamic_embedding.npy"):
                (flame_dir / landmark).symlink_to((original_root / landmark).resolve())
            native_model = native_smplx.create(
                temporary,
                "flame",
                gender="neutral",
                num_betas=10,
                num_expression_coeffs=100,
                use_face_contour=False,
            ).eval()
            model = standard_smplx.create(
                temporary,
                "flame",
                gender="neutral",
                num_betas=10,
                num_expression_coeffs=100,
                use_face_contour=False,
            ).eval()
            checked_buffers = (
                "v_template",
                "shapedirs",
                "expr_dirs",
                "posedirs",
                "J_regressor",
                "parents",
                "lbs_weights",
                "faces_tensor",
            )
            for key in checked_buffers:
                torch.testing.assert_close(
                    getattr(native_model, key), getattr(model, key), rtol=0, atol=0
                )
            if name == "FLAME_NEUTRAL.pkl":
                # These are the actual native baseline methods that consume
                # FLAME. They do not invoke the standalone FLAME forward.
                smplx_dir = Path(temporary) / "smplx"
                smplx_dir.mkdir()
                body_root = original_root.parent / "smplx"
                (smplx_dir / "SMPLX_NEUTRAL.npz").symlink_to(
                    (body_root / "SMPLX_NEUTRAL.npz").resolve()
                )
                (flame_dir / "2019").mkdir()
                (flame_dir / "2019/generic_model.pkl").symlink_to(
                    (numeric_root / "2019/generic_model.pkl").resolve()
                )
                face_ids = np.load(body_root / "SMPL-X__FLAME_vertex_ids.npy")
                consumer = SimpleNamespace(
                    human_model_path=temporary,
                    shape_param_dim=10,
                    expr_param_dim=100,
                    face_vertex_idx=face_ids,
                )
                body = native_smplx.create(
                    temporary,
                    "smplx",
                    gender="neutral",
                    num_betas=10,
                    num_expression_coeffs=100,
                    use_pca=False,
                    flat_hand_mean=True,
                    use_face_contour=True,
                )
                body = SMPLX_Mesh.get_expr_from_flame(consumer, body)
                torch.testing.assert_close(
                    body.expr_dirs[face_ids],
                    torch.from_numpy(source["shapedirs"][:, :, 300:400]).float(),
                    rtol=0,
                    atol=0,
                )
                native_ids = SMPLX_Mesh.get_expr_vertex_idx(consumer)
                old_flame, _ = read_numeric_flame(original_root / "2019/generic_model.pkl")
                has_expression = np.any(old_flame["shapedirs"][:, :, 300:400] != 0, axis=(1, 2))
                allowed_joint = ~np.isin(old_flame["weights"].argmax(1), [0, 3, 4])
                expected_ids = face_ids[np.where(has_expression & allowed_joint)[0]]
                np.testing.assert_array_equal(native_ids, expected_ids)
                report["native_expression_consumers"] = dict(
                    neutral_smplx_expression_transfer_exact=True,
                    expression_vertex_ids_exact=True,
                    expression_vertex_count=len(native_ids),
                )
        # One neutral sample and one nonzero shape/expression/pose/translation.
        betas = torch.cat((torch.zeros(1, 10), torch.randn(1, 10) * 0.1))
        expression = torch.cat((torch.zeros(1, 100), torch.randn(1, 100) * 0.05))
        poses = torch.cat((torch.zeros(1, 5, 3), torch.randn(1, 5, 3) * 0.1))
        translation = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 1.0]])
        with torch.no_grad():
            actual = model(
                betas=betas,
                expression=expression,
                global_orient=poses[:, 0],
                neck_pose=poses[:, 1],
                jaw_pose=poses[:, 2],
                leye_pose=poses[:, 3],
                reye_pose=poses[:, 4],
                transl=translation,
            ).vertices
            # Evaluate directly from original numeric arrays, independently of
            # the converted file and native constructor's registered buffers.
            parents = torch.from_numpy(source["kintree_table"][0].astype(np.int64))
            parents[0] = -1
            shapedirs = np.concatenate(
                (source["shapedirs"][:, :, :10], source["shapedirs"][:, :, 300:400]), axis=2
            )
            expected, _ = lbs(
                torch.cat((betas, expression), dim=1),
                poses.flatten(1),
                torch.from_numpy(source["v_template"]).float(),
                torch.from_numpy(shapedirs).float(),
                torch.from_numpy(source["posedirs"].reshape(-1, 36).T.copy()).float(),
                torch.from_numpy(source["J_regressor"]).float(),
                parents,
                torch.from_numpy(source["weights"]).float(),
                pose2rot=True,
            )
            expected += translation[:, None]
        assert actual.shape == (2, 5023, 3) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        report["models"][name] = dict(
            source_sha256=receipt["source_sha256"],
            output_sha256=receipt["output_sha256"],
            array_count=len(receipt["arrays"]),
            arrays_exact=True,
            native_constructor=True,
            native_buffers_match_standard=list(checked_buffers),
            standard_forward_shape=list(actual.shape),
            max_vertex_error=float((actual - expected).abs().max()),
        )
    if args.smpl_source:
        source, _ = read_numeric_smpl(args.smpl_source)
        converted, _ = read_numeric_smpl(args.smpl_numeric)
        assert array_metadata(source) == array_metadata(converted)
        report["smpl_regression"] = dict(
            source_sha256=file_sha256(args.smpl_source),
            output_sha256=file_sha256(args.smpl_numeric),
            arrays_exact=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
