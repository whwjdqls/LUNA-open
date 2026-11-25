"""Validate native source fits before constructing body-only transfer meshes.

The model-transfer procedure discards hand/face articulation. Original records
remain unchanged and fingerprinted. Native SMPL-X model outputs are in meters;
nonunit source scales require a separate audited conversion and are refused.
"""

import numpy as np


def numeric_npz(path):
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key] for key in source.files}
    if any(
        value.dtype.kind not in "iuf" or not np.isfinite(value).all() for value in result.values()
    ):
        raise ValueError("Non-numeric or nonfinite native body parameter")
    return result


def unit_scale(value):
    value = np.asarray(value)
    if (
        value.size != 1
        or not np.isfinite(value).all()
        or not np.isclose(value.item(), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("Nonunit native SMPL-X scale needs an audited conversion")


def mvhuman_individual(individual):
    """Validate a self-contained native fit without inventing aggregate agreement."""
    widths = dict(
        betas=10,
        global_orient=3,
        transl=3,
        body_pose=63,
        left_hand_pose=6,
        right_hand_pose=6,
        jaw_pose=3,
        expression=10,
    )
    if not set(widths).union({"scale"}) <= individual.keys():
        raise ValueError("Individual SMPL-X record lacks required arrays")
    unit_scale(individual["scale"])
    for key, width in widths.items():
        value = np.asarray(individual[key])
        if (
            value.shape != (1, width)
            or value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid individual SMPL-X array")
    for key in ("leye_pose", "reye_pose"):
        if key in individual and (
            individual[key].shape != (1, 3) or not np.isfinite(individual[key]).all()
        ):
            raise ValueError("Invalid individual eye pose")
    pose = np.concatenate((individual["global_orient"], individual["body_pose"]), axis=1)
    if "pose" in individual and (
        individual["pose"].shape != (1, 66)
        or not np.allclose(pose, individual["pose"], atol=1e-7, rtol=0)
    ):
        raise ValueError("Individual concatenated pose disagrees with root/body arrays")
    return {
        key: individual[key][0].astype(np.float32)
        for key in ("betas", "global_orient", "transl", "body_pose")
    }


def mvhuman_standard_smplx(record, components):
    """Decode full standard SMPL-X motion, retaining source hands and face.

    PCA components must come from the same neutral source asset. Cast before
    multiplying to match the FP32 source model. Translation is stored separately
    and composed with the calibrated camera exactly once by the dataset reader.
    """
    mvhuman_individual(record)
    values = {key: np.asarray(value, dtype=np.float32) for key, value in record.items()}
    hands = []
    for side in ("left", "right"):
        basis = np.asarray(components[side], dtype=np.float32)
        if basis.shape != (6, 45) or not np.isfinite(basis).all():
            raise ValueError("Standard SMPL-X needs the asset's finite six-component hand bases")
        hands.append(values[f"{side}_hand_pose"] @ basis)
    eyes = [
        values.get(key, np.zeros((1, 3), dtype=np.float32)) for key in ("leye_pose", "reye_pose")
    ]
    pose = np.concatenate(
        [values["global_orient"], values["body_pose"], values["jaw_pose"], *eyes, *hands], axis=1
    )[0]
    return dict(
        pose=pose,
        betas=values["betas"][0],
        expression=values["expression"][0],
        transl=values["transl"][0],
    )


def mvhuman_frame(aggregate, individual, frame_id):
    """Require actual per-frame agreement before using the example's index rule."""
    if type(frame_id) is not int or frame_id <= 0 or frame_id % 25:
        raise ValueError("Frame is outside the verified numeric MVHumanNet++ indexing rule")
    index = frame_id // 25 - 1
    result = mvhuman_individual(individual)
    widths = dict(
        betas=10,
        global_orient=3,
        transl=3,
        body_pose=63,
        left_hand_pose=6,
        right_hand_pose=6,
        jaw_pose=3,
        expression=10,
    )
    for key, width in widths.items():
        a, b = aggregate[key], individual[key]
        slot = 0 if key == "betas" else index
        if (
            a.ndim != 2
            or a.shape[1] != width
            or b.shape != (1, width)
            or slot >= len(a)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or not np.allclose(a[slot], b[0], atol=1e-7, rtol=0)
        ):
            raise ValueError("Aggregate and individual SMPL-X frame disagree")
    if aggregate["betas"].shape != (1, 10):
        raise ValueError("Expected one source shape for this outfit")
    return result


def dna_frame(values):
    """Decode the documented 55-joint native pose; actual SMC audit is separate."""
    unit_scale(values["scale"])
    pose = np.asarray(values["fullpose"])
    if pose.shape not in {(55, 3), (165,)}:
        raise ValueError("Expected the native 55-joint SMPL-X pose")
    pose = pose.reshape(55, 3)
    betas, translation = np.asarray(values["betas"]), np.asarray(values["transl"])
    if betas.shape != (10,) or translation.shape != (3,):
        raise ValueError("Unexpected native DNA shape or translation")
    if not all(np.isfinite(value).all() for value in (pose, betas, translation)):
        raise ValueError("Nonfinite native DNA body parameters")
    return dict(
        betas=betas.astype(np.float32),
        global_orient=pose[0].astype(np.float32),
        body_pose=pose[1:22].reshape(63).astype(np.float32),
        transl=translation.astype(np.float32),
    )
