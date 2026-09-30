"""Convert trusted legacy SMPL/FLAME pickles without importing obsolete Chumpy.

Only bare chumpy.ch.Ch leaves are supported: Ch.compute_r returns its stored x.
This is a narrow compatibility reader, not a sandbox for untrusted pickles.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # NumPy 1.x in the isolated baseline environment.
    from numpy.core.multiarray import _reconstruct


class ChumpyLeaf:
    def __setstate__(self, state):
        allowed = {
            "x",
            "_dirty_vars",
            "_itr",
            "_depends_on_deps",
            "_make_dense",
            "_make_sparse",
            "_status",
        }
        if not isinstance(state, dict) or set(state) - allowed:
            raise ValueError("Unsupported Chumpy leaf state; inspect the original model")
        if not isinstance(state.get("x"), np.ndarray) or state["x"].dtype.hasobject:
            raise ValueError("Chumpy leaf must contain a numeric ndarray")
        if state.get("_depends_on_deps"):
            raise ValueError("Computed Chumpy expressions are not supported")
        # Inspected FLAME files contain disabled representation flags and a
        # diagnostic status. Do not generalize this to active sparse conversion.
        if any(state.get(key, False) is not False for key in ("_make_dense", "_make_sparse")):
            raise ValueError("Active Chumpy representation flags are not supported")
        if "_status" in state and state["_status"] != "new":
            raise ValueError("Unsupported Chumpy diagnostic status")
        self.array = state["x"]


class LegacySMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("chumpy.ch", "Ch"): ChumpyLeaf,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
            ("scipy.sparse.csc", "csc_matrix"): sparse.csc_matrix,
            ("scipy.sparse._csc", "csc_matrix"): sparse.csc_matrix,
            ("__builtin__", "set"): set,
            ("builtins", "set"): set,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported model pickle global: {module}.{name}")
        return allowed[module, name]


def _read_numeric_dictionary(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as stream:
        original = LegacySMPLUnpickler(stream, encoding="latin1").load()
    if not isinstance(original, dict):
        raise ValueError("Expected a body-model dictionary")
    changes = {}

    def convert(value, key):
        if isinstance(value, ChumpyLeaf):
            value = value.array
            changes[key] = "bare chumpy.ch.Ch.x to ndarray"
        elif sparse.issparse(value):
            value = value.toarray()
            changes[key] = "sparse matrix to dense ndarray"
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Non-numeric/nonfinite array: {key}")
            return np.ascontiguousarray(value)
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError("Expected string model keys")
            return {k: convert(v, f"{key}.{k}" if key else k) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(convert(v, f"{key}.{i}") for i, v in enumerate(value))
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            raise ValueError(f"Unsupported model value: {key}: {type(value)}")

    return convert(original, ""), changes


def _validate_lbs_arrays(model: dict, vertices: int, joints: int, faces_count: int):
    expected = {
        "v_template": (vertices, 3),
        "posedirs": (vertices, 3, (joints - 1) * 9),
        "weights": (vertices, joints),
        "J_regressor": (joints, vertices),
        "kintree_table": (2, joints),
        "f": (faces_count, 3),
    }
    for key, shape in expected.items():
        if key not in model or model[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape; expected {shape}")
    faces = model["f"]
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= vertices:
        raise ValueError("Invalid body-model triangle indices")
    tree = model["kintree_table"]
    if not np.array_equal(tree[1], np.arange(joints)) or np.any(
        tree[0, 1:] >= np.arange(1, joints)
    ):
        raise ValueError("Unexpected body-model joint ordering")
    weights = model["weights"]
    if weights.min() < -1e-8 or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid normalized skinning weights")
    if not np.allclose(model["J_regressor"].sum(1), 1, atol=1e-6):
        raise ValueError("Invalid joint regressor normalization")


def read_numeric_smpl(path: Path) -> tuple[dict, dict]:
    model, changes = _read_numeric_dictionary(path)
    _validate_lbs_arrays(model, vertices=6890, joints=24, faces_count=13776)
    if model["shapedirs"].shape[:2] != (6890, 3) or model["shapedirs"].shape[2] < 10:
        raise ValueError("Expected at least ten SMPL shape components")
    return model, changes


def read_numeric_flame(path: Path) -> tuple[dict, dict]:
    """Read the inspected full FLAME model: 300 shape + 100 expression bases."""
    model, changes = _read_numeric_dictionary(path)
    _validate_lbs_arrays(model, vertices=5023, joints=5, faces_count=9976)
    if model["shapedirs"].shape != (5023, 3, 400):
        raise ValueError("Expected 300 FLAME shape and 100 expression components")
    if model.get("bs_style") != "lbs" or model.get("bs_type") != "lrotmin":
        raise ValueError("Unexpected FLAME blendshape convention")
    return model, changes


def array_metadata(model: dict) -> dict:
    metadata = {}

    def visit(value, key):
        if isinstance(value, np.ndarray):
            metadata[key] = dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{key}.{k}" if key else k)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(v, f"{key}.{i}")

    visit(model, "")
    return metadata
