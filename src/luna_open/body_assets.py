"""Convert a trusted legacy SMPL pickle without importing obsolete Chumpy.

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
except ImportError:
    from numpy.core.multiarray import _reconstruct


class ChumpyLeaf:
    def __setstate__(self, state):
        allowed = {"x", "_dirty_vars", "_itr", "_depends_on_deps"}
        if not isinstance(state, dict) or set(state) - allowed:
            raise ValueError("Unsupported Chumpy leaf state; inspect the original model")
        if not isinstance(state.get("x"), np.ndarray) or state["x"].dtype.hasobject:
            raise ValueError("Chumpy leaf must contain a numeric ndarray")
        if state.get("_depends_on_deps"):
            raise ValueError("Computed Chumpy expressions are not supported")
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


def read_numeric_smpl(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as stream:
        original = LegacySMPLUnpickler(stream, encoding="latin1").load()
    if not isinstance(original, dict):
        raise ValueError("Expected an SMPL dictionary")
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

    model = convert(original, "")
    expected = {
        "v_template": (6890, 3),
        "posedirs": (6890, 3, 207),
        "weights": (6890, 24),
        "J_regressor": (24, 6890),
        "kintree_table": (2, 24),
        "f": (13776, 3),
    }
    for key, shape in expected.items():
        if key not in model or model[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape; expected {shape}")
    if model["shapedirs"].shape[:2] != (6890, 3) or model["shapedirs"].shape[2] < 10:
        raise ValueError("Expected at least ten SMPL shape components")
    faces = model["f"]
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= 6890:
        raise ValueError("Invalid SMPL triangle indices")
    tree = model["kintree_table"]
    if not np.array_equal(tree[1], np.arange(24)) or np.any(tree[0, 1:] >= np.arange(1, 24)):
        raise ValueError("Unexpected SMPL joint ordering")
    weights = model["weights"]
    if weights.min() < -1e-8 or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid normalized skinning weights")
    if not np.allclose(model["J_regressor"].sum(1), 1, atol=1e-6):
        raise ValueError("Invalid joint regressor normalization")
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
