"""No licensed arrays are embedded in format-compatibility checks."""

import io
import pickle

import numpy as np
import pytest

from luna_open.body_assets import ChumpyLeaf, LegacySMPLUnpickler


def test_chumpy_leaf_requires_plain_numeric_storage():
    leaf = ChumpyLeaf()
    array = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    leaf.__setstate__({"x": array, "_dirty_vars": set(), "_itr": None, "_depends_on_deps": {}})
    np.testing.assert_array_equal(leaf.array, array)
    with pytest.raises(ValueError, match="Computed"):
        leaf.__setstate__({"x": array, "_depends_on_deps": {"expression": 1}})
    with pytest.raises(ValueError, match="numeric"):
        leaf.__setstate__({"x": np.array([object()], dtype=object)})


def test_unknown_pickle_global_is_rejected_before_execution():
    payload = b"cos\nsystem\n(S'touch SHOULD_NEVER_EXIST'\ntR."
    with pytest.raises(pickle.UnpicklingError, match="Unsupported"):
        LegacySMPLUnpickler(io.BytesIO(payload), encoding="latin1").load()


def test_inspected_flame_leaf_metadata_does_not_change_stored_values():
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    state = dict(x=array, _make_dense=False, _make_sparse=False, _status="new")
    leaf = ChumpyLeaf()
    leaf.__setstate__(state)
    assert leaf.array.tobytes() == array.tobytes()
    for key in ("_make_dense", "_make_sparse"):
        with pytest.raises(ValueError, match="representation flags"):
            leaf.__setstate__({**state, key: True})
    with pytest.raises(ValueError, match="diagnostic status"):
        leaf.__setstate__({**state, "_status": "uninspected"})


def test_numpy_pickle_globals_work_in_supported_environment():
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    restored = LegacySMPLUnpickler(io.BytesIO(pickle.dumps(array, protocol=4))).load()
    np.testing.assert_array_equal(restored, array)
