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
