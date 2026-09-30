import numpy as np
import pytest

from src.fields import SimpleWallBC


### Shared input: the rod from Assignment 1
# Rows are axes (x, y, z); columns are sides (negative, positive)
ROD_WALL_VALUES = [[100., 500.],   # x: fixed temperatures at both ends
                   [  0.,   0.],   # y: zero slope (insulated)
                   [  0.,   0.]]   # z: zero slope (insulated)
ROD_IS_VALUE = [[True,  True ],
                [False, False],
                [False, False]]


### 1. Good input is stored as arrays with the right shape and type
def test_stores_rod_bcs():
    bc = SimpleWallBC(ROD_WALL_VALUES, ROD_IS_VALUE)

    assert bc.wall_values.shape == (3, 2)
    assert bc.is_value.dtype == bool
    np.testing.assert_allclose(bc.wall_values, ROD_WALL_VALUES)


### 2. Bad input raises the right error
def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        SimpleWallBC([100., 500.], ROD_IS_VALUE)   # only x given, shape (2,)


def test_integer_flags_raise():
    with pytest.raises(TypeError):
        SimpleWallBC(ROD_WALL_VALUES, [[1, 1], [0, 0], [0, 0]])   # 0/1, not True/False
