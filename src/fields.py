import numpy as np
from .geometry import CellCentredGrid

# TODO: wall values that vary along a wall (one value per boundary face, not one per wall)
# TODO: Robin (convective) walls would need more than a True/False flag: maybe BC classes
class SimpleWallBC:
    """🤖 [CLAUDE] One uniform boundary condition on each of the six walls of the box.

    Both arrays have shape (3, 2): one row per axis (x, y, z) and one column
    per side (negative, positive). For example, ``[0, 1]`` is the positive-x wall.

    ``is_value`` picks the type of each wall:

    - ``True``: fixed temperature (Dirichlet). ``wall_values`` holds that temperature.
    - ``False``: fixed slope (Neumann). ``wall_values`` holds the slope normal
      to the wall, and 0 means insulated.

    Raises ValueError for wrong shapes or non-finite values, and TypeError if
    ``is_value`` isn't boolean.
    """
    def __init__(self, wall_values, derivative_order):
        ### Coerce inputs into arrays
        wall_values = np.asarray(wall_values, dtype=float)
        derivative_order = np.asarray(derivative_order)

        ### Check shapes: one entry per axis (rows) per side (columns: negative, positive)
        if wall_values.shape != (3, 2):
            raise ValueError(f"expected wall_values with shape (3, 2), got {wall_values.shape}")
        if derivative_order.shape != (3, 2):
            raise ValueError(f"expected derivative_order with shape (3, 2), got {derivative_order.shape}")

        ### Check the flags are integers
        if derivative_order.dtype != int:
            raise TypeError(f"expected derivative_order to contain integers, got dtype {derivative_order.dtype}")

        ### Check every wall number is a real number (no NaN or inf)
        if not np.all(np.isfinite(wall_values)):
            raise ValueError(f"expected finite wall_values, got {wall_values}")

        self.wall_values = wall_values
        self.derivative_order = derivative_order

class InternalScalarField:
    def __init__(self, grid:CellCentredGrid, bcs, values=None):
        self.grid = grid
        if isinstance(bcs, SimpleWallBC):
            self.bcs = bcs
        else:
            raise TypeError(f"Expected a boundary condition object, got {type(bcs)}")
        self.values = values

