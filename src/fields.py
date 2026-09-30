import numpy as np
from .geometry import CellCentredGrid

# TODO: wall values that vary along a wall (one value per boundary face, not one per wall)
# TODO: Robin (convective) walls would need more than a True/False flag: maybe BC classes
class SimpleWallBC:
    def __init__(self, wall_values, is_value):
        ### Coerce inputs into arrays
        wall_values = np.asarray(wall_values, dtype=float)
        is_value = np.asarray(is_value)

        ### Check shapes: one entry per axis (rows) per side (columns: negative, positive)
        if wall_values.shape != (3, 2):
            raise ValueError(f"expected wall_values with shape (3, 2), got {wall_values.shape}")
        if is_value.shape != (3, 2):
            raise ValueError(f"expected is_value with shape (3, 2), got {is_value.shape}")

        ### Check the flags are True/False, not 0/1 or strings
        if is_value.dtype != bool:
            raise TypeError(f"expected is_value to contain booleans, got dtype {is_value.dtype}")

        ### Check every wall number is a real number (no NaN or inf)
        if not np.all(np.isfinite(wall_values)):
            raise ValueError(f"expected finite wall_values, got {wall_values}")

        self.wall_values = wall_values
        self.is_value = is_value

    ### 1D Check (only heat flow along x: fixed-temperature ends, insulated sides)
    def check_1d(self):
        # x walls must be fixed temperatures
        if not self.is_value[0].all():
            raise NotImplementedError("only fixed-temperature (Dirichlet) x walls are supported")

        # y and z walls must be insulated (zero slope), so they add nothing to A or b
        # TODO: Neumann walls with non-zero slope: b += diffusion_coefficient * area * slope
        if self.is_value[1:].any() or np.any(self.wall_values[1:] != 0):
            raise NotImplementedError("only insulated (zero-slope) y and z walls are supported")
    
def insulated_rod_ends(left, right):
    """🤖 [CLAUDE] Boundary conditions for a 1D rod along x.

    The two x ends are held at fixed temperatures ``left`` and ``right``, and
    the y and z sides are insulated (zero slope).
    """
    wall_values = [[left, right],   # x: fixed temperatures
                   [0.,   0.   ],   # y: zero slope
                   [0.,   0.   ]]   # z: zero slope
    is_value = [[True,  True ],
                [False, False],
                [False, False]]
    return SimpleWallBC(wall_values, is_value)


class InternalScalarField:
    def __init__(self, grid:CellCentredGrid, bcs):
        self.grid = grid
        if isinstance(bcs, SimpleWallBC):
            self.bcs = bcs
        else:
            raise TypeError(f"Expected a boundary condition object, got {type(bcs)}")
        self.values = np.zeros(self.grid.shape)

