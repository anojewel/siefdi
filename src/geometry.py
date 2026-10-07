import numpy as np
from functools import cached_property
class RectangleSpace:
    """🤖 [CLAUDE] Axis-aligned rectangular domain, always stored as 3D.

    The domain is described by one ``bounds`` array of shape (3, 2), laid out
    like ``SimpleWallBC``: one row per axis (x, y, z) and one column per side
    (negative, positive). Axis ``i`` spans ``[bounds[i, 0], bounds[i, 1]]``.
    1D and 2D input is padded to 3D: each missing axis spans ``[0, 1]``, so
    areas and volumes are then per unit length (or per unit area) of the
    missing directions. To use real sizes, pass all three rows yourself.
    Use ``grid()`` to discretise the domain into a mesh.

    Attributes:
        dim (int): Number of spatial dimensions after padding, always 3.
        bounds (np.ndarray): Lower and upper bound on each axis, shape (3, 2).
        negative_min (np.ndarray): Lower bound on each axis, ``bounds[:, 0]``, shape (3,).
        positive_max (np.ndarray): Upper bound on each axis, ``bounds[:, 1]``, shape (3,).
    """

    def __init__(self, bounds):
        """🤖 [CLAUDE] Create the domain from its bounds.

        Args:
            bounds (array-like): ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``,
                with 1 to 3 rows. A single pair ``[x_min, x_max]`` means 1D.
                Missing rows are padded with ``[0, 1]``.

        Raises:
            ValueError: If ``bounds`` doesn't have 2 columns, has more than 3 rows,
                or has a row where min is not smaller than max.
        """
        ### Coerce bounds into a 2D array (a single pair [min, max] becomes one row)
        bounds = np.atleast_2d(np.asarray(bounds, dtype=float))

        ### Check shape: one row per axis, two columns (negative, positive)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"expected bounds with shape (n, 2), got {bounds.shape}")
        if bounds.shape[0] > 3:
            raise ValueError(f"expected at most 3 rows in bounds, got {bounds.shape[0]}")

        ### Check every axis has min < max (otherwise cell widths come out zero or negative)
        if not np.all(bounds[:, 0] < bounds[:, 1]):
            raise ValueError(f"expected bounds[:, 0] < bounds[:, 1] on every axis, got {bounds.tolist()}")

        ### Pad the missing axes with unit length rows [0, 1]
        missing_rows = np.tile([0., 1.], (3 - bounds.shape[0], 1))
        self.bounds = np.vstack([bounds, missing_rows])

        ### Split into the two sides (grid() reads these)
        self.negative_min = self.bounds[:, 0]
        self.positive_max = self.bounds[:, 1]

        self.dim = self.bounds.shape[0]

    def grid(self,centre_type:str,cuts):
        """🤖 [CLAUDE] Discretise the domain into a uniform grid.

        Choose the kind of grid first, then how many cells it has, e.g.
        ``space.grid('cell', [10, 4])``.

        Args:
            centre_type (str): The kind of grid. ``'cell'`` gives a cell-centred
                grid, where faces are placed first and nodes sit midway between
                them. ``'vertex'`` is reserved for a vertex-centred grid, which
                is not implemented yet.
            cuts (int or array-like of int): Number of control volumes per axis,
                1 to 3 entries. A scalar means 1D. Missing axes get 1 cell.

        Returns:
            CellCentredGrid: The grid when ``centre_type == 'cell'``.

        Raises:
            ValueError: If ``cuts`` has more than 3 entries, or if
                ``centre_type`` is not ``'cell'`` or ``'vertex'``.
        """
       
        ### Coerce cuts:
        cuts = np.atleast_1d(np.asarray(cuts, dtype=int))

        ### Check dimensions to be less than 3
        if len(cuts) > 3:
            raise ValueError(f"expected len(cuts) to be less or equal to 3, got {len(cuts)}")
        
        ### Pad missing dimensions with one cut count only
        cuts = np.pad(cuts, (0,3-len(cuts)), constant_values = 1)

        
        ### Switch between cell types

        if centre_type == 'cell':
            faces_list = [np.linspace(self.negative_min[axis],self.positive_max[axis],cuts[axis]+1) for axis in range(self.dim)]
            grid = CellCentredGrid(faces=faces_list)
            return grid
            
        elif centre_type == 'vertex':
            # TODO: vertex-centred grid (also: VertexCenteredGrid.centres doesn't exist, call VertexCenteredGrid(...))
            centres_list = 'do this later lmao'
            grid = VertexCenteredGrid.centres(centres_list) 
            return grid
        
        else:
            raise ValueError(f"expected centre_type as 'cell'/'vertex' got {centre_type}")

class CellCentredGrid:
    """🤖 [CLAUDE] Uniform cell-centred finite-volume grid, stored one axis at a time.

    Quantities that vary along an axis (``faces``, ``centres``, ``centre_spacing``)
    are lists with one array per axis. Quantities that are the same for every
    cell on a uniform grid (``cell_lengths``, ``areas``) are arrays of shape (3,),
    one number per axis. For an axis with N cells:

    Attributes:
        dim (int): Number of spatial dimensions (3 when built by ``RectangleSpace``).
        faces (list[np.ndarray]): Face positions, N+1 per axis, including both
            domain boundaries.
        centres (list[np.ndarray]): Node positions at the cell midpoints, N per axis.
        shape (tuple[int, ...]): Number of cells along each axis.
        centre_spacing (list[np.ndarray]): Distance between neighbouring nodes
            across each face, N+1 per axis. Entry ``i`` belongs to face ``i``: the
            first and last entries are wall-to-centre distances (half a cell on a
            uniform grid), and the rest are centre-to-centre distances.
        cell_lengths (np.ndarray): Face-to-face width of the cells along each
            axis, shape (3,): ``[Δx, Δy, Δz]``.
        areas (np.ndarray): Area of the faces normal to each axis, shape (3,):
            ``[Δy·Δz, Δx·Δz, Δx·Δy]``.
        cell_vol (float): Volume of every cell, ``Δx·Δy·Δz``.
    """

    def __init__(self, faces):
        """🤖 [CLAUDE] Build the grid geometry from face positions.

        Args:
            faces (list[np.ndarray]): Sorted, evenly spaced face positions for
                each axis, including both boundary faces. Even spacing is
                assumed, not checked: ``RectangleSpace.grid()`` guarantees it.
        """
        ### 1. FACES POSITIONS
        self.faces = faces

        ### 2. DIMENSION
        self.dim = len(faces)
    
        ### 3. CENTRE POSITIONS
    @cached_property
    def centres(self):
        # Centres are in between each face:
        centres_list = []
        for axis in range(self.dim):
            singledim_faces = self.faces[axis]
            singledim_centres = (singledim_faces[:-1] + singledim_faces[1:])/2
            centres_list.append(singledim_centres)
        return centres_list
        ### 4. Grid Shape
    @cached_property
    def shape(self):
        return tuple(len(row) for row in self.centres)
    
        ### 5. CENTRE SPACING (centre-to-centre; first/last entries will be wall-to-centre)
    @cached_property
    def centre_spacing(self):
        centre_spacing_list = []
        for axis in range(self.dim):
            singledim_centres = self.centres[axis]
            singledim_faces = self.faces[axis]
            singledim_centre_spacing = (singledim_centres[1:]-singledim_centres[:-1])
            

            # For the first and last centres, we need distance to the space boundaries
            singledim_first_dist =  singledim_centres[0] - singledim_faces[0]
            singledim_last_dist = singledim_faces[-1] - singledim_centres[-1]

            singledim_centre_spacing = np.concatenate(([singledim_first_dist],singledim_centre_spacing,[singledim_last_dist]))

            centre_spacing_list.append(singledim_centre_spacing)

        return centre_spacing_list
        ### 6. CELL LENGTHS (uniform grid: one face-to-face width per axis)
        # Every cell along an axis has the same width, so the first two faces are enough
    @cached_property
    def cell_lengths(self):
        return np.array([singledim_faces[1] - singledim_faces[0] for singledim_faces in self.faces])

        ### 7. SURFACE AREAS (one per axis: the face normal to an axis spans the other two axes)
    @cached_property
    def areas(self):
        # np.delete drops this axis's length, leaving the two lengths that span the face
        return np.array([np.prod(np.delete(self.cell_lengths, axis)) for axis in range(self.dim)])
        
    @cached_property
    def volumes(self):
        ### 8. VOLUMES (every cell has the same volume on a uniform grid)
        return np.prod(self.cell_lengths)


             
            
        
class VertexCenteredGrid:
    """🤖 [CLAUDE] Placeholder for a vertex-centred grid (nodes placed first, faces between them). Not implemented."""

    def __init__(self, centres):
        """🤖 [CLAUDE] Not implemented yet; always raises ``NotImplementedError``.

        Args:
            centres: Intended node positions for each axis.
        """
        self.centres = centres
        raise NotImplementedError("vertex centered grid not yet implemented")
    



        
