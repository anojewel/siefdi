import numpy as np

class RectangleSpace:
    """🤖 [CLAUDE] Axis-aligned rectangular domain, always stored as 3D.

    The domain spans ``[negative_min[axis], positive_max[axis]]`` along each axis.
    1D and 2D input is padded to 3D: each missing axis spans ``[0, 1]``, so
    areas and volumes are then per unit length (or per unit area) of the
    missing directions. To use real sizes, pass all three extents yourself.
    Use ``grid()`` to discretise the domain into a mesh.

    Attributes:
        dim (int): Number of spatial dimensions after padding, always 3.
        negative_min (np.ndarray): Lower bound of the domain on each axis, shape (3,).
        positive_max (np.ndarray): Upper bound of the domain on each axis, shape (3,).
    """

    # TODO: maybe take one (3, 2) bounds array, to match SimpleWallBC's layout
    def __init__(self, negative_min, positive_max):
        """🤖 [CLAUDE] Create the domain from its lower and upper bounds.

        Args:
            negative_min (float or array-like): Lower bound per axis, 1 to 3
                entries. A scalar means 1D. Missing axes are padded with 0.
            positive_max (float or array-like): Upper bound per axis, 1 to 3
                entries. A scalar means 1D. Missing axes are padded with 1.

        Raises:
            ValueError: If ``negative_min`` and ``positive_max`` have different
                lengths, or more than 3 entries.
        """
        ### Coerce the lengths input into array:
        normalized_positive_max = np.atleast_1d(np.asarray(positive_max, dtype=float))
        normalized_negative_min = np.atleast_1d(np.asarray(negative_min, dtype=float))

        # Check if two objects are in agreement
        if len(normalized_positive_max) == len(normalized_negative_min):
            self.dim = len(normalized_positive_max)
            self.positive_max = normalized_positive_max
            self.negative_min = normalized_negative_min 
        else:
            raise ValueError(f"expected positive_len and negative_len to be equal dimensions, got {len(normalized_positive_max)} and {len(normalized_negative_min)}")

        if self.dim > 3:
            raise ValueError(f"expected positive_max or negative_min lengths to be less or equal to 3, got {self.dim}")
        
        # Pad the missing values with unit length to the positive side
        self.positive_max = np.pad(self.positive_max, (0, 3-len(self.positive_max)), constant_values = 1.)
        self.negative_min = np.pad(self.negative_min, (0, 3-len(self.negative_min)), constant_values = 0.)

        self.dim = len(self.positive_max)
                             
    def grid(self, cuts, centre_type:str):
        """🤖 [CLAUDE] Discretise the domain into a uniform grid.

        Args:
            cuts (int or array-like of int): Number of control volumes per axis,
                1 to 3 entries. A scalar means 1D. Missing axes get 1 cell.
            centre_type (str): ``'cell'`` for a cell-centred grid, where faces are
                placed first and nodes sit midway between them. ``'vertex'`` is
                reserved for a vertex-centred grid, which is not implemented yet.

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

        # Centres are in between each face:
        centres_list = []
        for axis in range(self.dim):
            singledim_faces = faces[axis]
            singledim_centres = (singledim_faces[:-1] + singledim_faces[1:])/2
            centres_list.append(singledim_centres)
        # Store instance attribute
        self.centres = centres_list

        ### 4. SHAPE (number of cells per axis)
        self.shape = tuple([len(row) for row in self.centres])

        ### 5. CENTRE SPACING (centre-to-centre; first/last entries will be wall-to-centre)
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

        self.centre_spacing = centre_spacing_list

        # TODO: non-uniform grids would need cell_lengths, areas and volumes per cell, not per axis
        ### 6. CELL LENGTHS (uniform grid: one face-to-face width per axis)
        # Every cell along an axis has the same width, so the first two faces are enough
        self.cell_lengths = np.array([singledim_faces[1] - singledim_faces[0] for singledim_faces in self.faces])

        ### 7. SURFACE AREAS (one per axis: the face normal to an axis spans the other two axes)
        # np.delete drops this axis's length, leaving the two lengths that span the face
        self.areas = np.array([np.prod(np.delete(self.cell_lengths, axis)) for axis in range(self.dim)])

        ### 8. VOLUMES (every cell has the same volume on a uniform grid)
        self.cell_vol = np.prod(self.cell_lengths)


             
            
        
class VertexCenteredGrid:
    """🤖 [CLAUDE] Placeholder for a vertex-centred grid (nodes placed first, faces between them). Not implemented."""

    def __init__(self, centres):
        """🤖 [CLAUDE] Not implemented yet; always raises ``NotImplementedError``.

        Args:
            centres: Intended node positions for each axis.
        """
        self.centres = centres
        raise NotImplementedError("vertex centered grid not yet implemented")
    



        
