import numpy as np

# TODO: 3D assembly: number cells p = i*Ny*Nz + j*Nz + k, 7 non-zero diagonals in A, use scipy.sparse for big grids
# TODO: 3D Jacobi alternative: T = <D|T_nbr>, ghost layer via np.pad
# TODO: Equation class that sums terms (Diffusion, Source, Convection) and solves
# TODO: Source term: S*V goes into b


class Diffusion:

    def __init__(self, field, diffusion_coefficient):
        # TODO: make the diffusion coefficient vary per cell (harmonic mean at faces)
        self.field = field
        self.diffusion_coefficient = diffusion_coefficient

    ### 0. 1D Guard (this solver only handles heat flow along x)
    def check_1d(self):
        g = self.field.grid

        # Only one cell across y and z, so there are no interior faces in those directions
        if g.shape[1:] != (1, 1):
            raise NotImplementedError(f"only 1D grids along x are supported, got shape {g.shape}")

        # The walls must fit a 1D problem too
        self.field.bcs.check_1d()

    ### 1. Calculate Diffusion Conductances
    def diffusion_conductances(self):
        target_list = []
        g = self.field.grid
        for axis in range(g.dim):
            singledim_diffusion_conductances = (self.diffusion_coefficient * g.areas[axis]) * (1/g.centre_spacing[axis])
            target_list.append(singledim_diffusion_conductances)
        return target_list

    ### 2. Assemble A and b (1D along x)
    def assemble(self):
        self.check_1d()
        ### A. Assemble A matrix
        # 1. Take the list for just one dimensional diffusion conductances (x-direction)
        diffconds_1d = self.diffusion_conductances()[0]

        # 2. Make diagonal with A_{ii} = D_{i} + D_{i+1}
        matrix_diagonal = np.diag([diffconds_1d[i] + diffconds_1d[i+1] for i in range(len(diffconds_1d)-1)])

        # 3. Make off diagonal parts
        matrix_up_offdiag = np.diag([-diffconds_1d[i+1] for i in range(len(diffconds_1d)-2)],1)
    
        # 4. Combine them to get our target matrix (our result A)
        Amatrix_assembled = matrix_diagonal + matrix_up_offdiag + matrix_up_offdiag.T

        ### B. Assemble b vector
        
        # Get wall temp
        wall_values = self.field.bcs.wall_values
        left_end = diffconds_1d[0]*wall_values[0,0]
        right_end = diffconds_1d[-1]*wall_values[0,1]

        b_vector_assembled = np.concatenate([[left_end],np.zeros(self.field.grid.shape[0]-2),[right_end]])

    
        return Amatrix_assembled,b_vector_assembled
            
    
    ### 3. Solve A T = b
    def solve(self):
        A, b = self.assemble()

        # Here we use the linalg solve which i dont know yet about
        T = np.linalg.solve(A, b)

        # Convert to desired shape
        self.field.values = T.reshape(self.field.grid.shape)

    ### 4. Plotter
    def plot(self, exact=None, ax=None):
        """🤖 [CLAUDE] Plot the solved temperatures along x, optionally against an exact solution.

        Args:
            exact (callable, optional): Exact solution ``T(x)``, drawn as a line.
            ax (matplotlib Axes, optional): Axes to draw on. A new figure is made if omitted.

        Returns:
            matplotlib Axes: The axes the plot was drawn on.
        """
        import matplotlib.pyplot as plt

        g = self.field.grid
        x_centres = g.centres[0]
        T_numerical = self.field.values.ravel()

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))

        # Exact solution first, so the numerical markers sit on top of it
        if exact is not None:
            x_fine = np.linspace(g.faces[0][0], g.faces[0][-1], 200)
            ax.plot(x_fine, exact(x_fine), color='#eb6834', linewidth=2, label='Analytical', zorder=1)

        # Numerical solution: one marker per cell centre, with a white ring so it reads over the line
        ax.plot(x_centres, T_numerical, linestyle='none', marker='o', markersize=8,
                color='#2a78d6', markeredgecolor='white', markeredgewidth=1.5,
                label=f'Numerical (N = {len(x_centres)})', zorder=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('T [°C]')
        ax.set_title('Temperature along the rod')

        # Recessive grid and axes, so the data stands out
        ax.grid(True, color='#e0e0e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

        ax.legend(frameon=False)
        return ax

            