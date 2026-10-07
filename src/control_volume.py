import numpy as np
from .matrix_solver import SystemLinearEquations
# TODO: 3D assembly: number cells p = i*Ny*Nz + j*Nz + k, 7 non-zero diagonals in A, use scipy.sparse for big grids
# TODO: 3D Jacobi alternative: T = <D|T_nbr>, ghost layer via np.pad
# TODO: Equation class that sums terms (Diffusion, Source, Convection) and solves
# TODO: Source term: S*V goes into b


class Diffusion:

    def __init__(self, field, diffusion_coefficient):
        # TODO: make the diffusion coefficient vary per cell (harmonic mean at faces)
        self.field = field
        self.diffusion_coefficient = diffusion_coefficient

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

        return SystemLinearEquations(Amatrix_assembled,b_vector_assembled)
            