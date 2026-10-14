import numpy as np
from .matrix_solver import SystemLinearEquations
from .fields import InternalScalarField, SimpleWallBC
from .history import ScalarHistory
# TODO: 3D assembly: number cells p = i*Ny*Nz + j*Nz + k, 7 non-zero diagonals in A, use scipy.sparse for big grids
# TODO: 3D Jacobi alternative: T = <D|T_nbr>, ghost layer via np.pad
# TODO: Equation class that sums terms (Diffusion, Source, Convection) and solves
# TODO: Source term: S*V goes into b


class Diffusion:
    """🤖 [CLAUDE] Finite-volume diffusion term, ∇·(Γ∇φ), for a scalar field.

    General for any transported scalar φ: Γ is k for heat, μ for momentum,
    ρD for species. Builds the linear system ``A φ = b`` that a steady
    diffusion problem solves, and that a time-marching scheme reuses.

    Attributes:
        field (InternalScalarField): The field being diffused. Its grid gives
            the geometry and its ``bcs`` give the wall values.
        coefficient (float): Γ, uniform over the domain.
    """

    def __init__(self, field:InternalScalarField, coefficient):
        """🤖 [CLAUDE] Store the field and its diffusion coefficient.

        Args:
            field (InternalScalarField): The field this term acts on.
            coefficient (float): Γ, the same in every cell.
        """
        # TODO: make the diffusion coefficient vary per cell (harmonic mean at faces)
        self.field = field
        self.coefficient = coefficient

    ### 1. Calculate Diffusion Conductances
    def conductances(self):
        """🤖 [CLAUDE] Diffusion conductance D = Γ·A/δx of every face, per axis.

        Returns:
            list[np.ndarray]: One array per axis with N+1 entries for N cells,
            laid out like ``grid.centre_spacing``. Entry ``i`` belongs to face
            ``i``: the first and last are wall faces (δx is half a cell), and
            the rest are interior faces (δx is centre to centre).
        """
        target_list = []
        g = self.field.grid
        for axis in range(g.dim):
            ### ↓ THE FORMULA ↓ ###
            singledim_diffusion_conductances = (self.coefficient * g.areas[axis]) * (1/g.centre_spacing[axis])
            target_list.append(singledim_diffusion_conductances)
        return target_list

    def matrix(self):
        ### A. Assemble A matrix
        # 1. Take the list for just one dimensional diffusion conductances (x-direction)
        diffconds_1d = self.conductances()[0]
        
        # 2. Make diagonal with A_{ii} = D_{i} + D_{i+1}
        matrix_diagonal = np.diag([diffconds_1d[i] + diffconds_1d[i+1] for i in range(len(diffconds_1d)-1)])

        # 3. Make off diagonal parts
        matrix_up_offdiag = np.diag([-diffconds_1d[i+1] for i in range(len(diffconds_1d)-2)],1)
    
        # 4. Combine them to get our target matrix (our result A)
        A= matrix_diagonal + matrix_up_offdiag + matrix_up_offdiag.T

        # 5. Check for Neumann
        f = self.field
        if f.bcs.derivative_order[0,0] == 1:
            A[0,0] = diffconds_1d[1]
        if f.bcs.derivative_order[0,1] == 1:
            A[-1,-1] = diffconds_1d[-2]

        return A

    def source_vector(self):
        diffconds_1d = self.conductances()[0]
        ### B. Assemble b vector
        # Get wall temp
        wall_values = self.field.bcs.wall_values
        left_end = diffconds_1d[0]*wall_values[0,0]
        right_end = diffconds_1d[-1]*wall_values[0,1]
        b = np.concatenate([[left_end],np.zeros(self.field.grid.shape[0]-2),[right_end]])

        # Check for Neumann:
        f = self.field
        g = f.grid
        if f.bcs.derivative_order[0,0] == 1:
            b[0] = -diffconds_1d[0] * f.bcs.wall_values[0,0] *g.centre_spacing[0][0]
        if f.bcs.derivative_order[0,1] == 1:
            b[-1] = diffconds_1d[-1] * f.bcs.wall_values[0,1] *g.centre_spacing[0][-1]

        return b

class Transient:
    def __init__(self, field, coefficient):
        self.field = field
        self.coefficient = coefficient
 
class March:
    def __init__(self, diffusion, transient, time_step):
        self.diffusion = diffusion
        self.transient = transient
        self.time_step = time_step
class ExplicitMarch(March):
    def max_time_step(self):
        f = self.diffusion.field
        g = f.grid
        A = self.diffusion.matrix()
        ### TIME LIMIT FORMULA HERE
        return self.transient.coefficient * g.volume / max(np.diag(A))
    def run(self):
        max_time = self.max_time_step()
        T = self.diffusion.field.values
        A = self.diffusion.matrix()
        g = self.diffusion.field.grid
        b = self.diffusion.source_vector()
        dt = self.time_step
        ### CHECK IF TIME STEP IS SMALL ENOUGH
        if self.time_step < max_time:
            ### MARCHING FORMULA HERE
            T = dt/(self.transient.coefficient * g.volume) * (-A@T + b) + T
            # update field (transient and diffusion both point to the same field)
            self.transient.field.values = T
        else:
            raise ValueError(f"Expected time_step to be less than {max_time} got {self.time_step}")
    def record_run(self, t_end, save_every=1):
        n_steps = round(t_end / self.time_step)   
        frames = []
        for n in range(1, n_steps + 1):           
            t = n * self.time_step
            self.run()

            # Record
            if n % save_every == 0:
                single_frame = t,self.diffusion.field.values.copy()
                frames.append(single_frame)
        return ScalarHistory(frames)
            


        
    

    