'''
This file is for writing matrix solver algorithms
'''
import numpy as np

class SystemLinearEquations:
    # Define the problem
    def __init__(self, matrix, rhs_vector, unknown_vector=None):
        self.matrix = matrix
        self.rhs_vector = rhs_vector
        if unknown_vector is None:
            self.unknown_vector = np.zeros_like(rhs_vector)
        else:
            self.unknown_vector = unknown_vector
    def residual(self):
        return self.matrix @ self.unknown_vector - self.rhs_vector
    # TDMA Code starts here
    def thomas(self):
        # Take the values from the class attribute
        matrix = self.matrix
        rhs = self.rhs_vector # this is already d_i

        low_diag = np.diag(matrix,-1)  # this is a_i (lower diagonal, multiplies x_{i-1})
        diag = np.diag(matrix)         # this is b_i (main diagonal)
        up_diag = np.diag(matrix,1)    # this is c_i (upper diagonal, multiplies x_{i+1})

        # To match the index with diag and fulfill the a_1 = 0 and c_n = 0, we pad the left of low_diag and the end of up_diag.
        low_diag = np.pad(low_diag,(1,0))
        up_diag = np.pad(up_diag,(0,1))

        # Note that d_i is already stored in rhs_vector
        P_coeffs = np.zeros_like(rhs)
        Q_coeffs = np.zeros_like(rhs)

        P_coeffs[0] = -up_diag[0]/diag[0]
        Q_coeffs[0] = rhs[0]/diag[0]
        # Create the D_i
        denoms = np.zeros_like(rhs)

        # Here is where the iteration formula is applied
        for i in range(len(rhs)):
            denoms[i] = low_diag[i]*P_coeffs[i-1] + diag[i]
            P_coeffs[i] = -up_diag[i]/denoms[i]
            Q_coeffs[i] = (rhs[i] - low_diag[i]*Q_coeffs[i-1])/denoms[i]
        unknown_vector = self.unknown_vector
        unknown_vector[len(rhs)-1] = Q_coeffs[len(rhs)-1] # This is x_n = Q_n

        for i in range(len(rhs)-2,-1,-1): #<--- the loop starts from the second last
            unknown_vector[i] = P_coeffs[i]*unknown_vector[i+1] + Q_coeffs[i] 

        # Create a new same kind of module that has this as the unknown vector
        return SystemLinearEquations(self.matrix,self.rhs_vector,unknown_vector)

    # Gauss-Seidel Code starts here
    def gauss_seidel(self, initial_guess, tolerance, max_iterations=100_000):
        # Take the values from the class attribute
        matrix = self.matrix
        rhs = self.rhs_vector

        # Stretch a single number to every cell, and copy so the guess passed in is not overwritten
        unknown_vector = np.broadcast_to(initial_guess, rhs.shape).astype(float)

        for k in range(1, max_iterations + 1):
            previous_sweep = unknown_vector.copy() # frozen snapshot of the previous sweep
            # One sweep, this is the process of infection
            for i in range(len(rhs)):
                infected_part = matrix[i, :i] @ unknown_vector[:i]      # columns before i: already this sweep
                healthy_part = matrix[i, i+1:] @ unknown_vector[i+1:]   # columns after i: still last sweep
                unknown_vector[i] = (rhs[i] - infected_part - healthy_part)/matrix[i, i]
            if np.max(np.abs(unknown_vector - previous_sweep)) < tolerance:
                break

        # Create a new same kind of module that has this as the unknown vector
        return SystemLinearEquations(self.matrix,self.rhs_vector,unknown_vector), k

                        

        
        
        
            



            
