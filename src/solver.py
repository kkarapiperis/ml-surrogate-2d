# -*- coding: utf-8 -*-
"""
Solver class

@author: Konstantinos Karapiperis
"""
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

class NewtoRaphsonSolver():
    
    def __init__(self, assembler, max_iterations):
        
        self.assembler = assembler
        self.max_iterations = max_iterations
        self.n_dof = assembler.n_dof

    def compute_solution(self, essential_bcs, initial_guess, verbose=0, abs_tol=1e-6):
        """
        Returns the converged solution starting from 
        given displacement initial guess
        """
        # Initialize displacements
        displacements = np.copy(initial_guess)

        # Impose essential bcs initially
        displacements[essential_bcs.dofs] = essential_bcs.vals

        # Create mask for free DOFs
        all_dofs = np.arange(self.n_dof)
        free_dofs = np.setdiff1d(all_dofs, essential_bcs.dofs)
        
        # Start iterating
        iter = 0
        residual = 1.0
        residuals = [residual]

        while residual > abs_tol and iter < self.max_iterations:
            print(f"Iteration {iter}, Residual: {residual:.6e}")

            # Update local states for current displacement
            self.assembler.assign_local_states(displacements)

            # Get residual vector (internal - external forces)
            forces = self.assembler.assemble_force_vector()
            
            # Get tangent stiffness
            K = self.assembler.assemble_stiffness_matrix()

            # Prepare system for solution increment
            K_mod = K.copy()
            rhs = -forces.copy()  # negative residual on RHS
            
            # Modify system for essential BCs
            for dof in essential_bcs.dofs:
                # Zero row and column, set diagonal to 1
                K_mod[dof, :] = 0.0
                K_mod[:, dof] = 0.0
                K_mod[dof, dof] = 1.0
                # Zero RHS to maintain current essential values
                rhs[dof] = 0.0

            # Check for singularity
            if np.any(np.abs(np.diag(K_mod)) < 1e-10):
                sys.exit('Solver found zero diagonal in modified stiffness matrix. Exiting.')

            # Solve for displacement increment
            du = np.linalg.solve(K_mod, rhs)
            
            # Update displacements
            displacements -= du
            
            # Re-enforce essential BCs (numerical safety)
            displacements[essential_bcs.dofs] = essential_bcs.vals
            
            # Compute residual norm using only free DOFs
            residual = np.linalg.norm(forces[free_dofs])
            residuals.append(residual)
            
            print(f'Increment norm: {np.linalg.norm(du):.2e}')
            print('Residual:\n', residual)
            
            iter += 1

            if verbose:
                plt.clf()
                plt.xlabel('Iterations')
                plt.ylabel('Residual')
                plt.plot(np.arange(iter+1), residuals, '-o', c='r')
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.5)

        if verbose: 
            plt.close()

        if iter == self.max_iterations and residual > abs_tol:
            sys.exit('Reached maximum number of iterations. Exiting.')

        return displacements
    