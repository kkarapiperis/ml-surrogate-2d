# -*- coding: utf-8 -*-
"""
Material point class

@author: Konstantinos Karapiperis
"""
import numpy as np
from utilities import *

class MaterialPoint():
    
    def __init__(self, mp_id, density, thickness, nodal_positions, connectivity, quad_element, quad_rule, qp_index, constitutive_model):
        
        self.mp_id = mp_id
        self.density = density
        self.thickness = thickness
        self.nodal_positions = nodal_positions
        self.connectivity = connectivity
        self.quad_element = quad_element
        self.quad_rule = quad_rule
        self.qp_index = qp_index
        self.constitutive_model = constitutive_model
        self.nodal_support = 4
        self.n_dof = 2
        self.strain = np.zeros((self.n_dof,self.n_dof))
        self.stress = np.zeros((self.n_dof,self.n_dof))

        # One-time calculations
        self.compute_shape_functions()
    
    def compute_shape_functions(self):
        """
        Compute jacobian of the mapping, shape functions and their derivatives
        """
        # Find location of quadrature point in parent domain
        quad_point = self.quad_rule.points[self.qp_index]

        # Compute shape function
        self.N = self.quad_element.compute_N(quad_point)

        # Compute shape function derivatives in parent domain
        dN_parent = self.quad_element.compute_dN(quad_point)

        # Initialize shape function derivatives in physical domain
        self.dN = np.zeros((self.nodal_support,self.n_dof))

        # Compute jacobian
        jacobian = np.zeros((self.n_dof,self.n_dof))
        for node_idx in range(self.nodal_support):
            pos = self.nodal_positions[self.connectivity[node_idx]]
            for dof_idx1 in range(self.n_dof):
                for dof_idx2 in range(self.n_dof):
                    jacobian[dof_idx1,dof_idx2] += pos[dof_idx2]*dN_parent[node_idx][dof_idx1]
        
        # Compute shape function derivatives
        for node_idx in range(self.nodal_support):
            self.dN[node_idx] = np.linalg.inv(jacobian).dot(dN_parent[node_idx])

        # Compute volume and nodal weights
        self.volume = np.abs(np.linalg.det(jacobian)) * self.quad_rule.weights[self.qp_index] * self.thickness
        self.nodal_weights = np.full(self.nodal_support, self.volume/self.nodal_support)

    def compute_forces(self):
        '''
        Computes forces 
        '''
        forces = []
        for node_idx in range(self.nodal_support):
            forces.append(-self.stress.dot(self.dN[node_idx]) * self.volume)
        return forces

    def compute_stiffness_matrix(self):
        '''
        Computes local stiffness matrix
        '''
        k_e = np.zeros((self.n_dof*self.nodal_support, 
                        self.n_dof*self.nodal_support))        

        for node_idx_A in range(self.nodal_support):
            for node_idx_B in range(self.nodal_support):
                for i in range(self.n_dof):
                    voigt_idx_Ai = convert_to_voigt_idx(node_idx_A, i)
                    for k in range(self.n_dof):
                        voigt_idx_Bk = convert_to_voigt_idx(node_idx_B, k)
                        for j in range(self.n_dof):
                            voigt_idx_ij = convert_to_voigt_idx(i,j)
                            for l in range(self.n_dof):
                                voigt_idx_kl = convert_to_voigt_idx(k,l)
                                k_e[voigt_idx_Ai,voigt_idx_Bk] += \
                                    self.tangent[voigt_idx_ij,voigt_idx_kl] * \
                                    self.dN[node_idx_A][j] * self.dN[node_idx_B][l]

        return k_e * self.volume

    def assign_local_states(self, displacements):
        """
        Updates strain, stress, tangent in the material point
        """
        # Update strain
        disp_gradient = np.zeros((self.n_dof, self.n_dof))
        for node_idx in range(self.nodal_support):
            disp_gradient += np.outer(displacements[node_idx],self.dN[node_idx])
        self.strain = 0.5*(disp_gradient + disp_gradient.T)
        strain_voigt = convert_to_voigt_tensor(self.strain)
        strain_voigt = convert_voigt_to_reduced_voigt(strain_voigt)
        stress_voigt = self.constitutive_model.compute_stress(strain_voigt)
        stress_voigt = convert_reduced_voigt_to_voigt(stress_voigt)
        self.stress = convert_to_standard_tensor(stress_voigt)
        self.tangent = self.constitutive_model.compute_stiffness(strain_voigt)
        self.tangent = convert_reduced_voigt_to_voigt_2order(self.tangent)