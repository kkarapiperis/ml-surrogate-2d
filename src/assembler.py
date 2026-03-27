# -*- coding: utf-8 -*-
"""
Assembler class

@author: Konstantinos Karapiperis
"""

import numpy as np
from utilities import *

class Assembler():

    def __init__(self, mat_points, n_nodes):
        
        self.mat_points = mat_points
        self.n_mat_points = len(mat_points)
        self.n_nodes = n_nodes
        self.nodal_support = mat_points[0].nodal_support
        self.n_dof = mat_points[0].n_dof

    def assign_local_states(self, displacements):
        """
        Assign local states for each material point
        """
        for mp_idx in range(self.n_mat_points):
            node_idxs = self.mat_points[mp_idx].connectivity
            dof_idxs = np.vstack((self.n_dof*node_idxs,self.n_dof*node_idxs+1))
            dof_idxs = dof_idxs.reshape((-1,),order='F')
            disp_mp = displacements[dof_idxs].reshape(-1,self.n_dof)
            self.mat_points[mp_idx].assign_local_states(disp_mp)

    def assemble_force_vector(self):
        """
        Assemble the force vector from all material points
        """
        # Initialize
        forces = np.zeros(self.n_nodes*self.n_dof)
        
        # Iterate over material points
        for mp_idx in range(self.n_mat_points):
            force_mp = self.mat_points[mp_idx].compute_forces()
            node_idxs = self.mat_points[mp_idx].connectivity
            dof_idxs = np.vstack((self.n_dof*node_idxs,self.n_dof*node_idxs+1))
            dof_idxs = dof_idxs.reshape((-1,),order='F')
            forces[dof_idxs] += np.array(force_mp).ravel()
        
        return forces

    def assemble_stiffness_matrix(self):
        """
        Assemble the stiffness matrix from all material points
        """
        # Initialize
        stiffness = np.zeros((self.n_nodes*self.n_dof, self.n_nodes*self.n_dof))

        # Iterate over material points
        for mp_idx in range(self.n_mat_points):
            stiffness_mp = self.mat_points[mp_idx].compute_stiffness_matrix()
            for node_idx_A in range(self.nodal_support):
                node_A = self.mat_points[mp_idx].connectivity[node_idx_A]
                for node_idx_B in range(self.nodal_support):
                    node_B = self.mat_points[mp_idx].connectivity[node_idx_B]
                    for i in range(self.n_dof):
                        voigt_idx_Ai = convert_to_voigt_idx(node_A, i)
                        voigt_idx_Ai_loc = convert_to_voigt_idx(node_idx_A, i)
                        for j in range(self.n_dof):
                            voigt_idx_Bj = convert_to_voigt_idx(node_B, j)
                            voigt_idx_Bj_loc = convert_to_voigt_idx(node_idx_B, j)
                            stiffness[voigt_idx_Ai,voigt_idx_Bj] += \
                                stiffness_mp[voigt_idx_Ai_loc, voigt_idx_Bj_loc] 

        return stiffness

    def compute_node_strains(self):
        """
        Return average strain at nodes from neighboring material points
        """
        # Initialize
        node_strains = np.zeros((self.n_nodes,self.n_dof, self.n_dof))
        node_weights = np.zeros(self.n_nodes)

        # Weighted contribution from all material points
        for mp_idx in range(self.n_mat_points):
            eps_mp = self.mat_points[mp_idx].strain
            weight_mp = self.mat_points[mp_idx].nodal_weights

            for node_idx,node in enumerate(self.mat_points[mp_idx].connectivity):
                node_strains[node] += eps_mp * weight_mp[node_idx]
                node_weights[node] += weight_mp[node_idx]

        for node_idx in range(self.n_nodes):
            node_strains[node_idx] /= node_weights[node_idx] 

        return node_strains

    def compute_node_stresses(self):
        """
        Return average stress at nodes from neighboring material points
        """
        # Initialize
        node_stresses = np.zeros((self.n_nodes,self.n_dof, self.n_dof))
        node_weights = np.zeros(self.n_nodes)

        # Weighted contribution from all material points
        for mp_idx in range(self.n_mat_points):
            sigma_mp = self.mat_points[mp_idx].stress
            weight_mp = self.mat_points[mp_idx].nodal_weights

            for node_idx,node in enumerate(self.mat_points[mp_idx].connectivity):
                node_stresses[node] += sigma_mp * weight_mp[node_idx]
                node_weights[node] += weight_mp[node_idx]

        for node_idx in range(self.n_nodes):
            node_stresses[node_idx] /= node_weights[node_idx] 

        return node_stresses