# Regularized Gauß-Newton
# similar to https://arxiv.org/pdf/2103.15138.pdf

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import time 
import os 
from scipy.linalg import block_diag
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

"""
Here, we use the Jacobian calculated using Fenics.

"""

from ..ktc_methods import create_tv_matrix, SMPrior

class LinearisedRecoFenics():
    def __init__(self, Uref, B, vincl, mesh_name="dense", 
                    base_path = "data",
                    version2=True, noise_level=(0.05, 0.01)):
        """
        Uref: reference measurements from empty water-tank
        B: observation operator 31x32 Mpat.T
        solver: instance of EITFEM solver 
        Ltv: TV matrix 
        mesh_name: mesh for sigma
        """
        
        self.Uref = Uref
        self.mesh_name = mesh_name
        
        ## load regularizers ##
        # TV matrix
        Rtv = np.load(os.path.join(base_path, f"mesh_neighbour_matrix_{self.mesh_name}.npy"))*1.0
        np.fill_diagonal(Rtv, -Rtv.sum(axis=0))
        self.Rtv = -Rtv

        # Smoothness prior 
        self.Rsm = np.load(os.path.join(base_path, f"smoothnessR_{self.mesh_name}.npy"))

        self.noise_std1 = noise_level[0]  # standard deviation of the noise as percentage of each voltage measurement
        self.noise_std2 = noise_level[1] #standard deviation of 2nd noise component (this is proportional to the largest measured value)
        
        J = np.load(os.path.join(base_path, f"jac_{mesh_name}.npy"))
        J = J.reshape(76*32, J.shape[-1])

        self.B = block_diag(*[np.array(B) for i in range(76)])

        self.vincl = vincl
        self.vincl_flatten = vincl.T.flatten()

        # The Fenics forward operator does not include the measurement operator B 
        # THESE TWO LEAD TO THE SAME RESULT. 
        if version2:
            BJ = self.B[self.vincl_flatten,:] @ J 
            self.BJ = BJ
        else:
            BJ = self.B @ J
            self.BJ = BJ[self.vincl_flatten,:]

        self.sigma_background = 0.745 # the Jacobian was constructed with this background


        # load mesh information
        self.coordinates = np.load(os.path.join(base_path,f"mesh_coordinates_{mesh_name}.npy"))
        self.cells = np.load(os.path.join(base_path,f"cells_{mesh_name}.npy"))

        # compute centroids of triangles 
        pos = [[(self.coordinates[self.cells[i,0],0] + self.coordinates[self.cells[i,1],0] + self.coordinates[self.cells[i,2],0])/3.
                ,(self.coordinates[self.cells[i,0],1] + self.coordinates[self.cells[i,1],1] + self.coordinates[self.cells[i,2],1])/3.] for i in range(self.cells.shape[0])]
        self.pos = np.array(pos)


    def reconstruct(self, Uel, alpha_tv, alpha_sm, alpha_lm):

        Uel = np.array(Uel)
        
        deltaU = Uel - np.array(self.Uref)
        deltaU = deltaU[self.vincl_flatten]

        # construct InvGamma_n
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        GammaInv = 1./var_meas # L.T L
        GammaInv = np.diag(GammaInv[:,0])


        JGJ = self.BJ.T @ GammaInv @ self.BJ
        b = self.BJ.T  @ GammaInv @ deltaU

        A = JGJ + alpha_tv*self.Rtv + alpha_sm*self.Rsm + alpha_lm * np.diag(np.diag(JGJ))

        delta_sigma = np.linalg.solve(A,b)
            
        return delta_sigma

    def reconstruct_list(self, Uel, alpha_list):
        """
        alpha_list: should be a list of three tuples [ [alpha_tv, alpha_sm, alpha_lm], 
                                                       [alpha_tv, alpha_sm, alpha_lm],
                                                       [alpha_tv, alpha_sm, alpha_lm], ...   ]

        This method is useful is several different reconstructions for one measurements are needed as
        the noise matrix GammaInv, JGJ and b only have to be constructed once. 
        """

        Uel = np.array(Uel)
        
        deltaU = Uel - np.array(self.Uref)
        deltaU = deltaU[self.vincl_flatten]

        # construct InvGamma_n
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        GammaInv = 1./var_meas # L.T L
        GammaInv = np.diag(GammaInv[:,0])

        JGJ = self.BJ.T @ GammaInv @ self.BJ
        b = self.BJ.T  @ GammaInv @ deltaU

        delta_sigma_list = []
        for alphas in alpha_list:
            A = JGJ + alphas[0]*self.Rtv + alphas[1]*self.Rsm + alphas[2] * np.diag(np.diag(JGJ)) 

            delta_sigma = np.linalg.solve(A,b)
            delta_sigma_list.append(delta_sigma)

        return delta_sigma_list

    def reconstruct_from_deltaU(self, deltaU, alphas):
        deltaU = deltaU[self.vincl_flatten]

        # construct InvGamma_n
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        GammaInv = 1./var_meas # L.T L
        GammaInv = np.diag(GammaInv[:,0])

        JGJ = self.BJ.T @ GammaInv @ self.BJ
        b = self.BJ.T  @ GammaInv @ deltaU

        A = JGJ + alphas[0]*self.Rtv + alphas[1]*self.Rsm + alphas[2] * np.diag(np.diag(JGJ)) 

        delta_sigma = np.linalg.solve(A,b)

        return delta_sigma

    def interpolate_to_image(self, sigma):

        pixwidth = 0.23 / 256
        # pixcenter_x = np.arange(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixwidth)
        pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
        pixcenter_y = pixcenter_x
        X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))
        
        interp = LinearNDInterpolator(self.pos, sigma, fill_value=0)
        
        sigma_grid = interp(pixcenters)
        
        return np.flipud(sigma_grid.reshape(256, 256))