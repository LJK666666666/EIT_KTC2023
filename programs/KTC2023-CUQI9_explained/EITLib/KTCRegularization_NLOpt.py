#%%
import numpy as np
import matplotlib.pyplot as plt

class SMPrior:
    def __init__(self, ginv, corrlength, var, mean, covariancetype=None):
        self.corrlength = corrlength
        self.mean = mean
        self.c = 1e-9  # default value
        if covariancetype is not None:
            self.covariancetype = covariancetype
        else:
            self.covariancetype = 'Squared Distance'  # default
        self.compute_L(ginv, corrlength, var)

    def compute_L(self, g, corrlength, var):
        ng = g.shape[0]
        a = var - self.c
        b = np.sqrt(-corrlength**2 / (2 * np.log(0.01)))
        Gamma_pr = np.zeros((ng, ng))

        for ii in range(ng):
            for jj in range(ii, ng):
                dist_ij = np.linalg.norm(g[ii, :] - g[jj, :])
                if self.covariancetype == 'Squared Distance':
                    gamma_ij = a * np.exp(-dist_ij**2 / (2 * b**2))
                elif self.covariancetype == 'Ornstein-Uhlenbeck':
                    gamma_ij = a * np.exp(-dist_ij / corrlength)
                else:
                    raise ValueError('Unrecognized prior covariance type')
                if ii == jj:
                    gamma_ij = gamma_ij + self.c
                Gamma_pr[ii, jj] = gamma_ij
                Gamma_pr[jj, ii] = gamma_ij
        

        self.L = np.linalg.cholesky(np.linalg.inv(Gamma_pr)).T

    def draw_samples(self, nsamples):
        samples = self.mean + np.linalg.solve(self.L, np.random.randn(self.L.shape[0], nsamples))
        return samples

    def eval_fun(self, args):
        sigma = args[0]
        res = 0.5 * np.linalg.norm(self.L @ (sigma - self.mean))**2
        return res
    
    def evaluate_target_external(self, x, compute_grad=False):
        x = x.reshape((-1,1))
        # print("x.shape: ", x.shape)
        # print("self.mean.shape: ", self.mean.shape)
        if compute_grad:
            grad = self.L.T @ self.L @ (x - self.mean)
        else:
            grad = None
        
        return self.eval_fun(x), grad
        

    def compute_hess_and_grad(self, args, nparam):
        sigma = args[0]
        Hess = self.L.T @ self.L
        grad = Hess @ (sigma - self.mean)

        if nparam > len(sigma):
            Hess = np.block([[Hess, np.zeros((len(sigma), nparam - len(sigma)))],
                             [np.zeros((nparam - len(sigma), len(sigma))), np.zeros((nparam - len(sigma), nparam - len(sigma)))]])
            grad = np.concatenate([grad, np.zeros(nparam - len(sigma))])

        return Hess, grad


if __name__ ==  '__main__':
    from utils import EITFenics, create_disk_mesh
    from dolfin import *
    import pickle
    L = 32
    F = 25
    n = 300
    radius = 0.115
    mesh = create_disk_mesh(radius, n, F)
    myeit = EITFenics(mesh, L, background_conductivity=0.8)
    H = FunctionSpace(myeit.mesh, 'CG', 1)

    plot(myeit.mesh)

    v2d = vertex_to_dof_map(H)
    d2v = dof_to_vertex_map(H)

    sigma0 = 0.8*np.ones((myeit.mesh.num_vertices(), 1)) #linearization point
    corrlength =  radius#* 0.115 #used in the prior
    var_sigma = 0.05 ** 2 #prior variance
    mean_sigma = sigma0
    smprior = SMPrior(myeit.mesh.coordinates()[d2v], corrlength, var_sigma, mean_sigma)
    

    sample = smprior.draw_samples(1)
    fun = Function(H)
    fun.vector().set_local(sample)
    im = plot(fun)
    plt.colorbar(im)

    mesh_file =XDMFFile('mesh_file_'+str(L)+'_'+str(n)+'.xdmf')
    mesh_file.write(myeit.mesh)
    mesh_file.close()
    #%%
    file = open('smprior_'+str(L)+'_'+str(n)+'.p', 'wb')
    pickle.dump(smprior, file)
    file.close()








# %%
