'''
Based on the variance exploding (VE) and variance presenrving (VP) SDE.
The derivations are given in [https://arxiv.org/pdf/2011.13456.pdf] Appendix C.
Based on: https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
'''
import torch
import numpy as np 
import abc


class SDE(abc.ABC):
	"""
	SDE abstract class. Functions are designed for a mini-batch of inputs.
	"""
	def __init__(self):
		"""
		Construct an SDE.
		"""
		super().__init__()

	def diffusion_coeff(self, t):
		"""
		Outputs f
		"""
		pass

	def sde(self, x, t):
		"""
		Outputs f and G
		"""
		pass

	def marginal_prob(self, x, t):
		"""
		Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
		"""
		pass

	def marginal_prob_std(self, t):
		pass 

	def marginal_prob_mean(self, t):
		"""
		Outputs the scaling factor of mean of p_{0t}(x(t)|x(0)) (for VE-SDE and VP-SDE the mean is a scaled x(0))
		"""
		pass 

	def prior_sampling(self, shape):
		"""
		Generate one sample from the prior distribution, $p_T(x)$.
		"""
		pass

class DDPM(SDE):
	def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, num_steps: int = 1000):
		super().__init__()
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.num_steps = num_steps
		self.betas = torch.from_numpy(np.linspace( 
				self.beta_min, self.beta_max, self.num_steps, dtype=np.float64)
			) # uses fp64 for accuracy
		assert len(self.betas.shape) == 1, 'betas must be 1-D'
		assert (self.betas > 0).all() and (self.betas <= 1).all()
		self.alphas = 1.0 - self.betas

	def _compute_alpha_cumprod(self, t):
		betas = torch.cat([torch.zeros(1), self.betas], dim=0)
		return (1 - betas.to(t.device)).cumprod(dim=0).index_select(0, t.long() + 1).to(torch.float32)
	
	def diffusion_coeff(self, ):
		raise NotImplementedError

	def sde(self, ):
		raise NotImplementedError

	def marginal_prob(self, x, t):
		return x * self.marginal_prob_mean(t=t)[:, None, None, None], self.marginal_prob_std(t=t)

	def marginal_prob_std(self, t):
		bar_a = self._compute_alpha_cumprod(t=t)
		return (1. - bar_a).pow(.5)

	def marginal_prob_mean(self, t):
		bar_a = self._compute_alpha_cumprod(t=t)
		return bar_a.pow(.5)
		
	def prior_sampling(self, shape):
		return torch.randn(*shape) 
