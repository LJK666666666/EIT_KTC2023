from typing import Optional, Any, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor

from src.diffusion import SDE
from src.third_party_models import OpenAiUNetModel

def Ancestral_Sampling(
    score: OpenAiUNetModel, 
    sde: SDE, 
    x: Tensor,
    time_step: Union[Tensor, Tuple[Tensor,Tensor]],
    step_size: float,
    cond_inp: Tensor) -> Tuple[Tensor, Tensor]:

    """
    Implements the ancestral sampling used for DPS in the discrete DDPM framework.

    We are using the formulation of 
        "Diffusion Posterior Sampling for General Noisy Inverse Problems" (2023) 
    Algortithm 1, however with a fixed standard deviation sigma_i

    """    
    t = time_step[0]
    tminus1 = time_step[1]

    s = score(x, t)

    xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=t)

    std_t = sde.marginal_prob_std(t=t)[:, None, None, None] # sqrt(1 - alphabr)
    alpha_t = sde.alphas[int(t[0].item())]

    x_mean = 1/torch.sqrt(alpha_t)*(x - (1 - alpha_t)/std_t*s)

    noise = torch.sqrt(1 - alpha_t)*torch.randn_like(x)
    
    x = x_mean + noise # Algo.1  in 3. line 6

    return x.detach(), xhat0.detach()


def ddim(
    sde: SDE,
    s: Tensor,
    xhat: Tensor,
    time_step: Union[Tensor, Tuple[Tensor,Tensor]],
    step_size: Tensor, 
    eta: float, 
    use_simplified_eqn: bool = False
    ) -> Tensor:
    
    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    tminus1 = time_step-step_size if not isinstance(time_step,Tuple) else time_step[1]
    std_t = sde.marginal_prob_std(t=t)[:, None, None, None]
    
    mean_tminus1 = sde.marginal_prob_mean(t=tminus1)[:, None, None, None]
    mean_t = sde.marginal_prob_mean(t=t)[:, None, None, None]
    tbeta = ((1 - mean_tminus1.pow(2)) / ( 1 - mean_t.pow(2) ) ).pow(.5) * (1 - mean_t.pow(2) * mean_tminus1.pow(-2) ).pow(.5)
    if any(tbeta.isnan()): tbeta = torch.zeros(*tbeta.shape, device=s.device)
    xhat = xhat*mean_tminus1
    eps_ = s
    noise_deterministic = torch.sqrt( 1 - mean_tminus1.pow(2) - tbeta.pow(2)*eta**2 )*eps_
    noise_stochastic = eta*tbeta*torch.randn_like(xhat)
    
    return xhat + noise_deterministic + noise_stochastic

def apTweedy(s: Tensor, x: Tensor, sde: SDE, time_step:Tensor) -> Tensor:

    div = sde.marginal_prob_mean(time_step)[:, None, None, None].pow(-1)
    std_t = sde.marginal_prob_std(time_step)[:, None, None, None]

    update = x - s*std_t

    return update*div


def _check_times(times, t_0, num_steps):

    assert times[0] > times[1], (times[0], times[1])

    assert times[-1] == -1, times[-1]

    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= num_steps, (t, num_steps)

def _schedule_jump(num_steps, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t-1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)
    _check_times(time_steps, -1, num_steps)

    return time_steps

def wrapper_ddim(
    score: OpenAiUNetModel, 
    sde: SDE, 
    x: Tensor, 
    time_step: Tensor, 
    step_size: Tensor, 
    cond_inp: Tensor,
    eta: float) -> Tuple[Tensor, Tensor]:

    t = time_step if not isinstance(time_step, Tuple) else time_step[0]
    with torch.no_grad():
        s_inp = torch.cat([x, cond_inp], dim=1)
        s = score(s_inp, t).detach()
        xhat0 = apTweedy(s=s, x=x, sde=sde, time_step=t)
    # setting ``eta'' equals to ``0'' turns ddim into ddpm
    x = ddim(sde=sde, s=s, xhat=xhat0, time_step=time_step, step_size=step_size, eta=eta, use_simplified_eqn=False)
    
    return x.detach(), xhat0.detach()