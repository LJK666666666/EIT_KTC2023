"""
扩散模型辅助函数
从 diffusion_core 导入并创建扩散模型
"""
from .diffusion_core.gaussian_diffusion import GaussianDiffusion
from .diffusion_core.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from .diffusion_core.respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing="",
    noise_schedule="linear",
    num_diffusion_timesteps=1000,
    use_kl=False,
    predict_xstart=False,
    learn_sigma=False,
    rescale_learned_sigmas=False,
):
    """
    创建扩散模型实例

    Args:
        timestep_respacing: 时间步重采样字符串
        noise_schedule: 噪声调度类型
        num_diffusion_timesteps: 扩散时间步数
        use_kl: 是否使用 KL 散度损失
        predict_xstart: 是否预测 x_0
        learn_sigma: 是否学习方差
        rescale_learned_sigmas: 是否重缩放学习的方差

    Returns:
        GaussianDiffusion 或 SpacedDiffusion 实例
    """
    betas = get_named_beta_schedule(noise_schedule, num_diffusion_timesteps)

    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if predict_xstart:
        model_mean_type = ModelMeanType.START_X
    else:
        model_mean_type = ModelMeanType.EPSILON

    if learn_sigma:
        model_var_type = ModelVarType.LEARNED_RANGE
    else:
        model_var_type = ModelVarType.FIXED_SMALL

    if not timestep_respacing:
        timestep_respacing = [num_diffusion_timesteps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(num_diffusion_timesteps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
    )
