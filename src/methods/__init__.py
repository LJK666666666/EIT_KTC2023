"""
Methods 模块初始化
提供方法工厂函数
"""
from typing import Dict
from ..core.base import BaseReconstructionMethod


def create_method(method_type: str, config: Dict) -> BaseReconstructionMethod:
    """
    创建重建方法实例（工厂模式）

    Args:
        method_type: 方法类型 ('cnn', 'diffusion', 'traditional')
        config: 配置字典

    Returns:
        重建方法实例

    Raises:
        ValueError: 如果方法类型不支持
    """
    if method_type == 'cnn':
        from .cnn import create_cnn_method
        return create_cnn_method(config)

    elif method_type == 'diffusion':
        from .diffusion import create_diffusion_method
        return create_diffusion_method(config)

    elif method_type == 'traditional':
        from .traditional import create_traditional_method
        return create_traditional_method(config)

    else:
        raise ValueError(f"Unknown method type: {method_type}")


__all__ = ['create_method']
