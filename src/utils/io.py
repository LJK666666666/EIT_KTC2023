"""
输入输出工具模块
"""
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle
import scipy.io as sio
import numpy as np
import torch


def save_mat(data: Dict[str, Any], save_path: str):
    """
    保存为 MATLAB .mat 文件

    Args:
        data: 数据字典
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(save_path, data)


def load_mat(file_path: str) -> Dict[str, Any]:
    """
    加载 MATLAB .mat 文件

    Args:
        file_path: 文件路径

    Returns:
        数据字典
    """
    return sio.loadmat(file_path)


def save_pickle(data: Any, save_path: str):
    """
    保存为 pickle 文件

    Args:
        data: 数据对象
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """
    加载 pickle 文件

    Args:
        file_path: 文件路径

    Returns:
        数据对象
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, save_path: str, indent: int = 2):
    """
    保存为 JSON 文件

    Args:
        data: 数据字典
        save_path: 保存路径
        indent: 缩进空格数
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Dict:
    """
    加载 JSON 文件

    Args:
        file_path: 文件路径

    Returns:
        数据字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_numpy(data: np.ndarray, save_path: str):
    """
    保存为 numpy .npy 文件

    Args:
        data: numpy 数组
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)


def load_numpy(file_path: str) -> np.ndarray:
    """
    加载 numpy .npy 文件

    Args:
        file_path: 文件路径

    Returns:
        numpy 数组
    """
    return np.load(file_path)


def save_torch(data: torch.Tensor, save_path: str):
    """
    保存为 PyTorch .pt 文件

    Args:
        data: PyTorch tensor
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)


def load_torch(file_path: str, device: Optional[str] = None) -> torch.Tensor:
    """
    加载 PyTorch .pt 文件

    Args:
        file_path: 文件路径
        device: 目标设备

    Returns:
        PyTorch tensor
    """
    if device:
        return torch.load(file_path, map_location=device)
    return torch.load(file_path)


def ensure_dir(path: str):
    """
    确保目录存在

    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)
