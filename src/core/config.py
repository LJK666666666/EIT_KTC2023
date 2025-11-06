"""
配置管理模块
负责加载和管理 YAML 配置文件
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载 YAML 配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        保存配置到 YAML 文件

        Args:
            config: 配置字典
            save_path: 保存路径
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个配置字典（override_config 会覆盖 base_config）

        Args:
            base_config: 基础配置
            override_config: 覆盖配置

        Returns:
            合并后的配置
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
        """
        验证配置是否包含必需的键

        Args:
            config: 配置字典
            required_keys: 必需的键列表（支持嵌套，如 'model.input_dim'）

        Returns:
            是否有效
        """
        for key in required_keys:
            keys = key.split('.')
            current = config

            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[k]

        return True

    @staticmethod
    def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        获取嵌套配置值

        Args:
            config: 配置字典
            key: 键（支持点分隔的嵌套键，如 'model.hidden_dim'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        current = config

        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return default
            current = current[k]

        return current

    @staticmethod
    def update_nested_value(config: Dict[str, Any], key: str, value: Any):
        """
        更新嵌套配置值

        Args:
            config: 配置字典
            key: 键（支持点分隔的嵌套键）
            value: 新值
        """
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
