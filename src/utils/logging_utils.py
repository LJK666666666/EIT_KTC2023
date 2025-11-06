"""
日志记录工具模块
"""
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """日志记录器"""

    def __init__(
        self,
        name: str = 'EIT',
        log_dir: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Args:
            name: 日志记录器名称
            log_dir: 日志保存目录
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(
                log_path / f'{name}_{timestamp}.log',
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)

    def debug(self, message: str):
        """记录调试级别日志"""
        self.logger.debug(message)

    def critical(self, message: str):
        """记录严重错误级别日志"""
        self.logger.critical(message)


def get_logger(
    name: str = 'EIT',
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志保存目录
        level: 日志级别

    Returns:
        Logger 实例
    """
    return Logger(name, log_dir, level)
