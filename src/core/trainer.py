"""
统一训练器模块
支持所有重建方法的训练和验证
"""
from typing import Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseReconstructionMethod
from .evaluator import EITEvaluator


class UnifiedTrainer:
    """统一训练器"""

    def __init__(
        self,
        method: BaseReconstructionMethod,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        result_dir: Optional[str] = None
    ):
        """
        Args:
            method: 重建方法实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            result_dir: 结果保存目录
        """
        self.method = method
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 训练参数
        self.num_epochs = config.get('num_epochs', 100)
        self.save_every = config.get('save_every', 10)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 结果目录
        if result_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            method_name = config.get('method_name', 'unknown')
            result_dir = f"results/{method_name}_{timestamp}"
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(self.result_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # 评估器
        self.evaluator = EITEvaluator()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'val_metrics': []
        }

        # 优化器和调度器
        self.optimizer = method.get_optimizer()
        self.scheduler = method.get_scheduler(self.optimizer) if self.optimizer else None

        # 恢复训练相关
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"Resuming training from checkpoint: {checkpoint_path}")

        # 加载模型、优化器和调度器状态
        checkpoint_info = self.method.load_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        self.start_epoch = checkpoint_info['epoch'] + 1  # 从下一个epoch开始

        print(f"Resumed from epoch {checkpoint_info['epoch']}")
        print(f"Will start training from epoch {self.start_epoch}")

        # 尝试加载训练历史
        history_path = Path(checkpoint_path).parent / 'history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                saved_history = json.load(f)

            # 截断历史到当前epoch
            for key in self.history.keys():
                if key in saved_history:
                    self.history[key] = saved_history[key][:self.start_epoch]

            # 找到历史中的最佳验证损失
            if self.history['val_loss']:
                self.best_val_loss = min(self.history['val_loss'])
                print(f"Best validation loss from history: {self.best_val_loss:.6f}")
        else:
            print("Warning: No training history found, starting fresh history")

    def train(self):
        """执行训练"""
        if self.start_epoch > 0:
            print(f"Resuming training from epoch {self.start_epoch + 1}")
        print(f"Training for {self.num_epochs} epochs")
        print(f"Results will be saved to: {self.result_dir}")

        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 训练阶段
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)

            # 验证阶段
            val_loss, val_metrics = self._validate_epoch()
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # 记录学习率
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'].append(current_lr)

            # 打印信息
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if val_metrics:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                print(f"Val Metrics: {metrics_str}")
            if self.optimizer:
                print(f"Learning Rate: {current_lr:.6e}")

            # 更新学习率调度器
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.method.save_checkpoint(
                    str(self.result_dir / 'best_model.pth'),
                    epoch,
                    self.optimizer.state_dict() if self.optimizer else None,
                    self.scheduler.state_dict() if self.scheduler else None
                )
                print(f"Saved best model (val_loss: {val_loss:.6f})")

            # 定期保存检查点
            if (epoch + 1) % self.save_every == 0:
                self.method.save_checkpoint(
                    str(self.result_dir / f'checkpoint_epoch_{epoch + 1}.pth'),
                    epoch,
                    self.optimizer.state_dict() if self.optimizer else None,
                    self.scheduler.state_dict() if self.scheduler else None
                )

            # 保存训练历史
            self._save_history()

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
        return self.history

    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.method.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # 训练步骤
            loss_dict = self.method.train_step(batch)
            loss = loss_dict['loss']

            # 反向传播和优化
            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward() if isinstance(loss, torch.Tensor) else None
                self.optimizer.step()

            # 累计损失
            batch_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            total_loss += batch_loss
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': batch_loss})

        return total_loss / num_batches

    def _validate_epoch(self) -> tuple:
        """验证一个 epoch"""
        self.method.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                # 验证步骤
                loss_dict = self.method.val_step(batch)
                loss = loss_dict['loss']

                # 累计损失
                batch_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
                total_loss += batch_loss
                num_batches += 1

                # 计算评估指标（如果有真实值）
                measurements, target = batch
                if target is not None:
                    measurements = measurements.to(self.device)
                    target = target.to(self.device)

                    pred = self.method.inference(measurements)

                    # 调整target尺寸以匹配pred（处理不同输出尺寸的模型）
                    if target.shape[-1] != pred.shape[-1]:
                        target = torch.nn.functional.interpolate(
                            target,
                            size=(pred.shape[-1], pred.shape[-2]),
                            mode='bilinear',
                            align_corners=False
                        )

                    metrics = self.evaluator.compute_all_metrics(pred, target)
                    all_metrics.append(metrics)

                # 更新进度条
                pbar.set_postfix({'loss': batch_loss})

        avg_loss = total_loss / num_batches
        avg_metrics = self.evaluator.aggregate_metrics(all_metrics) if all_metrics else {}

        return avg_loss, avg_metrics

    def _save_history(self):
        """保存训练历史"""
        with open(self.result_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
