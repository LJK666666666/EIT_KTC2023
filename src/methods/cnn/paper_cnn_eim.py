"""
论文中的CNN-EIM网络架构实现
基于论文: Image reconstruction for electrical impedance tomography
based on spatial invariant feature maps and convolutional neural network
"""
import torch
import torch.nn as nn


class PaperCNNEIM(nn.Module):
    """
    论文中提出的CNN-EIM网络架构
    输入: 1×16×16 EIM
    输出: 1×64×64 重建图像

    网络结构：
    - 编码器: 5个卷积层（使用PReLU激活）
    - 解码器: 4个反卷积层（使用LeakyReLU激活）
    - 无Skip Connection
    """

    def __init__(self):
        super(PaperCNNEIM, self).__init__()

        # 编码器（下采样路径）
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 64 (16×16)
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 8×8

            # Layer 2: 64 -> 128 (8×8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 4×4

            # Layer 3: 128 -> 256 (4×4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 2×2

            # Layer 4: 256 -> 256 (2×2)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            # Layer 5: 256 -> 256 (2×2)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        # 解码器（上采样路径）
        self.decoder = nn.Sequential(
            # Layer 1: 256 -> 128 (2×2 -> 4×4)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 2: 128 -> 64 (4×4 -> 8×8)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 3: 64 -> 32 (8×8 -> 16×16)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 4: 32 -> 1 (16×16 -> 32×32)
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, 1, 16, 16] EIM输入
        Returns:
            out: [B, 1, 32, 32] 重建图像
        """
        # 编码
        encoded = self.encoder(x)  # [B, 256, 2, 2]

        # 解码
        decoded = self.decoder(encoded)  # [B, 1, 32, 32]

        return decoded


class ImprovedPaperCNNEIM(nn.Module):
    """
    改进版本：支持输出 64×64 或 128×128
    通过调整最后一层反卷积的参数实现不同输出尺寸
    """

    def __init__(self, output_size=64):
        super(ImprovedPaperCNNEIM, self).__init__()
        self.output_size = output_size

        # 编码器（下采样路径）- 保持不变
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 64 (16×16 -> 8×8)
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2: 64 -> 128 (8×8 -> 4×4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: 128 -> 256 (4×4 -> 2×2)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4: 256 -> 256 (2×2)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            # Layer 5: 256 -> 256 (2×2)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        # 解码器（上采样路径）- 根据 output_size 选择最后一层的参数
        if output_size == 64:
            # 输出 64×64: kernel=2, stride=2
            final_layer_kernel = 2
            final_layer_stride = 2
        elif output_size == 128:
            # 输出 128×128: kernel=4, stride=4
            # 计算: (32-1)*4 + 4 = 128
            final_layer_kernel = 4
            final_layer_stride = 4
        else:
            raise ValueError(f"Unsupported output_size: {output_size}. Only 64 and 128 are supported.")

        self.decoder = nn.Sequential(
            # Layer 1: 256 -> 256 (2×2 -> 4×4)
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 2: 256 -> 128 (4×4 -> 8×8)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 3: 128 -> 64 (8×8 -> 16×16)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 4: 64 -> 32 (16×16 -> 32×32)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),

            # Layer 5: 32 -> 1 (32×32 -> output_size×output_size)
            nn.ConvTranspose2d(32, 1, kernel_size=final_layer_kernel, stride=final_layer_stride, padding=0),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, 1, 16, 16] EIM输入
        Returns:
            out: [B, 1, output_size, output_size] 重建图像
        """
        encoded = self.encoder(x)  # [B, 256, 2, 2]
        decoded = self.decoder(encoded)  # [B, 1, output_size, output_size]
        return decoded


def create_cnn_eim(config):
    """
    根据配置创建CNN-EIM模型

    Args:
        config: 配置字典，可以包含 'output_size' 参数
               - 'output_size': 32 (原论文), 64 (改进版本), 128 (直接输出)

    Returns:
        模型实例
    """
    output_size = config.get('model', {}).get('output_size', 64)

    if output_size == 32:
        return PaperCNNEIM()
    elif output_size in [64, 128]:
        return ImprovedPaperCNNEIM(output_size=output_size)
    else:
        raise ValueError(f"Unsupported output_size: {output_size}. Supported: 32, 64, 128")


if __name__ == '__main__':
    # 测试模型
    import torch

    # 测试原论文版本（32×32输出）
    model32 = PaperCNNEIM()
    x = torch.randn(4, 1, 16, 16)
    y32 = model32(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状 (原论文): {y32.shape}")
    print(f"参数数量 (原论文): {sum(p.numel() for p in model32.parameters()):,}")

    # 测试改进版本（64×64输出）
    model64 = ImprovedPaperCNNEIM()
    y64 = model64(x)
    print(f"\n输出形状 (改进版): {y64.shape}")
    print(f"参数数量 (改进版): {sum(p.numel() for p in model64.parameters()):,}")

    # 对比通用UNet的参数数量
    print(f"\n对比：")
    print(f"- 论文模型处理16×16输入，输出32×32或64×64")
    print(f"- 参数量相对较少")
    print(f"- 专门针对EIM格式优化")
