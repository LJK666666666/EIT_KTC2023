"""
DeepDbar U-Net Model (PyTorch implementation)

Based on: Hamilton & Hauptmann (2018).
Deep D-bar: Real time Electrical Impedance Tomography Imaging with
Deep Neural Networks. IEEE Transactions on Medical Imaging.

Converted from TensorFlow to PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DeepDbarUNet(nn.Module):
    """
    DeepDbar U-Net for EIT image reconstruction

    Architecture:
    - Encoder: 5 blocks with 2 conv layers each + maxpool (except last)
    - Decoder: 4 blocks with transposed conv + concat + 2 conv layers
    - Output: 1x1 conv -> ReLU

    Input size: [batch, 1, 64, 64]
    Output size: [batch, 1, 64, 64]
    """

    def __init__(self, input_channels: int = 1):
        super(DeepDbarUNet, self).__init__()

        # ========== Encoder (下采样路径) ==========

        # Block 1: 1 -> 32
        self.enc_conv1_1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.enc_conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 -> 64
        self.enc_conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.enc_conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64 -> 128
        self.enc_conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.enc_conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128 -> 256
        self.enc_conv4_1 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.enc_conv4_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ========== Bottleneck ==========
        # Block 5: 256 -> 512
        self.bottleneck_conv1 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.bottleneck_conv2 = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        # ========== Decoder (上采样路径) ==========

        # Upsampling 4: 512 -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2)  # 256 + 256 = 512 after concat
        self.dec_conv4_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)

        # Upsampling 3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2)  # 128 + 128 = 256 after concat
        self.dec_conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        # Upsampling 2: 128 -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)  # 64 + 64 = 128 after concat
        self.dec_conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        # Upsampling 1: 64 -> 32
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv1_1 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  # 32 + 32 = 64 after concat
        self.dec_conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # ========== Output layer ==========
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重和偏置
        - 权重: truncated normal with stddev=0.025
        - 偏置: constant 0.025
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # PyTorch中没有truncated_normal，使用normal_然后截断
                nn.init.normal_(m.weight, mean=0.0, std=0.025)
                # 截断到 [-2*std, 2*std]
                m.weight.data.clamp_(-0.05, 0.05)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.025)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch, 1, 64, 64]

        Returns:
            output: 重建图像 [batch, 1, 64, 64]
        """
        # ========== Encoder ==========
        # Block 1
        enc1 = F.relu(self.enc_conv1_1(x))
        enc1 = F.relu(self.enc_conv1_2(enc1))
        pool1 = self.pool1(enc1)  # [batch, 32, 32, 32]

        # Block 2
        enc2 = F.relu(self.enc_conv2_1(pool1))
        enc2 = F.relu(self.enc_conv2_2(enc2))
        pool2 = self.pool2(enc2)  # [batch, 64, 16, 16]

        # Block 3
        enc3 = F.relu(self.enc_conv3_1(pool2))
        enc3 = F.relu(self.enc_conv3_2(enc3))
        pool3 = self.pool3(enc3)  # [batch, 128, 8, 8]

        # Block 4
        enc4 = F.relu(self.enc_conv4_1(pool3))
        enc4 = F.relu(self.enc_conv4_2(enc4))
        pool4 = self.pool4(enc4)  # [batch, 256, 4, 4]

        # ========== Bottleneck ==========
        bottleneck = F.relu(self.bottleneck_conv1(pool4))
        bottleneck = F.relu(self.bottleneck_conv2(bottleneck))  # [batch, 512, 4, 4]

        # ========== Decoder ==========
        # Upsampling + Concat + Conv (Block 4)
        up4 = F.relu(self.up4(bottleneck))  # [batch, 256, 8, 8]
        concat4 = torch.cat([enc4, up4], dim=1)  # [batch, 512, 8, 8]
        dec4 = F.relu(self.dec_conv4_1(concat4))
        dec4 = F.relu(self.dec_conv4_2(dec4))  # [batch, 256, 8, 8]

        # Upsampling + Concat + Conv (Block 3)
        up3 = F.relu(self.up3(dec4))  # [batch, 128, 16, 16]
        concat3 = torch.cat([enc3, up3], dim=1)  # [batch, 256, 16, 16]
        dec3 = F.relu(self.dec_conv3_1(concat3))
        dec3 = F.relu(self.dec_conv3_2(dec3))  # [batch, 128, 16, 16]

        # Upsampling + Concat + Conv (Block 2)
        up2 = F.relu(self.up2(dec3))  # [batch, 64, 32, 32]
        concat2 = torch.cat([enc2, up2], dim=1)  # [batch, 128, 32, 32]
        dec2 = F.relu(self.dec_conv2_1(concat2))
        dec2 = F.relu(self.dec_conv2_2(dec2))  # [batch, 64, 32, 32]

        # Upsampling + Concat + Conv (Block 1)
        up1 = F.relu(self.up1(dec2))  # [batch, 32, 64, 64]
        concat1 = torch.cat([enc1, up1], dim=1)  # [batch, 64, 64, 64]
        dec1 = F.relu(self.dec_conv1_1(concat1))
        dec1 = F.relu(self.dec_conv1_2(dec1))  # [batch, 32, 64, 64]

        # ========== Output ==========
        # 注意：原始TF代码在最后有ReLU，这里保持一致
        output = self.output_conv(dec1)  # [batch, 1, 64, 64]
        output = F.relu(output)

        return output


def create_deepdbar_model(config: Dict) -> DeepDbarUNet:
    """
    创建 DeepDbar U-Net 模型

    Args:
        config: 配置字典

    Returns:
        DeepDbar U-Net 模型实例
    """
    model_config = config.get('model', {})
    input_channels = model_config.get('input_channels', 1)

    model = DeepDbarUNet(input_channels=input_channels)

    return model
