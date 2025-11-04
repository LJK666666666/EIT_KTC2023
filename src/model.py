"""
EIT重建神经网络模型

从EIT测量数据重建导电率分布的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EITReconstructionNet(nn.Module):
    """
    EIT重建网络：从992维测量数据重建256x256的导电率分布

    架构：
    - 编码器：将测量数据编码为潜在表示
    - 解码器：将潜在表示解码为图像
    """

    def __init__(self, input_dim=992, output_size=256, latent_dim=512):
        super(EITReconstructionNet, self).__init__()

        self.input_dim = input_dim
        self.output_size = output_size
        self.latent_dim = latent_dim

        # 编码器：测量数据 -> 潜在向量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(2048, latent_dim)
        )

        # 解码器：潜在向量 -> 图像特征图
        # 从512维向量解码到16x16的特征图
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True)
        )

        # 上采样层：16x16 -> 256x256
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 最后的卷积层
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: 测量数据 [batch_size, 992]

        Returns:
            output: 重建的导电率分布 [batch_size, 3, 256, 256]
        """
        # 编码
        latent = self.encoder(x)  # [batch, 512]

        # 解码到特征图
        features = self.fc_decode(latent)  # [batch, 256*16*16]
        features = features.view(-1, 256, 16, 16)  # [batch, 256, 16, 16]

        # 上采样到256x256
        output = self.decoder(features)  # [batch, 3, 256, 256]

        return output


class UNetEITReconstruction(nn.Module):
    """
    基于U-Net的EIT重建网络

    使用U-Net架构处理空间信息，但输入是测量数据
    """

    def __init__(self, input_dim=992):
        super(UNetEITReconstruction, self).__init__()

        # 将测量数据投影到特征图
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64 * 64 * 64),
            nn.ReLU(inplace=True)
        )

        # 编码器
        self.enc1 = self.conv_block(64, 128)
        self.enc2 = self.conv_block(128, 256)
        self.enc3 = self.conv_block(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128)

        # 最终上采样到256x256
        self.final_upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: 测量数据 [batch_size, 992]

        Returns:
            output: 重建的导电率分布 [batch_size, 3, 256, 256]
        """
        # 投影到64x64特征图
        x = self.input_projection(x)
        x = x.view(-1, 64, 64, 64)  # [batch, 64, 64, 64]

        # 编码器
        enc1 = self.enc1(x)  # [batch, 128, 64, 64]
        x = F.max_pool2d(enc1, 2)  # [batch, 128, 32, 32]

        enc2 = self.enc2(x)  # [batch, 256, 32, 32]
        x = F.max_pool2d(enc2, 2)  # [batch, 256, 16, 16]

        enc3 = self.enc3(x)  # [batch, 512, 16, 16]
        x = F.max_pool2d(enc3, 2)  # [batch, 512, 8, 8]

        # 瓶颈层
        x = self.bottleneck(x)  # [batch, 1024, 8, 8]

        # 解码器（带跳跃连接）
        x = self.upconv3(x)  # [batch, 512, 16, 16]
        x = torch.cat([x, enc3], dim=1)  # [batch, 1024, 16, 16]
        x = self.dec3(x)  # [batch, 512, 16, 16]

        x = self.upconv2(x)  # [batch, 256, 32, 32]
        x = torch.cat([x, enc2], dim=1)  # [batch, 512, 32, 32]
        x = self.dec2(x)  # [batch, 256, 32, 32]

        x = self.upconv1(x)  # [batch, 128, 64, 64]
        x = torch.cat([x, enc1], dim=1)  # [batch, 256, 64, 64]
        x = self.dec1(x)  # [batch, 128, 64, 64]

        # 最终上采样
        x = self.final_upconv(x)  # [batch, 64, 128, 128]
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # [batch, 3, 256, 256]

        return x


class SegmentationLoss(nn.Module):
    """
    分割损失函数：结合交叉熵和Dice损失
    """

    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(SegmentationLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Dice损失

        Args:
            pred: [batch, 3, H, W] - 预测的logits
            target: [batch, H, W] - 真实标签
        """
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=3).permute(0, 3, 1, 2).float()

        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def forward(self, pred, target):
        """
        Args:
            pred: [batch, 3, H, W] - 预测的logits
            target: [batch, H, W] - 真实标签
        """
        ce_loss = self.ce_loss(pred, target.long())
        dice_loss = self.dice_loss(pred, target)

        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss

        return total_loss, ce_loss, dice_loss


def get_model(model_name='simple', input_dim=992):
    """
    获取模型实例

    Args:
        model_name: 'simple' 或 'unet'
        input_dim: 输入测量数据的维度

    Returns:
        model: 模型实例
    """
    if model_name == 'simple':
        return EITReconstructionNet(input_dim=input_dim)
    elif model_name == 'unet':
        return UNetEITReconstruction(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing EITReconstructionNet...")
    model = EITReconstructionNet()
    model = model.to(device)

    # 测试输入
    x = torch.randn(4, 992).to(device)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting UNetEITReconstruction...")
    model_unet = UNetEITReconstruction()
    model_unet = model_unet.to(device)

    output_unet = model_unet(x)
    print(f"Output shape: {output_unet.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_unet.parameters()):,}")

    # 测试损失函数
    print("\nTesting loss function...")
    target = torch.randint(0, 3, (4, 256, 256)).to(device)
    loss_fn = SegmentationLoss()
    total_loss, ce_loss, dice_loss = loss_fn(output, target)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"CE loss: {ce_loss.item():.4f}")
    print(f"Dice loss: {dice_loss.item():.4f}")
