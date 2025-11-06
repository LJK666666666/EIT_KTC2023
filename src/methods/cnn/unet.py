"""
UNet 模型实现
用于 EIT 重建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积层块：(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块：MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块：Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()

        # 使用双线性插值或转置卷积进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 使用 1x1 卷积来减少通道数
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            # concat 后的通道数是 in_channels // 2 + in_channels // 2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            # 转置卷积同时进行上采样和通道减少
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # concat 后的通道数是 in_channels // 2 + in_channels // 2 = in_channels
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if hasattr(self, 'conv_reduce'):
            x1 = self.conv_reduce(x1)

        # 处理输入大小不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接 skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet 架构

    Args:
        input_channels: 输入通道数
        output_channels: 输出通道数
        base_channels: 第一层的通道数
        num_layers: U-Net 的层数（深度）
        bilinear: 是否使用双线性插值上采样（否则使用转置卷积）
        dropout: Dropout 率
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        base_channels=64,
        num_layers=4,
        bilinear=True,
        dropout=0.0
    ):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.bilinear = bilinear

        # 输入层
        self.inc = DoubleConv(input_channels, base_channels, dropout=dropout)

        # 下采样路径
        self.down_layers = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_layers):
            out_ch = in_ch * 2
            self.down_layers.append(Down(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch

        # 上采样路径
        self.up_layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = in_ch // 2
            self.up_layers.append(Up(in_ch, out_ch, bilinear, dropout=dropout))
            in_ch = out_ch

        # 输出层
        self.outc = OutConv(base_channels, output_channels)

    def forward(self, x):
        # 下采样路径，保存每层的输出用于 skip connections
        x1 = self.inc(x)
        skip_connections = [x1]

        x = x1
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)

        # 上采样路径，使用 skip connections
        skip_connections = skip_connections[:-1]  # 移除最后一层（已经在 x 中）

        for up in self.up_layers:
            skip = skip_connections.pop()
            x = up(x, skip)

        # 输出
        logits = self.outc(x)
        return logits


def create_unet(config):
    """
    根据配置创建 UNet 模型

    Args:
        config: 配置字典

    Returns:
        UNet 模型实例
    """
    model_config = config.get('model', {})

    return UNet(
        input_channels=model_config.get('input_channels', 1),
        output_channels=model_config.get('output_channels', 1),
        base_channels=model_config.get('base_channels', 64),
        num_layers=model_config.get('num_layers', 4),
        bilinear=model_config.get('bilinear', True),
        dropout=model_config.get('dropout', 0.0)
    )


if __name__ == '__main__':
    # 测试 UNet
    model = UNet(input_channels=1, output_channels=1, base_channels=64, num_layers=4)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape should match input shape"
    print("UNet test passed!")
