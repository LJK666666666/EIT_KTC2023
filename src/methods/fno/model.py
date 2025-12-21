"""
Fourier Neural Operator (2D) blocks for EIT reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(height, width))
        return x


class FNO2d(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        width: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        output_size: int = 128,
        add_coord: bool = True
    ):
        super().__init__()
        self.output_size = output_size
        self.add_coord = add_coord

        in_channels = input_channels + (2 if add_coord else 0)
        self.input_projection = nn.Conv2d(in_channels, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)
        ])
        self.pointwise_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)
        ])
        self.act = nn.GELU()

        self.output_projection = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, output_channels, kernel_size=1)
        )

    @staticmethod
    def _make_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(0, 1, height, device=device, dtype=dtype)
        x = torch.linspace(0, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_size is not None and (
            x.shape[-2] != self.output_size or x.shape[-1] != self.output_size
        ):
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False
            )

        if self.add_coord:
            grid = self._make_grid(x.shape[-2], x.shape[-1], x.device, x.dtype)
            grid = grid.expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, grid], dim=1)

        x = self.input_projection(x)

        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = self.act(spectral(x) + pointwise(x))

        x = self.output_projection(x)
        return x
