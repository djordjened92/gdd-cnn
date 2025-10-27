import torch
from torch import nn
import logging

from networks.LayerNorms import GRN

class AggDownSample(nn.Module):
    def __init__(self, resolution: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 dilations_len: int,
                 kernel_size: int,
                 stride: int,
                 pooling:nn.Module=None) -> None:
        super(AggDownSample, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations_len = dilations_len

        self.dense_channels = in_channels + hidden_channels
        self.downsampler_channels = out_channels

        self.dense_fc = nn.Conv2d(self.dense_channels, out_channels, kernel_size=1, stride=1, groups=self.dense_channels // (2 * self.dilations_len))
        self.activation = nn.ReLU6()
        self.batch_norm = nn.BatchNorm2d(self.out_channels)

        self.downsampler = None
        if pooling is None or pooling == nn.Identity:
            self.downsampler = nn.Identity()
        elif pooling == nn.Conv2d:
            self.downsampler = nn.Conv2d(self.downsampler_channels, out_channels, kernel_size=2, stride=2, groups=out_channels, bias=False)
        elif pooling == nn.MaxPool2d or pooling == nn.AvgPool2d:
            self.downsampler = pooling(kernel_size, stride, padding=kernel_size//2)
        
        self.grn = GRN(self.out_channels)

        logging.info(f"Setting DownSampler {hidden_channels} -> {out_channels} with downsampler type {type(self.downsampler)} and dense_fc {type(self.dense_fc)}")

    def get_output_resolution(self) -> int:
        if self.pooling == nn.MaxPool2d or self.pooling == nn.AvgPool2d:
            output_resolution = (self.resolution - self.kernel_size) // self.stride + 1
        else:
            raise NotImplementedError(f"Pooling layer {self.pooling} not implemented")
        return output_resolution

    def aggregate(self, x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        # Recombine channel dimension to have
        # fmaps from different dilations grouped
        B, C, H, W = x.shape
        group_size = C // self.dilations_len
        x = x.view(B, self.dilations_len, group_size, H, W)
        x.swapaxes_(1, 2)
        x = x.reshape(B, C, H, W)

        # Mix Concatenation
        b, c, h, w = x.size()
        x = torch.cat([x.view(b, -1, 1, h, w), dense_x.view(b, -1, 1, h, w)], dim=2)
        x = x.reshape(b, -1, h, w)
        x = self.activation(self.batch_norm(self.dense_fc(x)))

        return x

    def forward(self, x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        x = self.aggregate(x, dense_x)
        x = self.grn(self.downsampler(x))
        return x