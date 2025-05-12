import torch
from torch import nn

class Stem(nn.Module):
    def __init__(self, resolution:int, in_channels: int, out_channels: int, reduction:int=1) -> None:
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.resolution = resolution

        self.stride1 = 2 if reduction % 2 == 0 else 1
        self.stride2 = 2 if reduction % 4 == 0 else 1

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.stride2, padding=2, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def get_output_resolution(self) -> int:
        return (((self.resolution - 1) // self.stride1) + 1) // self.stride2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x

class GroupedDilationBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 kernel_size,
                 stride,
                 dilations):
        super(GroupedDilationBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.groups = len(self.dilations)

        assert self.in_channels % self.groups == 0
        assert self.out_channels % self.groups == 0
        self.group_size = self.in_channels // self.groups
        self.out_group_size = self.out_channels // self.groups

        convs = []
        for d in self.dilations:
            convs.append(nn.Conv2d(self.group_size,
                                   self.out_group_size,
                                   kernel_size,
                                   padding='same',
                                   dilation=d,
                                   groups=self.group_size if self.group_size==self.out_group_size else 1))
        self.convs = nn.ModuleList(convs)

        self.skip_conn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU6()

    def forward(self, x):
        skip = self.skip_conn(x)

        x_reshaped = x.view(x.shape[0],
                            self.groups,
                            self.group_size,
                            x.shape[2],
                            x.shape[3])
        out_shape = list(x.shape)
        out_shape[1] = self.out_channels
        out = torch.empty(out_shape, device=x.device)
        for i, conv in enumerate(self.convs):
            out[:, i * self.group_size:(i + 1) * self.group_size] = conv(x_reshaped[:, i])

        out = self.activation(self.bn(out))
        out = out + skip

        return out