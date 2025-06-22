import torch
from torch import nn

import gc
import logging

from networks.Modules import Stem, GroupedDilationBlock
from networks.LightningNet import LightningNet
from networks.Downsampler import DownSampler

class GlimmerNet(LightningNet):
    r""" GlimmerNet
    """
    def __init__(self, 
                net_kwargs: dict,
                criterion: nn.Module,
                optim_kwargs:dict,
                ):
        super(GlimmerNet, self).__init__(net_kwargs, criterion, optim_kwargs)
        logging.info(f"Building GlimmerNet with {net_kwargs}")

        self.input_channels = net_kwargs["input_channels"]
        self.output_classes = net_kwargs["output_classes"]
        self.depths = net_kwargs["depths"]
        self.widths = net_kwargs["widths"]
        self.dilations = net_kwargs["dilations"]
        self.poolings = net_kwargs["poolings"]
        self.net_modules = net_kwargs["modules"]
        self.reduction = net_kwargs["stem_reduction"]
        self.resolution = net_kwargs["resolution"]
        
        assert len(self.depths) == len(self.widths) == len(self.dilations), "depths, dilations and widths must have the same length"

        self.stages = nn.ModuleList()
        self.stages.append(Stem(self.resolution, self.input_channels, self.widths[0], reduction=self.reduction))
        curr_resolution = self.stages[0].get_output_resolution()
        prev_channel_dim = self.widths[0]
        for i in range(len(self.depths)):
            hidden_channels = self.widths[i]
            out_channels = self.widths[i + 1] if i < len(self.depths) - 1 else self.widths[i]
            
            prev_channel_dim = self.widths[i]
            self.stages.append(GroupedDilationStage(self.net_modules[i],
                                                    curr_resolution,
                                                    prev_channel_dim,
                                                    hidden_channels,
                                                    out_channels,
                                                    self.depths[i],
                                                    self.dilations[i],
                                                    self.poolings[i]))
            curr_resolution = self.stages[i + 1].downsampler.get_output_resolution()

        self.backbone = nn.Sequential(*self.stages)

        self.refiner = nn.Sequential(
            nn.Conv2d(prev_channel_dim, prev_channel_dim, kernel_size=3, stride=1, padding=1, groups=prev_channel_dim),
            nn.BatchNorm2d(prev_channel_dim),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(prev_channel_dim, self.output_classes)
        )

        del self.stages
        gc.collect()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.refiner(x)
        x = self.classifier(x)
        
        return x

class GroupedDilationStage(nn.Module):
    def __init__(self, module: nn.Module, resolution: int, in_channels: int, hidden_channels: int, out_channels: int, depth: int, dilations: list, pooling: nn.Module=None) -> None:
        super(GroupedDilationStage, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            cur_in_channels = in_channels if i == 0 else hidden_channels
            self.layers.append(module(cur_in_channels,
                                      hidden_channels,
                                      resolution,
                                      kernel_size=3,
                                      stride=1,
                                      dilations=dilations))

        self.stage = nn.Sequential(*self.layers)
        self.downsampler = DownSampler(resolution, in_channels, hidden_channels, out_channels, len(dilations), kernel_size=2, stride=2, pooling=pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage(x)
        out = self.downsampler(out, dense_x=x)
        return out

def create_glimmernet(net_kwargs: dict, criterion: nn.Module, optim_kwargs:dict, ckpt_path: str = None):
    net_kwargs['modules'] = [GroupedDilationBlock, GroupedDilationBlock, GroupedDilationBlock, GroupedDilationBlock]
    net_kwargs['depths'] = [4, 4, 4, 1]
    net_kwargs['widths'] = [40, 80, 160, 240]
    net_kwargs['dilations'] = [[1, 2, 2, 3], [1, 2, 2, 3], [1, 2, 2, 3], [1, 2, 2, 3]]
    net_kwargs['poolings'] = [nn.MaxPool2d, nn.MaxPool2d, nn.MaxPool2d, nn.AvgPool2d]

    if ckpt_path is not None:
        model = GlimmerNet.load_from_checkpoint(ckpt_path, net_kwargs=net_kwargs, criterion=criterion, optim_kwargs=optim_kwargs)
    else:
        model = GlimmerNet(net_kwargs, criterion, optim_kwargs)

    return model