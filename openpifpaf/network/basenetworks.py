import logging
import torch

LOG = logging.getLogger(__name__)
from .conv import conv, conv_dw, conv_dw_no_bn
import torch.nn as nn


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, input_output_scale, out_features):
        super(BaseNetwork, self).__init__()

        self.net = net
        self.shortname = shortname
        self.input_output_scale = input_output_scale
        self.out_features = out_features

        # print(list(net.children()))
        LOG.info('stride = %d', self.input_output_scale)
        LOG.info('output features = %d', self.out_features)

    def forward(self, *args):
        return self.net(*args)

class MobileNetFactory(torch.nn.Module):
    def __init__(self, out_features, input_output_scale, num_heatmaps = 170, num_pafs = 380):
        super().__init__()
        self.out_features = out_features
        self.input_output_scale = input_output_scale
        self.model = torch.nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, out_features)   # conv5_5
        )

        num_channels = 512
        nettype = 'cpmaff'
        
        
        num_refinement_stages = 1

        self.cpm = Cpm(out_features, num_channels)
        self.initial_stage = CPMInitialStage(num_channels, num_heatmaps, num_pafs, nettype)
        self.refinement_stages = nn.ModuleList()

        intermediate_channel = num_channels + num_heatmaps + num_pafs
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(CPMRefinementStage(intermediate_channel, num_channels,
                                                          num_heatmaps, num_pafs, nettype))  
        self.num_channels = [num_heatmaps, num_pafs]
        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs


    def forward(self, x):
        x = self.model(x)
        backbone_features = self.cpm(x)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                    refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output[-2], stages_output[-1]

class ShuffleNetV2Factory(object):
    def __init__(self, torchvision_shufflenetv2):
        self.torchvision_shufflenetv2 = torchvision_shufflenetv2

    def blocks(self):
        return [
            self.torchvision_shufflenetv2.conv1,
            # self.torchvision_shufflenetv2.maxpool,
            self.torchvision_shufflenetv2.stage2,
            self.torchvision_shufflenetv2.stage3,
            self.torchvision_shufflenetv2.stage4,
            self.torchvision_shufflenetv2.conv5,
        ]


class DownsampleCat(torch.nn.Module):
    def __init__(self):
        super(DownsampleCat, self).__init__()
        self.pad = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.0)

    def forward(self, x):  # pylint: disable=arguments-differ
        p = self.pad(x)
        o = torch.cat((p[:, :, :-1:2, :-1:2], p[:, :, 1::2, 1::2]), dim=1)
        return o


class ResnetBlocks(object):
    def __init__(self, resnet):
        self.modules = list(resnet.children())
        LOG.debug('modules = %s', self.modules)

    def input_block(self, use_pool=False, conv_stride=2, pool_stride=2):
        modules = self.modules[:4]

        if not use_pool:
            modules.pop(3)
        else:
            if pool_stride != 2:
                modules[3].stride = torch.nn.modules.utils._pair(pool_stride)  # pylint: disable=protected-access

        if conv_stride != 2:
            modules[0].stride = torch.nn.modules.utils._pair(conv_stride)  # pylint: disable=protected-access

        return torch.nn.Sequential(*modules)

    @staticmethod
    def stride(block):
        """Compute the output stride of a block.

        Assume that convolutions are in serious with pools; only one
        convolutions with non-unit stride.
        """
        if isinstance(block, list):
            stride = 1
            for b in block:
                stride *= ResnetBlocks.stride(b)
            return stride

        conv_stride = max(m.stride[0]
                          for m in block.modules()
                          if isinstance(m, torch.nn.Conv2d))

        pool_stride = 1
        pools = [m for m in block.modules() if isinstance(m, torch.nn.MaxPool2d)]
        if pools:
            for p in pools:
                pool_stride *= p.stride

        return conv_stride * pool_stride

    @staticmethod
    def replace_downsample(block):
        print('!!!!!!!!!!')
        first_bottleneck = block[0]
        print(first_bottleneck.downsample)
        first_bottleneck.downsample = DownsampleCat()
        print(first_bottleneck)

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]

#cpm aff
class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class CPMInitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs, nettype="cpmaff"):
        # type: 'cpm' for joint detectionm 'aff' for alignment, 'cpmaff' for both stages
        super().__init__()
        self.nettype = nettype
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        if 'cpm' in nettype:
            self.heatmaps = nn.Sequential(
                conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
                conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
            )
        if 'aff' in nettype:
            self.pafs = nn.Sequential(
                conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
                conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
            )

    def forward(self, x):
        trunk_features = self.trunk(x)
        if 'cpm' == self.nettype:
            heatmaps = self.heatmaps(trunk_features)
            return [heatmaps]
        if 'aff' == self.nettype:
            pafs = self.pafs(trunk_features)
            return [pafs]
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class CPMRefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class CPMRefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs, nettype="cpmaff"):
        super().__init__()
        self.nettype = nettype
        self.trunk = nn.Sequential(
            CPMRefinementStageBlock(in_channels, out_channels),
            CPMRefinementStageBlock(out_channels, out_channels),
            CPMRefinementStageBlock(out_channels, out_channels),
            CPMRefinementStageBlock(out_channels, out_channels),
            CPMRefinementStageBlock(out_channels, out_channels)
        )
        if 'cpm' in nettype:
            self.heatmaps = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
                conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
            )
        if 'aff' in nettype:
            self.pafs = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
                conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
            )

    def forward(self, x):
        trunk_features = self.trunk(x)
        if 'cpm' == self.nettype:
            heatmaps = self.heatmaps(trunk_features)
            return [heatmaps]
        if 'aff' == self.nettype:
            pafs = self.pafs(trunk_features)
            return [pafs]
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]