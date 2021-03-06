"""Head networks."""

import logging
import re

import torch
import torch.nn as nn
from .conv import conv, conv_dw, conv_dw_no_bn

LOG = logging.getLogger(__name__)


class HeadStacks(torch.nn.Module):
    def __init__(self, stacks):
        super(HeadStacks, self).__init__()
        self.stacks_by_pos = {s[0]: s for s in stacks}
        self.ignore = {head_i for s in stacks for head_i in s[1:]}

    def forward(self, *args):
        heads = args

        stacked = []
        for head_i, head in enumerate(heads):
            if head_i in self.ignore:
                continue
            if head_i not in self.stacks_by_pos:
                stacked.append(head)
                continue

            fields = [heads[si] for si in self.stacks_by_pos[head_i]]
            stacked.append([
                torch.cat(fields_by_type, dim=1)
                for fields_by_type in zip(*fields)
            ])

        return stacked


class PifHFlip(torch.nn.Module):
    def __init__(self, keypoints, hflip):
        super(PifHFlip, self).__init__()

        flip_indices = torch.LongTensor([
            keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i
            for kp_i, kp_name in enumerate(keypoints)
        ])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)


    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector component
        out[1][:, :, 0, :, :] *= -1.0

        return out


class PafHFlip(torch.nn.Module):
    def __init__(self, keypoints, skeleton, hflip):
        super(PafHFlip, self).__init__()
        skeleton_names = [
            (keypoints[j1 - 1], keypoints[j2 - 1])
            for j1, j2 in skeleton
        ]
        flipped_skeleton_names = [
            (hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2)
            for j1, j2 in skeleton_names
        ]
        LOG.debug('skeleton = %s, flipped_skeleton = %s',
                  skeleton_names, flipped_skeleton_names)

        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)

        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector components
        out[1][:, :, 0, :, :] *= -1.0
        out[2][:, :, 0, :, :] *= -1.0

        # reverse direction
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, paf_i])
            out[1][:, paf_i] = out[2][:, paf_i]
            out[2][:, paf_i] = cc

        return out

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


class InitialStage(nn.Module):
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


class RefinementStageBlock(nn.Module):
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


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs, nettype="cpmaff"):
        super().__init__()
        self.nettype = nettype
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
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

class CPMAFF(nn.Module):
    def __init__(self, input_features = 1024,num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38, nettype='cpmaff'):
        super().__init__()
        self.nettype = nettype
        self.shortname = nettype
        self.cpm = Cpm(input_features, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs, nettype)
        self.refinement_stages = nn.ModuleList()

        input_channel = num_channels
        if 'cpm' in nettype:
            input_channel += num_heatmaps
        if 'aff' in nettype:
            input_channel += num_pafs
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(input_channel, num_channels,
                                                          num_heatmaps, num_pafs, nettype))

    def forward(self, x):
        # backbone_features = self.model(x)
        backbone_features = self.cpm(x)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            if nettype == 'cpm' or nettype == 'aff':
                stages_output.extend(
                        refinement_stage(torch.cat([backbone_features, stages_output[-1]], dim=1)))
            else:
                stages_output.extend(
                    refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output



class CompositeField(torch.nn.Module):
    dropout_p = 0.0
    quad = 0

    def __init__(self, head_name, in_features, *,
                 n_fields,
                 n_confidences,
                 n_vectors,
                 n_scales,
                 kernel_size=1, padding=0, dilation=1):
        super(CompositeField, self).__init__()

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  head_name, n_fields, n_confidences, n_vectors, n_scales,
                  kernel_size, padding, dilation)

        self.shortname = head_name
        self.dilation = dilation
     
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        # classification
        out_features = n_fields * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_vectors)
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in self.reg_convs
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if not self.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]
        regs_x_spread = [torch.nn.functional.leaky_relu(x + 2.0) - 2.0
                         for x in regs_x_spread]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        # upscale
        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        # reshape regressions
        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x


def determine_nfields(head_name):
    m = re.match('p[ia]f[s]?([0-9]+)$', head_name)
    if m is not None:
        return int(m.group(1))

    return {
        'paf': 19,
        'pafs': 19,
        'pafb': 19,
        'pafsb': 19,
        'wpaf': 19,
    }.get(head_name, 17)


def determine_nvectors(head_name):
    if 'pif' in head_name:
        return 1
    if 'paf' in head_name:
        return 2
    return 0


def determine_nscales(head_name):
    if 'pif' in head_name:
        return 1
    if 'pafs' in head_name:
        return 2
    if 'paf' in head_name:
        return 0
    return 0


def factory(name, n_features):
    if name in ('pif',
                'paf',
                'pafs',
                'wpaf',
                'pafb',
                'pafsb') or \
       re.match('p[ia]f[s]?([0-9]+)$', name) is not None:
        n_fields = determine_nfields(name)
        n_vectors = determine_nvectors(name)
        n_scales = determine_nscales(name)

        LOG.info('selected head CompositeField for %s', name)
        return CompositeField(name, n_features,
                              n_fields=n_fields,
                              n_confidences=1,
                              n_vectors=n_vectors,
                              n_scales=n_scales)

    if name in ('cpm', 'aff','cpmaff'):
        return CPMAFF(input_features=n_features, num_channels=128, num_heatmaps=17, num_pafs=19, nettype=name)

    raise Exception('unknown head to create a head network: {}'.format(name))
