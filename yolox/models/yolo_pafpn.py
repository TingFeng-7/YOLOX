#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOPAFPN_hbo(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise,
                                   act=act, out_features=in_features)
        self.in_features = in_features
        # self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # order: from up to bottom
        self.fpn_convs = nn.ModuleList([])
        self.fpn_convs2 = nn.ModuleList([])  # the other conv to reduce channels more

        self.pa_bu_convs = nn.ModuleList([])
        self.pa_convs = nn.ModuleList([])

        in_channels = in_channels[::-1]
        for i in range(len(in_channels)-1):
            fpn_conv2_i = BaseConv(
                int(in_channels[i] * width), int(in_channels[i+1] * width), 1, 1, act=act
            )
            self.fpn_convs2.append(fpn_conv2_i)

            fpn_conv_i = CSPLayer(
                int(2 * in_channels[i+1] * width),
                int(in_channels[i+1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )  # cat
            self.fpn_convs.append(fpn_conv_i)

            pa_bu_conv_i = Conv(
                int(in_channels[i+1] * width), int(in_channels[i+1] * width), 3, 2, act=act
            )
            self.pa_bu_convs.append(pa_bu_conv_i)

            pa_conv_i = CSPLayer(
                int(2 * in_channels[i+1] * width),
                int(in_channels[i] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )
            self.pa_convs.append(pa_conv_i)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        # order: from up to bottom
        features = features[::-1]

        fpn_outs = [None for i in features]
        pan_outs = [None for i in features]

        ''' fpn: up to bottom '''
        for i, f_i in enumerate(features):
            if i == 0:
                fpn_out_i = self.fpn_convs2[i](f_i)
            else:
                mid_f_i = self.upsample(fpn_outs[i-1])
                mid_f_i = torch.cat([mid_f_i, f_i], 1)
                fpn_out_i = self.fpn_convs[i-1](mid_f_i)
                if i != len(features) - 1:
                    fpn_out_i = self.fpn_convs2[i](fpn_out_i)
            fpn_outs[i] = fpn_out_i

        ''' pan: bottom to up '''
        for i, fpn_out_i in list(enumerate(fpn_outs))[::-1]:
            if i == len(fpn_outs) - 1:
                pan_out_i = fpn_out_i
            else:
                mid_f_i = self.pa_bu_convs[i](pan_outs[i+1])
                mid_f_i = torch.cat([mid_f_i, fpn_out_i], 1)
                pan_out_i = self.pa_convs[i](mid_f_i)
            pan_outs[i] = pan_out_i
        return pan_outs[::-1]
