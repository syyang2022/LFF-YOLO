from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53
from nets.chostnet import ghostnet
from nets.shufflenet import shufflenetv2

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y

class ARFFE(nn.Module):
    def __init__(self, rates, channel):
        super(ARFFE, self).__init__()
        self.ch = channel
        modules = []
        for rate in rates:
            modules.append(ARFFEBlock(self.ch, self.ch, rate))
        self.convs = nn.ModuleList(modules)
        self.channel_change = nn.Sequential(
            nn.Conv2d(3 * self.ch, self.ch, 1, bias=False),
            nn.BatchNorm2d(self.ch),
            nn.ReLU()
        )
        self.se_block = SEBlock(self.ch)
    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.channel_change(res)
        res = self.se_block(res)
        res = res + x
        return res

class ARFFEBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, rate):
        super(ARFFEBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_mid_layers(filters_list, in_filters, out_filters, upSample=True):
    if upSample:
        m = nn.Sequential(
            conv2d(in_filters, filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], out_filters, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    else:
        m = nn.Sequential(
            conv2d(in_filters, filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], out_filters, 1),
        )
    return m

def make_down_layer(in_ch, mid_ch, out_ch):
    m = nn.Sequential(
        conv2d(in_ch, mid_ch, 1),
        conv2d(mid_ch, out_ch, 3),
        conv2d(out_ch, mid_ch, 1),
        conv2d(mid_ch, out_ch, 3)
    )
    return m
def make_last_layers(ch_in, ch_mid, ch_out):
    m = nn.Sequential(
        conv2d(ch_in, ch_mid, 3),
        nn.Conv2d(ch_mid, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()

        #self.backbone = darknet53()
        self.backbone = shufflenetv2()
        #self.backbone = ghostnet()
        if pretrained:
            self.backbone.load_state_dict(torch.load(""))


        out_filters = self.backbone.layers_out_filters

        self.up_sample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')

        #104,104,64
        self.down_sample0 = conv2d(64, 128, 3, 2)
        self.down_sample0_out = make_down_layer(128, 64, 128)
        self.se0 = SEBlock(64)
        self.last_layer0 = make_last_layers(128, 256, len(anchors_mask[2]) * (num_classes + 5))

        #52,52,128
        self.down_sample1 = conv2d(128, 256, 3, 2)
        self.down_sample1_out = make_down_layer(256, 128, 256)
        self.se1 = SEBlock(128)
        self.last_layer1 = make_last_layers(256, 512, len(anchors_mask[1]) * (num_classes + 5))

        self.down_sample2 = conv2d(256, 512, 3, 2)
        self.down_sample2_out = make_down_layer(512, 256, 512)
        self.se2 = SEBlock(256)
        self.last_layer2 = make_last_layers(512, 1024, len(anchors_mask[0]) * (num_classes + 5))


    def forward(self, x):
        # ------------------------------------------------------------------------#
        #   x3,x2,x1,x0分别对应backbone 104,104,128; 52,52,256; 26,26,512; 13,13,1024特征图
        # ------------------------------------------------------------------------#
        x3, x2, x1, x0 = self.backbone(x)

        x0 = x0.view(x0.shape[0], x0.shape[1]//2, 2, x0.shape[2], x0.shape[3])
        x0 = torch.mean(x0,dim=2)
        x0 = self.up_sample0(x0)
        cat0 = torch.cat([x0, x1], 1)

        cat0 = cat0.view(cat0.shape[0], cat0.shape[1]//4, 4, cat0.shape[2], cat0.shape[3])
        cat0 = torch.mean(cat0,dim=2)
        cat0 = self.up_sample1(cat0)
        cat1 = torch.cat([cat0, x2], 1)

        cat2 = cat1.view(cat1.shape[0], cat1.shape[1] // 4, 4, cat1.shape[2], cat1.shape[3])
        cat2 = torch.mean(cat2, dim=2)
        cat2 = self.up_sample2(cat2)
        cat_out = torch.cat([cat2, x3], 1)
        cat_out = cat_out.view(cat_out.shape[0], cat_out.shape[1] // 4, 4, cat_out.shape[2], cat_out.shape[3])
        cat_out = torch.mean(cat_out, dim=2)

        out0_branch = self.se0(cat_out)
        out0_branch = self.down_sample0(out0_branch)
        out0_branch = self.down_sample0_out(out0_branch)
        out0 = self.last_layer0(out0_branch)

        out1_branch = self.se1(out0_branch)
        out1_branch = self.down_sample1(out1_branch)
        out1_branch = self.down_sample1_out(out1_branch)
        out1 = self.last_layer1(out1_branch)

        out2_branch = self.se2(out1_branch)
        out2_branch = self.down_sample2(out1_branch)
        out2_branch = self.down_sample2_out(out2_branch)
        out2 = self.last_layer2(out2_branch)

        return out2, out1, out0