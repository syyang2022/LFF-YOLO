import math
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#

# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes):
#         super(BasicBlock, self).__init__()
#
#         self.inplanes = inplanes // 2
#         self.planes0 = planes[0] // 2
#         self.planes1 = planes[1] // 2
#         branch_main = [
#             # pw
#             nn.Conv2d(self.inplanes, self.planes0, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(self.planes0),
#             nn.ReLU(inplace=True),
#             # dw
#             nn.Conv2d(self.planes0, self.planes0, 3, 1, 1, groups=self.planes0, bias=False),
#             nn.BatchNorm2d(self.planes0),
#             # pw-linear
#             nn.Conv2d(self.planes0, self.planes1, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(self.planes1),
#             nn.ReLU(inplace=True),
#         ]
#         self.branch_main = nn.Sequential(*branch_main)
#
#     def forward(self, old_x):
#         x_proj, x = self.channel_shuffle(old_x)
#         return torch.cat((x_proj, self.branch_main(x)), 1)
#
#     def channel_shuffle(self, x):
#         batchsize, num_channels, height, width = x.data.size()
#         assert (num_channels % 4 == 0)
#         x = x.reshape(batchsize * num_channels // 2, 2, height * width)
#         x = x.permute(1, 0, 2)
#         x = x.reshape(2, -1, num_channels // 2, height, width)
#         return x[0], x[1]

#特征融合模块

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

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

# 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Rspp(nn.Module):
    def __init__(self, rates, channel):
        super(Rspp, self).__init__()
        self.ch = channel
        modules = []
        for rate in rates:
            modules.append(RsppBlock(self.ch, self.ch, rate))
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

class RsppBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, rate):
        super(RsppBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)


        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        #out = self.ca(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32

        self.rates1 = [3, 9, 13]
        #self.rates2 = [1, 3, 5]
        #self.Rspp1 = Rspp(self.rates1, 32)
        #self.Rspp2 = Rspp(self.rates1, 64)
        #self.Rspp3 = Rspp(self.rates2, 128)
        #self.se1 = SEBlock(64)
        #self.Se2 = SEBlock(512)
        #self.Se3 = SEBlock(1024)
        #self.aspp1 = ASPP(64,self.rates1,64)

        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])

        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])

        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])

        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])

        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        #x = self.aspp1(x)
        #x = self.se1(x)

        x = self.layer2(x)
        #x = self.Rspp3(x)

        out3 = self.layer3(x)

        out4 = self.layer4(out3)

        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
