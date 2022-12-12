import torch
import torch.nn as nn

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

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=416, n_class=1000, model_size='2.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.rate1 = [1, 3, 5]

        self.rspp0 = ARFFE(self.rate1, 128)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            #self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
            self.stage_out_channels = [-1, 128, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.stage0 = nn.Sequential(*features[:4])
        self.stage1 = nn.Sequential(*features[4:12])
        self.stage2 = nn.Sequential(*features[12:16])
        self.layers_out_filters = [64, 128, 256, 512, 1024]



        # self.conv_last = nn.Sequential(
        #     nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(self.stage_out_channels[-1]),
        #     nn.ReLU(inplace=True)
        # )
        # self.globalpool = nn.AvgPool2d(7)
        # if self.model_size == '2.0x':
        #     self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.rspp0(x)
        out0 = self.maxpool(x)
        out1 = self.stage0(out0)
        #out1_rspp = self.rspp1(out1)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        # x = self.conv_last(x)
        #
        # x = self.globalpool(x)
        # if self.model_size == '2.0x':
        #     x = self.dropout(x)
        # x = x.contiguous().view(-1, self.stage_out_channels[-1])
        # x = self.classifier(x)
        return out0, out1, out2, out3

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def shufflenetv2():
    return ShuffleNetV2()

if __name__ == "__main__":
    model = ShuffleNetV2()
    # print(model)
    # model.eval()
    # # print(model)
    # input = torch.randn(32, 3, 416, 416)
    # y = model(input)
    # print(y)

