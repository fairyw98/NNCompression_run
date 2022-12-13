from torch import nn
import torch

FLAGS_num_list = [0.03125,0.0625,0.09375,0.1250,0.15625, 0.1875, 0.21875, 0.2500, 0.5000,1.0]
FLAGS_quant_list = [1,3,15,255,65535]

def Quant(detach_x,quant_bits):
    _x = detach_x
    for _ in range(_x.shape[1]):
        _max=torch.max(_x[:,_])
        _min=torch.min(_x[:,_])         
        code_book=torch.round((_x[:,_]-_min)*quant_bits/(_max-_min))
        _x[:,_]=_min+code_book*((_max-_min)/quant_bits)
    return _x

def make_divisible(v, divisor=1, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SWBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1,us = False):
        super(SWBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)
        self.num_features_max = num_features
        a=FLAGS_num_list
        b=FLAGS_quant_list
        zzh_bn=[]

        for i in range(len(a)):
            for j in range(len(b)):
                if us:
                    zzh_bn.append(int(self.num_features_max*a[i]))
                else:
                    zzh_bn.append(int(self.num_features_max))

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=True) for i in zzh_bn])
        self.ratio = ratio
        self.width_mult = FLAGS_num_list[-1]
        self.quant_bits= FLAGS_quant_list[-1]
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = int(FLAGS_num_list.index(self.width_mult)*len(FLAGS_quant_list)+FLAGS_quant_list.index(self.quant_bits))
        y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                self.bn[idx].weight,
                self.bn[idx].bias,
                self.training,
                self.momentum,
                self.eps)
        return y


class SWConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[False, False], ratio=[1, 1]):
        super(SWConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = FLAGS_num_list[-1]
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            # nn.BatchNorm2d(out_channel),
            SWBatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class SWConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1,us = [False,False]):
        padding = (kernel_size - 1) // 2
        super(SWConvBNReLU, self).__init__(
            # nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            SWConv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False,us = us),
            # nn.BatchNorm2d(out_channel),
            SWBatchNorm2d(out_channel,us = us[1]),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channel),
            SWBatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SWInvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio,us = [False,False]):
        super(SWInvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio

        layers = []
        layers.extend([
            # 3x3 depthwise conv
            # ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            SWConvBNReLU(hidden_channel, hidden_channel, stride=stride,us = us),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # SWConv2d(hidden_channel, out_channel, kernel_size=1, bias=False,us = us),
            # nn.BatchNorm2d(out_channel),
            SWBatchNorm2d(out_channel,us = us[1]),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        self.quant_bits = FLAGS_quant_list[-1]
        block = InvertedResidual
        SWblock = SWInvertedResidual
        input_channel = make_divisible(32 * alpha, round_nearest)
        last_channel = make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        # features.append(ConvBNReLU(3, input_channel, stride=2))
        features.append(SWConvBNReLU(3, input_channel, stride=2,us = [False,True]))
        # building inverted residual residual blockes
        for index,(t, c, n, s) in enumerate(inverted_residual_setting):
            if index == 0:
                output_channel = make_divisible(c * alpha, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    # features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    features.append(SWblock(input_channel, output_channel, stride, expand_ratio=t,us = [True,False]))
                    input_channel = output_channel
            else:
                output_channel = make_divisible(c * alpha, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = self.features(x)
        x = self.features[0](x)
        x.data = Quant(detach_x=x.detach(),quant_bits=self.quant_bits)
        x = self.features[1:](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    train_tmp = [[i,j] for i in FLAGS_num_list for j in FLAGS_quant_list]
    print(train_tmp)
    
    # model = MobileNetV2()
    # def f(m):
    #     if 'width_mult' in dir(m):
    #         print(m.width_mult)
    # model.apply(f)
    # for name in model.children():
    #     print(name)
    # print(model.features)
    # print(dir(model))
    # for name in model.parameters():
    #     print(name)
    # for name, m in model.named_modules():
    #     print(name)
    # for i in model.state_dict():
    #     # if i == 'width_mult':
    #     print(i)
    # print(model.width_mult)
    # img = torch.randn(5, 3, 224, 224)
    # model = MobileNetV2()
    # output = model(img)
    # print(output.shape)
    # with open('sw_model_v2.txt','a+') as f:
    #     f.write(str(model))