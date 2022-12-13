from torch import nn
import torch
from pathlib import Path
import sys
sys.path.append('../../')
from  utils.utils import Quant


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
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
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8,partition_id = 1, quant_bits = 3, coder_cfg = {'coder_channels':32,'en_stride':1,'de_stride':1}):
        super(MobileNetV2, self).__init__()
        self.partition_id = partition_id
        self.quantization=int(pow(2,quant_bits)-1)
        self.coder_cfg = coder_cfg
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

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
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
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
        cfg = self._cal_feature_cfg()
        # print(cfg)
        # in_channels = cfg[self.partition_id]['in_channels']

        # coder_channels = self.coder_cfg['coder_channels']
        # en_stride = self.coder_cfg['en_stride']

        # self.EnCoder = nn.Conv2d(in_channels=in_channels, out_channels= coder_channels, kernel_size=3,stride = en_stride)
        # self.DeCoder = nn.Conv2d(in_channels=coder_channels, out_channels=in_channels, kernel_size=3,stride=1)
        
        # factor = self._get_upsample_factor()
        # self.upsample = nn.Upsample(size = factor, mode='bicubic')

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
        x = self.features(x)
        # # Device
        # x = self.features[0:self.partition_id](x)
        # x = self.EnCoder(x)
        # x.data = Quant(detach_x=x.detach(),quant_bits=self.quantization)

        # # Cloud
        # x = self.upsample(x)
        # x = self.DeCoder(x)
        # x = self.features[self.partition_id:](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _cal_feature_cfg(self):
        cfg = []
        for name, layer in self.named_modules():
            if name == 'features':
                # for f_name,f_layer in layer.named_children():
                for f_name,f_layer in layer.named_modules():
                    # print(f_name)
                    if isinstance(f_layer, nn.ReLU):
                        cfg.append('R')
                    elif isinstance(f_layer, nn.Conv2d):
                        cfg.append({'in_channels':f_layer.in_channels,'out_channels':f_layer.out_channels,'kernel_size': f_layer.kernel_size,'stride':f_layer.stride,'padding':f_layer.padding})
                    elif isinstance(f_layer,nn.MaxPool2d):
                        cfg.append('M')
        # print(cfg)
        return cfg
    def _get_upsample_factor(self):
        x = torch.randn(5,3,224,224)
        with torch.no_grad():
            x = self.features[0:self.partition_id](x)
            factor = x.shape[2]
            De_stride = 1
            De_kernel_size = 3
            factor = factor*De_stride-1+De_kernel_size
        return (factor,factor)

if __name__ == "__main__":
    img = torch.randn(5, 3, 224, 224)
    model = MobileNetV2()
    output = model(img)
    print(output.shape)
    with open('model_v2.txt','w') as f:
        f.write(str(model))