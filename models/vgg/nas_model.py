import torch.nn as nn
import torch
import sys
sys.path.append('../../')
from  utils.utils import Quant

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, profile = False, init_weights=False,partition_id = 5, quant_bits = 3, coder_cfg = {'coder_channels':32,'en_stride':1}):
        super(VGG, self).__init__()
        self.partition_id = partition_id
        self.quantization=int(pow(2,quant_bits)-1)
        self.coder_cfg = coder_cfg

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        cfg = self._cal_feature_cfg()
        if profile:
            self.profile_partition_info(cfg)

        in_channels = cfg[self.partition_id]['in_channels']

        coder_channels = self.coder_cfg['coder_channels']
        en_stride = self.coder_cfg['en_stride']

        self.EnCoder = nn.Conv2d(in_channels=in_channels, out_channels= coder_channels, kernel_size=3,stride = en_stride)
        self.DeCoder = nn.Conv2d(in_channels=coder_channels, out_channels=in_channels, kernel_size=3,stride=1)
        
        factor = self._get_upsample_factor()
        self.upsample = nn.Upsample(size = factor, mode='bicubic')

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        # x = self.features(x)
        # Device
        x = self.features[0:self.partition_id](x)
        x = self.EnCoder(x)
        x.data = Quant(detach_x=x.detach(),quant_bits=self.quantization)
        # print(x.data.shape)
        # Cloud
        x = self.upsample(x)
        x = self.DeCoder(x)
        x = self.features[self.partition_id:](x)

        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x
    
    def _cal_feature_cfg(self):
        cfg = []
        for name, layer in self.named_modules():
            if name == 'features':
                for f_name,f_layer in layer.named_children():
                # for f_name,f_layer in layer.named_modules():
                    # print(f_name)
                    if isinstance(f_layer, nn.ReLU):
                        cfg.append('R')
                    elif isinstance(f_layer, nn.Conv2d):
                        cfg.append({'in_channels':f_layer.in_channels,'out_channels':f_layer.out_channels,'kernel_size': f_layer.kernel_size,'stride':f_layer.stride,'padding':f_layer.padding})
                    elif isinstance(f_layer,nn.MaxPool2d):
                        cfg.append('M')

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

    def profile_partition_info(self,cfg):
        partition_info = []
        for index,data in enumerate(cfg):    
            if isinstance(data,dict):
                partition_info.append(index)
        print(f'partition_id = {partition_info}')
        return partition_info

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model

if __name__ == "__main__":
    img = torch.randn(5, 3, 224, 224)
    model = vgg(profile = True)
    output = model(img)
    print(output.shape)
    with open('vgg.txt','w') as f:
        f.write(str(model))