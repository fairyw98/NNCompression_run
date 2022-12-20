import torch.nn as nn
import torch

def Quant(detach_x,quant_bits):
    _x = detach_x
    for _ in range(_x.shape[1]):
        _max=torch.max(_x[:,_])
        _min=torch.min(_x[:,_])         
        code_book=torch.round((_x[:,_]-_min)*quant_bits/(_max-_min))
        _x[:,_]=_min+code_book*((_max-_min)/quant_bits)
    return _x


class Unit(nn.Module):
    def __init__(self, c2 = 1, quant_bits = 1, coder_channels = 8,en_stride = 1):
        super().__init__()
        if quant_bits > 0:
            self.quantization=int(pow(2,quant_bits)-1)
        else:
            self.quantization = -1
        coder_channels = coder_channels
        en_stride =en_stride

        self.EnCoder = nn.Conv2d(in_channels=c2, out_channels= coder_channels, kernel_size=3,stride = en_stride)
        self.DeCoder = nn.Conv2d(in_channels=coder_channels, out_channels=c2, kernel_size=3,stride=1)

    def forward(self,x):
        factor = self._get_upsample_factor(x)
        upsample = nn.Upsample(size = factor, mode='bicubic')

        # Device
        x = self.EnCoder(x)
        if self.quantization > 0:
            x.data = Quant(detach_x=x.detach(),quant_bits=self.quantization)

        # Cloud
        x = upsample(x)
        x = self.DeCoder(x)

        return x

    def _get_upsample_factor(self,x):
        with torch.no_grad():
            w_out = x.shape[2]
            h_out = x.shape[3]
            De_stride = 1
            De_kernel_size = 3
            De_padding = 0
            w_factor = (w_out-1)*De_stride-2*De_padding+De_kernel_size
            h_factor = (h_out-1)*De_stride-2*De_padding+De_kernel_size
        return (w_factor,h_factor)

class sw_Unit(nn.Module):
    def __init__(self, c2 = 1, quant_bits = 1, coder_channels = 8,en_stride = 1):
        super().__init__()

        self.coder_channels = coder_channels
        self.en_stride =en_stride

        # self.c=[8,4,16,32]
        # self.e=[1,2]
        self.param = [[8,1],[8,2],[4,1],[16,1],[32,2],[32,1]]
        _conv2d=[]

        # for i in self.c:
        #     for j in self.e:
        #         _conv2d.append((i,j))

        for item in self.param:
            _conv2d.append(item)
        self.sw_unit = nn.ModuleList([
            Unit(c2 = c2, quant_bits = quant_bits, coder_channels = _coder_channels,en_stride = _en_stride)
            for _coder_channels,_en_stride in _conv2d])
        
    def forward(self,x):
        idx = int(self.param.index([self.coder_channels,self.en_stride]))
        x = self.sw_unit[idx](x)

        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, partition_id = 3, quant_bits = -2, coder_channels = 32,en_stride = 1):
        super(AlexNet, self).__init__()
        
        self.partition_id = partition_id

        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        cfg = self._cal_feature_cfg()
        # print(cfg)
        in_channels = cfg[self.partition_id]['in_channels']

        if quant_bits == -2:
            fn = sw_Unit
        else:
            fn = Unit
        self.unit = fn(c2=in_channels,coder_channels=coder_channels,en_stride=en_stride,quant_bits=quant_bits)
        
        if init_weights:
            self._initialize_weights()
        

    def forward(self, x):
        # Device
        x = self.features[0:self.partition_id](x)
        x = self.unit(x)

        x = self.features[self.partition_id:](x)

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _cal_feature_cfg(self):
        cfg = []
        for name, layer in self.named_modules():
            if name == 'features':
                for f_name,f_layer in layer.named_children():
                    if isinstance(f_layer, nn.ReLU):
                        cfg.append('R')
                    elif isinstance(f_layer, nn.Conv2d):
                        # print(len(cfg))
                        cfg.append({'in_channels':f_layer.in_channels,'out_channels':f_layer.out_channels,'kernel_size': f_layer.kernel_size,'stride':f_layer.stride,'padding':f_layer.padding})
                    elif isinstance(f_layer,nn.MaxPool2d):
                        cfg.append('M')
        # print(cfg)
        return cfg

if __name__ == "__main__":
    data = torch.randn(5,3,224,224)
    model = AlexNet()
    model.apply(lambda m: setattr(m, 'coder_channels', 32))
    print(model)
    print(model.coder_channels)
    out = model(data)
    print(out.shape)
