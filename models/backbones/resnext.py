import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(__file__).replace('models/backbones', ''))
from models.common import Conv

PARAMS = {
    'p': {'num_blocks': [3, 4, 6,  3], 'channels': [32,  64,  128, 256]},  ## pico
    'n': {'num_blocks': [3, 4, 6,  3], 'channels': [64,  128, 256, 512]},  ## nano
    'm': {'num_blocks': [3, 4, 6,  3], 'channels': [96,  192, 384, 768]},  ## micro
    't': {'num_blocks': [3, 4, 6,  3], 'channels': [128, 256, 512, 1024]}, ## tiny      ## equal to 50
    's': {'num_blocks': [3, 4, 13, 3], 'channels': [128, 256, 512, 1024]}, ## small
    'l': {'num_blocks': [3, 4, 23, 3], 'channels': [128, 256, 512, 1024]}, ## large     ## equal to 101
    'h': {'num_blocks': [3, 8, 23, 3], 'channels': [128, 256, 512, 1024]}, ## huge
    'g': {'num_blocks': [3, 8, 36, 3], 'channels': [128, 256, 512, 1024]}, ## giant     ## equal to 152
}


class BottleNeck(nn.Module):
    expansion = 1
    def __init__(self, in_channel, channel, stride=1, C=32, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, stride=1, groups=C)
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(channel, channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel*self.expansion)

        # self.relu = nn.ReLU(True)
        self.relu = nn.GELU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))  # bs,c,h,w
        out = self.relu(self.bn2(self.conv2(out)))  # bs,c,h,w
        out = self.relu(self.bn3(self.conv3(out)))  # bs,4c,h,w

        if(self.downsample != None):
            residual = self.downsample(residual)

        out += residual
        return self.relu(out)


class ResNeXtBackbone(nn.Module):
    def __init__(self, variant='s', focus_ch=32):
        super().__init__()
        
        layers_blocks = PARAMS[variant]['num_blocks']
        layers_channs = PARAMS[variant]['channels']
        focus = Conv(3, focus_ch, 6, 2, 2) if focus_ch > 3 else nn.Identity()
        self.in_channel = 64                ## net input, channel of net after stem
        stem   = self._make_stem(focus_ch)
        layer0 = self._make_layer(layers_channs[0], layers_blocks[0], stride=1)
        layer1 = self._make_layer(layers_channs[1], layers_blocks[1], stride=2)
        layer2 = self._make_layer(layers_channs[2], layers_blocks[2], stride=2)
        layer3 = self._make_layer(layers_channs[3], layers_blocks[3], stride=2)
        
        bkbo_layers = [focus, stem, layer0, layer1, layer2, layer3]
        self.backbone = nn.Sequential(*bkbo_layers)

    def forward(self, x):
        fpn_feats = []
        for idx, m in enumerate(self.backbone):
            x = m(x)
            if idx in [2, 3, 4]:        ## 取80、40、20特征图，for_yolox
                fpn_feats.append(x)
        return fpn_feats

    def _make_stem(self, focus_ch):
        conv1 = nn.Conv2d(focus_ch, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(self.in_channel)
        # relu = nn.ReLU(True)
        relu = nn.GELU()
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        return nn.Sequential(conv1, bn1, relu, maxpool)
    
    def _make_layer(self, channel, blocks, stride=1):
        downsample = None
        if(stride != 1 or self.in_channel != channel*BottleNeck.expansion):
            downsample = nn.Conv2d(self.in_channel, channel*BottleNeck.expansion, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(BottleNeck(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel*BottleNeck.expansion
        for _ in range(1, blocks):
            layers.append(BottleNeck(self.in_channel, channel))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    variants = ['p', 'n', 'm', 't', 's', 'l', 'h', 'g']
    input_size = 384 
    img = torch.randn(1, 3, input_size, input_size).cuda()
    for va in variants:
        model = ResNeXtBackbone(variant=va).cuda()
        fpn_feats = model(img)
        print(*[v.shape for v in fpn_feats], sep='\n')
        print(f'params: {count_parameters(model)/(1024*1024):.2f} M')