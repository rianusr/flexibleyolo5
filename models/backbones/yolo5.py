import os
import sys
import math

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__).replace('models/backbones', ''))
from utils.torch_utils import model_summary
from models.common import Conv, C3, SPPF

YOLO5_VARIANT = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'l': {'depth_multiple': 1.0,  'width_multiple': 1.0},
        'x': {'depth_multiple': 1.33, 'width_multiple': 1.25}
    }


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class YoloV5Backbone(nn.Module):
    def __init__(self, variant='s'):  # model, input channels, number of classes
        super(YoloV5Backbone, self).__init__()
        self.variant_params = YOLO5_VARIANT[variant]
        gd, gw = self.variant_params['depth_multiple'], self.variant_params['width_multiple']

        # layers
        base_channel = math.ceil(64 * gw / 8) * 8   # 32
        
        ##! backbone!
        # backbone_0 = Focus(3, base_channel, 3)
        backbone_0 = Conv(3, base_channel, 6, 2, 2)
        backbone_1 = Conv(base_channel, base_channel * 2, 3, 2)
        backbone_2 = C3(base_channel * 2, base_channel * 2, max(round(3 * gd), 1))
        backbone_3 = Conv(base_channel * 2, base_channel * 4, 3, 2)
        backbone_4 = C3(base_channel * 4, base_channel * 4, max(round(6 * gd), 1))
        backbone_5 = Conv(base_channel * 4, base_channel * 8, 3, 2)
        backbone_6 = C3(base_channel * 8, base_channel * 8, max(round(9 * gd), 1))
        backbone_7 = Conv(base_channel * 8, base_channel * 16, 3, 2)
        backbone_8 = C3(base_channel * 16, base_channel * 16, max(round(3 * gd), 1))
        backbone_9 = SPPF(base_channel * 16, base_channel * 16)
        ## build backbone
        bkbo_layers = [backbone_0, backbone_1, backbone_2, backbone_3, backbone_4, backbone_5, backbone_6, backbone_7, backbone_8, backbone_9]
        self.backbone = nn.Sequential(*bkbo_layers)

        # Init weights, biases
        # initialize_weights(self)

    def forward(self, x):
        fpn_feats = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == 4:
                fpn_feats.append(x)
            elif idx == 6:
                fpn_feats.append(x)
            elif idx == 9:
                fpn_feats.append(x)
            # print(x.shape)
        return fpn_feats


if __name__ == '__main__':
    # Create model
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).cuda()
    model = YoloV5Backbone(variant='s')

    model = model.cuda()
    model.train()
    fpn_feats = model(img)
    print(*[v.shape for v in fpn_feats], sep='\n')
    model_summary(model, 640)