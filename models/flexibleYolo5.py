import os
import sys
import math
import json

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(__file__).replace('models', ''))
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, copy_attr, save_load_state_dict
from utils.autoanchor import check_anchor_order

from models.common import Conv, AutoShape
from models.backbones import *
from models.heads import YoloV5Head, Detect

__all_backbones__ = ['yolo5', 'resnext', 'coatnet', 'convnext', 'uniformer', 'swin_transformer', 'swin_mlp']
__all_heads__ = ['yolo5']


ANCHORS = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]


class FlexibleYolo5(nn.Module):
    def __init__(self, bkbo_variant='yolo5-x', head_variant='x', nc=80, input_size=640, inplace=True):  # model, input channels, number of classes
        super(FlexibleYolo5, self).__init__()
        self.input_size = input_size
        self.anchors = ANCHORS
        self.inplace = inplace
        
        #! build backbone
        self.backbone = self._get_backbone(bkbo_variant, input_size)
        in_fpn_feats = self._bkbo_forward_once()      ## backbone forward once to get feat_feats
        
        #! build head & detect
        if 'yolo5' in head_variant.lower():
            self.head = YoloV5Head(variant=head_variant.split('-')[-1], nc=nc, in_fpn_feats=in_fpn_feats, anchors=self.anchors)
            #! set stride and anchors
            self._build_stride_and_anchors()
            self.stride = self.head.model[-1].stride
        else:
            raise ValueError(f'Only {__all_heads__} are accepted!')

        # Init weights, biases
        initialize_weights(self)
        
    def forward(self, imgs, labels=None):
        bkbo_fpn_feats = self.backbone(imgs)
        if labels is None:
            x = self.head(bkbo_fpn_feats)
        else:
            x = self.head(bkbo_fpn_feats, labels=labels, imgs=imgs)
        return x

    def _get_backbone(self, bkbo_variant, input_size):
        if 'yolo5' in bkbo_variant.lower():
            backbone = YoloV5Backbone(variant=bkbo_variant.split('-')[-1])
        elif 'coatnet' in bkbo_variant.lower():
            backbone = CoAtNetBackbone(variant=bkbo_variant.split('-')[-1], image_size=(input_size, input_size))
        elif 'resnext' in bkbo_variant.lower():
            backbone = ResNeXtBackbone(variant=bkbo_variant.split('-')[-1])
        elif 'convnext' in bkbo_variant.lower():
            backbone = ConvNeXtBackbone(variant=bkbo_variant.split('-')[-1])
        elif 'uniformer' in bkbo_variant.lower():
            backbone = UniFormerBackbone(variant=bkbo_variant.split('-')[-1], img_size=input_size)
        elif 'swin_transformer' in bkbo_variant.lower():
            backbone = SwinTransformerBackbone(variant=bkbo_variant.split('-')[-1])
        elif 'swin_mlp' in bkbo_variant.lower():
            backbone = SwinMLPBackbone(variant=bkbo_variant.split('-')[-1])
        else:
            raise ValueError(f'Only {__all_backbones__} are accepted!')
        return backbone
    
    def _bkbo_forward_once(self):
        x = torch.rand((1, 3, self.input_size, self.input_size))
        bkbo_fpn_feats = self.backbone(x)
        
        assert bkbo_fpn_feats and len(bkbo_fpn_feats) == 3, '''
            fpn_shapes: feats for fpn layer, it contains 3 stages: p1, p2, p3, for example:
            fpn_shapes = {
                'p1':(b, 128, 80, 80),     # 'p1'  (batch_size, channel, width, height)
                'p2':(b, 256, 40, 40),     # 'p2'
                'p3':(b, 512, 20, 20),     # 'p3'
            }
        '''
        return bkbo_fpn_feats

    def _build_stride_and_anchors(self):
        detect_layer = self.head.model[-1]
        s = self.input_size
        detect_layer.inplace = self.inplace
        detect_layer.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))])  # forward
        detect_layer.anchors /= detect_layer.stride.view(-1, 1, 1)
        self.stride = detect_layer.stride
        check_anchor_order(detect_layer)
        self._initialize_biases(detect_layer)  # only run once

    def _initialize_biases(self, detect_layer, cf=None):
        for mi, s in zip(detect_layer.m, detect_layer.stride):  # from
            b = mi.bias.view(detect_layer.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (detect_layer.nc - 0.999999))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if hasattr(self.head, 'head'):
            m = self.head.model[-1]  # Detect()
            if isinstance(m, Detect):
                m.stride = fn(m.stride)
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
        return self
    
    def _show_variant(self):
        print(json.dumps(self.variant_params, ensure_ascii=False, indent=4))

def build_model_2(num_classes, input_size, bkbo_variant, head_variant, hyp, device, pretrained='', freeze=[]):
    flexibleYolo = FlexibleYolo5(bkbo_variant=bkbo_variant, head_variant=head_variant, nc=num_classes, input_size=input_size)
    model = flexibleYolo.model.to(device)
    
    ckpt = None
    if pretrained:
        ckpt = torch.load(pretrained, map_location='cpu')  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        exclude = ['anchor'] if hyp.get('anchors') else []  # exclude keys
        ckpt_state_dict = ckpt['model'].float().state_dict()  # to FP32
        model = save_load_state_dict(model, ckpt_state_dict)
        
    # Freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    return model, ckpt


def build_model(num_classes, input_size, bkbo_variant, head_variant, hyp, device, pretrained='', freeze=[]):
    model = FlexibleYolo5(bkbo_variant=bkbo_variant, head_variant=head_variant, nc=num_classes, input_size=input_size)
    model.backbone = model.backbone.to(device)
    model.head = model.head.to(device)
    
    ckpt = None
    if pretrained:
        ckpt = torch.load(pretrained, map_location='cpu')  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        exclude = ['anchor'] if hyp.get('anchors') else []  # exclude keys
        ckpt_state_dict = ckpt['model'].float().state_dict()  # to FP32
        model = save_load_state_dict(model, ckpt_state_dict)

    # Freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    return model, ckpt


if __name__ == '__main__':
    variant = 'yolo5-s'
    # Create model
    model = FlexibleYolo5(bkbo_variant=variant, head_variant=variant, nc=80, input_size=640)
    # print('\n==============================\n')
    # model_state_dict = model.state_dict()
    # for idx, key in enumerate(model_state_dict):
    #     print(f"{idx:3d}    {key}\t{model_state_dict[key].shape}".expandtabs(64))
    # exit()
    # model.info(verbose=True)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model = model.cuda()
    # model.train()
    # img = torch.rand(2 if torch.cuda.is_available() else 1, 3, 640, 640).cuda()
    # y = model(img)
    # for r in y:
    #     print(type(r))
    #     print(r.shape)
