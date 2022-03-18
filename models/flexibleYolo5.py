import os
import sys
import math
import json
import numpy as np
from thop import profile
from copy import deepcopy

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__).replace('models', ''))
from models.backbones import *
from models.common import Conv, AutoShape
from models.heads import YoloV5Head, Detect, YOLOXHead

from tactics.autoanchor import check_anchor_order
from tactics.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, copy_attr, save_load_state_dict

__all_backbones__ = ['yolo5', 'resnext', 'coatnet', 'convnext', 'uniformer', 'swin_transformer', 'swin_mlp']
__all_heads__ = ['yolo5', 'yolox']


ANCHORS1_0   = [[10, 13, 16, 30], None, None]
ANCHORS1_1   = [None, [30, 61, 62, 45], None]
ANCHORS1_2   = [None, None, [116, 90, 156, 198]]
ANCHORS2_0_1 = [[30, 61], [116, 90], None]
ANCHORS2_0_2 = [[10, 13], None, [116, 90]]
ANCHORS2_1_2 = [None, [30, 61], [116, 90]]
ANCHORS3     = [[10, 13], [30,  61], [116, 90]]
ANCHORS      = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]


def get_model_info(model, tsize):
    stride = 32
    img = torch.zeros((1, 3, *tsize), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M\tflops: {:.2f}G".format(params, flops)
    return info, params, flops


def cal_flops_for_flexibleyolo5():
    bkbos = ['yolo5', 'resnext', 'coatnet', 'convnext', 'uniformer']
    variants = ['p', 'n', 'm', 't', 's', 'l', 'h', 'g']
    heads = ['yolo5-n', 'yolo5-s', 'yolo5-m', 'yolo5-l', 'yolo5-x', 'yolox-n', 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x']
    inputs = [320, 384, 448, 512, 576, 640]
    record_f = open('/tmp/flexible_yolo5_flops.txt', 'w')
    record_f.write(f'backbone\thead\tinput_size\tparams(M)\tflops(G)\n')
    #! for normal backbones
    for ip in inputs:
        for head in heads:
            for bkbo in [f'{b}-{v}'for b in bkbos for v in variants]:
                if 'yolo5' in bkbo and bkbo not in ['yolo5-s', 'yolo5-n']:
                    continue
                model = FlexibleYolo5(bkbo_variant=bkbo, head_variant=head, nc=2, input_size=ip, anchors=ANCHORS)
                info, params, flops = get_model_info(model, (ip, ip))
                print(f"bkbo:{bkbo}\thead:{head}\tinput:{ip}\tinfo:{info}")
                record_f.write(f"{bkbo}\t{head}\t{ip}\t{params:3f}\t{flops:.5f}\n")
    #! for swin backbones
    for head in heads:
        for bkbo in ['swin_transformer', 'swin_mlp']:
            for var in variants:
                if var in ['p', 'n']:
                    inputs = [256, 512]
                else:
                    inputs = [320, 640]
                for ip in inputs:
                    model = FlexibleYolo5(bkbo_variant=f'{bkbo}-{var}', head_variant=head, nc=2, input_size=ip, anchors=ANCHORS)
                    info, params, flops = get_model_info(model, (ip, ip))
                    print(f"bkbo:{bkbo}\thead:{head}\tinput:{ip}\tinfo:{info}")
                    record_f.write(f"{bkbo}\t{head}\t{ip}\t{params:3f}\t{flops:.5f}\n")
    record_f.close()
    exit()


class FlexibleYolo5(nn.Module):
    def __init__(self, bkbo_variant='yolo5-x', head_variant='x', nc=80, input_size=640, anchors=ANCHORS3, inplace=True):  # model, input channels, number of classes
        super(FlexibleYolo5, self).__init__()
        bkbo_str = bkbo_variant.split('-')[0]
        if bkbo_str in ['coatnet']:
            assert input_size % 64 == 0, f'Cause of attention, input img-size for {bkbo_str} must be times of 64'
        self.input_size = input_size
        self.anchors = anchors
        self.inplace = inplace
        
        #! build backbone
        self.backbone = self._get_backbone(bkbo_variant, input_size)
        in_fpn_feats  = self._bkbo_forward_once()      ## backbone forward once to get feat_feats
        #! build head & detect
        self._build_head(head_variant, nc, in_fpn_feats, anchors)
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
            backbone = SwinTransformerBackbone(variant=bkbo_variant.split('-')[-1], input_size=input_size)
        elif 'swin_mlp' in bkbo_variant.lower():
            backbone = SwinMLPBackbone(variant=bkbo_variant.split('-')[-1], input_size=input_size)
        else:
            raise ValueError(f'Only {__all_backbones__} are accepted!')
        return backbone
    
    def _build_head(self, head_variant, nc, in_fpn_feats, anchors):
        if 'yolo5' in head_variant.lower():
            self.head = YoloV5Head(variant=head_variant.split('-')[-1], nc=nc, in_fpn_feats=in_fpn_feats, anchors=anchors)
            #! set stride and anchors
            self._build_stride_and_anchors()
            self.stride = self.head.model[-1].stride
        elif 'yolox' in head_variant.lower():
            in_channels = [x.shape[1] for x in in_fpn_feats]
            self.head = YOLOXHead(variant=head_variant.split('-')[-1], nc=nc, in_channels=in_channels)
            self.stride = torch.from_numpy(np.array([8., 16., 32.]))
        else:
            raise ValueError(f'Only {__all_heads__} are accepted!')
    
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


def build_model(num_classes, input_size, bkbo_variant, head_variant, hyp, device, pretrained='', freeze=[], anchors=ANCHORS):
    model = FlexibleYolo5(bkbo_variant=bkbo_variant, head_variant=head_variant, nc=num_classes, input_size=input_size, anchors=anchors)
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
    # cal_flops_for_flexibleyolo5()
    
    bkbo_variant = 'yolo5-n'
    head_variant = 'yolo5-n'
    # Create model
    model = FlexibleYolo5(bkbo_variant=bkbo_variant, head_variant=head_variant, nc=80, input_size=416, anchors=ANCHORS2_0_2)
    
    model.info(verbose=False)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.cuda()
    img = torch.rand(2 if torch.cuda.is_available() else 1, 3, 640, 640).cuda()
    model.train()
    outputs = model(img)
    for out in outputs:
        if isinstance(out, list):
            print(*[v.shape for v in out], sep='\n')
        else:
            print(out.shape)
