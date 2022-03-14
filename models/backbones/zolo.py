import os
import sys
import math

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__).replace('models/backbones', ''))
from models.common import Conv, C3, SPPF, Concat
from tactics.general import check_version
from tactics.torch_utils import model_info, cal_flops

YOLO5_VARIANT = {'p': {'depth_multiple': 0.33, 'width_multiple': 0.125}}
fpn_names = ['p1', 'p2', 'p3']

ANCHORS1_0   = [[10, 13, 16, 30], None, None]
ANCHORS1_1   = [None, [30, 61, 62, 45], None]
ANCHORS1_2   = [None, None, [116, 90, 156, 198]]
ANCHORS2_0_1 = [[30, 61], [116, 90], None]
ANCHORS2_0_2 = [[10, 13], None, [116, 90]]
ANCHORS2_1_2 = [None, [30, 61], [116, 90]]
ANCHORS3     = [[10, 13], [30,  61], [116, 90]]
ANCHORS      = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

HEAD_PARAMS = [
                [-1,           ],      #! backbone-0  
                [-1,           ],      #! backbone-1  
                [-1,           ],      #! backbone-2  
                [-1,           ],      #! backbone-3  
                [-1,           ],      #! backbone-4  
                [-1,           ],      #! backbone-5  
                [-1,           ],      #! backbone-6  
                [-1,           ],      #! backbone-7  
                [-1,           ],      #! backbone-8  
                [-1,           ],      #! backbone-9  
                [-1,           ],      #! head-0
                [-1,           ],      #! head-1
                [[-1, 'p2'],   ],      #! head-2
                [-1,           ],      #! head-3
                [-1,           ],      #! head-4
                [-1,           ],      #! head-5
                [[-1, 'p1'],   ],      #! head-6
                [-1,           ],      #! head-7
                [-1,           ],      #! head-8
                [[-1, 'h1'],   ],      #! head-9
                [-1,           ],      #! head-10
                [-1,           ],      #! head-11
                [[-1, 'h0'],   ],      #! head-12
                [-1,           ],      #! head-13
                [['d2', 'd3']] ]       #! head-14


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=ANCHORS1_0, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class zolo(nn.Module):
    def __init__(self, variant='p', nc=80, anchors=ANCHORS1_0, head_params=HEAD_PARAMS):  # model, input channels, number of classes
        super(zolo, self).__init__()
        self.anchors, self.valid_anchors_idx, self.head_d_n = self.check_anchors(anchors) 
        self.head_params    = head_params
        self.update_head_params()
        
        self.variant_params = YOLO5_VARIANT[variant]
        gh, gw = self.variant_params['depth_multiple'], self.variant_params['width_multiple']
        # layers
        base_channel = math.ceil(64 * gw / 8) * 8   # 32
        
        ##! backbone!
        backbone_0 = Conv(3, base_channel, 6, 2, 2)                                     ## Focus
        backbone_1 = Conv(base_channel, base_channel * 2, 3, 2)
        backbone_2 = C3(base_channel * 2, base_channel * 2,   max(round(3 * gh), 1))
        backbone_3 = Conv(base_channel * 2, base_channel * 4, 3, 2)
        backbone_4 = C3(base_channel * 4, base_channel * 4,   max(round(3 * gh), 1))
        backbone_5 = Conv(base_channel * 4, base_channel * 8, 3, 2)
        backbone_6 = C3(base_channel * 8, base_channel * 8,   max(round(6 * gh), 1))
        backbone_7 = Conv(base_channel * 8, base_channel * 16, 3, 2)
        backbone_8 = C3(base_channel * 16, base_channel * 16, max(round(3 * gh), 1))
        backbone_9 = SPPF(base_channel * 16, base_channel * 16)
        ## build backbone
        bkbo_layers = [backbone_0, backbone_1, backbone_2, backbone_3, backbone_4, backbone_5, backbone_6, backbone_7, backbone_8, backbone_9]
        
        ## 8 * 512 * 20 * 20
        head_0 = Conv(base_channel * 16, base_channel * 8, 1, 1)
        head_1 = nn.Upsample(None, 2, 'nearest')
        head_2 = Concat(1)
        head_3 = C3(base_channel * 16, base_channel * 8,   max(round(3 * gh), 1), False)
        head_4 = Conv(base_channel * 8, base_channel * 4, 1, 1)
        head_5 = nn.Upsample(None, 2, 'nearest')
        head_6 = Concat(1)
        head_7 = C3(base_channel * 8, base_channel * 4,    max(round(3 * gh), 1), False)
        head_8 = Conv(base_channel * 4, base_channel * 4, 3, 2)
        head_9 = Concat(1)
        head_10 = C3(base_channel * 8, base_channel * 8,   max(round(3 * gh), 1), False)
        head_11 = Conv(base_channel * 8, base_channel * 8, 3, 2)
        head_12 = Concat(1)
        head_13 = C3(base_channel * 16, base_channel * 16, max(round(3 * gh), 1), False)

        fpn_p3_ch = base_channel * 4
        fpn_p4_ch = base_channel * 8
        fpn_p5_ch = base_channel * 16
        fpn_chs = (fpn_p3_ch, fpn_p4_ch, fpn_p5_ch)
        
        detect_layer = Detect(nc=nc, anchors=self.anchors, ch=[fpn_chs[ix] for ix in self.valid_anchors_idx])
        
        head_layers = [head_0, head_1, head_2, head_3, head_4, head_5, head_6, head_7, head_8, head_9, head_10, head_11, head_12, head_13, detect_layer]
        
        layers = [*bkbo_layers, *head_layers]
        self.model = nn.Sequential(*layers)
        self._set_layers_attr()

    def forward(self, x, profile=False):
        pyramid_feats = {}
        pyramid_feats['h0'], pyramid_feats['h1'] = None, None
        pyramid_feats['p1'], pyramid_feats['p2'], pyramid_feats['p3'] = None, None, None
        pyramid_feats['d1'], pyramid_feats['d2'], pyramid_feats['d3'] = None, None, None
        
        for idx, m in enumerate(self.model):
            if profile:
                in_shape = x.shape
                
            if m.f != -1:
                x = [x if j == -1 else pyramid_feats[j] for j in m.f]
            x = m(x)
            
            if idx == 4:
                pyramid_feats['p1'] = x
            if idx == 6:
                pyramid_feats['p2'] = x
            if idx == 8:
                pyramid_feats['p3'] = x
            
            if idx == 10:
                pyramid_feats['h0'] = x
            if idx == 14:
                pyramid_feats['h1'] = x  
            if idx == 17:
                pyramid_feats['d1'] = x
            if idx == 20:
                pyramid_feats['d2'] = x
            if idx == 23:
                pyramid_feats['d3'] = x
                
            if profile:         ## print input output info
                if isinstance(x, list):
                    out_shape = [xi.shape for xi in x]
                else:
                    out_shape = x.shape
                print(f"model_idx:{idx:02d}{'':8}model_type:{m.type}\tinput_shape:{in_shape}{'':4}output_shape:{out_shape}".expandtabs(36))
        del pyramid_feats
        return x

    def update_head_params(self):
        detect_fpns = ['d1', 'd2', 'd3']
        self.head_params[-1][0] = [detect_fpns[ix] for ix in self.valid_anchors_idx]
    
    def check_anchors(self, anchors):
        valid_a_n = 0
        valid_a_idx_lst = []
        new_anchors = []
        for i, a in enumerate(anchors):
            if a is not None:
                valid_a_n += 1
                valid_a_idx_lst.append(i)
                new_anchors.append(a)
        assert len(set([len(na) for na in new_anchors])) == 1, 'Invalid anchors param with different shape for different fpn layer'
        return new_anchors, valid_a_idx_lst, valid_a_n
    
    def _set_layers_attr(self, profile=False):
        for idx, m in enumerate(self.model):
            t = str(m.__class__)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m.parameters()])  # number params
            m.i, m.f, m.type, m.np = idx, self.head_params[idx][0], t, np
            
            if profile:
                print(f"{m.i}\t{m.f}\t{m.np}\t{ m.type}".expandtabs(16))


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Create model
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640)
    model = zolo(variant='p', anchors=ANCHORS1_0)
    cal_flops(model, input_size=640, batch_size=1)
    fpn_feats = model(img)
    print(*[v.shape for v in fpn_feats], sep='\n')
    print(f'params: {count_parameters(model)/(1024*1024):.2f} M')
    model_info(model, verbose=True, img_size=640)