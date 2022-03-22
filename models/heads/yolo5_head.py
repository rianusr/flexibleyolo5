import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(__file__).replace('/models/heads', ''))
from tactics.general import check_version
from models.common import Conv, C3, Concat


fpn_names = ['p1', 'p2', 'p3']
YOLO5_VARIANT = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        't': {'depth_multiple': 0.33, 'width_multiple': 0.375},
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'l': {'depth_multiple': 1.0,  'width_multiple': 1.0},
        'x': {'depth_multiple': 1.33, 'width_multiple': 1.25}
    }

ANCHORS1_0   = [[10, 13, 16, 30], None, None]
ANCHORS1_1   = [None, [30, 61, 62, 45], None]
ANCHORS1_2   = [None, None, [116, 90, 156, 198]]
ANCHORS2_0_1 = [[30, 61], [116, 90], None]
ANCHORS2_0_2 = [[10, 13], None, [116, 90]]
ANCHORS2_1_2 = [None, [30, 61], [116, 90]]
ANCHORS3     = [[10, 13], [30,  61], [116, 90]]
ANCHORS      = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

HEAD_PARAMS = [
                [-1,                 ],     #! layer-idx = 0
                [-1,                 ],     #! layer-idx = 1
                [[-1, 'p2'],         ],     #! layer-idx = 2
                [-1,                 ],     #! layer-idx = 3
                [-1,                 ],     #! layer-idx = 4
                [-1,                 ],     #! layer-idx = 5
                [[-1, 'p1'],         ],     #! layer-idx = 6
                [-1,                 ],     #! layer-idx = 7
                [-1,                 ],     #! layer-idx = 8
                [[-1, 'h1'],         ],     #! layer-idx = 9
                [-1,                 ],     #! layer-idx = 10
                [-1,                 ],     #! layer-idx = 11
                [[-1, 'h0'],         ],     #! layer-idx = 12
                [-1,                 ],     #! layer-idx = 13
                [['d1', 'd2', 'd3']] ]      #! layer-idx = 14

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
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

        if len(z) == 1:
            return x if self.training else z
        return x if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class YoloV5Head(nn.Module):
    def __init__(self, variant='s', nc=80, in_fpn_feats={}, anchors=ANCHORS, head_params=HEAD_PARAMS):  # model, input channels, number of classes
        super(YoloV5Head, self).__init__()
        assert in_fpn_feats and len(in_fpn_feats) == 3, '''
            fpn_shapes: feats for fpn layer, it contains 3 stages: p1, p2, p3, for example:
            fpn_shapes = {
                'p1':(b, 128, 80, 80),     # 'p1'  (batch_size, channel, width, height) <tensor>
                'p2':(b, 256, 40, 40),     # 'p2'
                'p3':(b, 512, 20, 20),     # 'p3'
            }
        '''
        self.anchors, self.valid_anchors_idx, self.head_d_n = self.check_anchors(anchors) 
        self.head_params = head_params
        self.update_head_params()
        
        gh = YOLO5_VARIANT[variant]['depth_multiple']
        self.p1_ch, self.p2_ch, self.p3_ch = in_fpn_feats[0].shape[1], in_fpn_feats[1].shape[1], in_fpn_feats[2].shape[1]
        
        ## 8 * 512 * 20 * 20
        head_0 = Conv(self.p3_ch, self.p2_ch, 1, 1)
        head_1 = nn.Upsample(None, 2, 'nearest')
        head_2 = Concat(1)
        head_3 = C3(self.p3_ch, self.p2_ch, max(round(3 * gh), 1), False)
        head_4 = Conv(self.p2_ch, self.p1_ch, 1, 1)
        head_5 = nn.Upsample(None, 2, 'nearest')
        head_6 = Concat(1)
        head_7 = C3(self.p2_ch, self.p1_ch, max(round(3 * gh), 1), False)
        fpn_p3_ch = self.p1_ch
        head_8 = Conv(self.p1_ch, self.p1_ch, 3, 2)
        head_9 = Concat(1)
        head_10 = C3(self.p2_ch, self.p2_ch, max(round(3 * gh), 1), False)
        fpn_p4_ch = self.p2_ch
        head_11 = Conv(self.p2_ch, self.p2_ch, 3, 2)
        head_12 = Concat(1)
        head_13 = C3(self.p3_ch, self.p3_ch, max(round(3 * gh), 1), False)
        fpn_p5_ch = self.p3_ch
        fpn_chs = (fpn_p3_ch, fpn_p4_ch, fpn_p5_ch)
        detect_layer = Detect(nc=nc, anchors=self.anchors, ch=[fpn_chs[ix] for ix in self.valid_anchors_idx])
        head_layers = [head_0, head_1, head_2, head_3, head_4, head_5, head_6, head_7, head_8, head_9, head_10, head_11, head_12, head_13, detect_layer]

        self.model = nn.Sequential(*head_layers)
        self._set_layers_attr()

    def _set_layers_attr(self, profile=False):
        for idx, m in enumerate(self.model):
            t = str(m.__class__)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m.parameters()])  # number params
            m.i, m.f, m.type, m.np = idx, HEAD_PARAMS[idx][0], t, np
            
            if profile:
                print(f"{m.i}\t{m.f}\t{m.np}\t{ m.type}".expandtabs(16))

    def forward(self, bkbo_fpn_feats, profile=False):
        x = bkbo_fpn_feats[-1]          ## 最后一层
        pyramid_feats = {}
        for idx, v in enumerate(bkbo_fpn_feats):
            pyramid_feats[fpn_names[idx]] = v
        pyramid_feats['h0'], pyramid_feats['h1'] = None, None
        pyramid_feats['d1'], pyramid_feats['d2'], pyramid_feats['d3'] = None, None, None
        
        for idx, m in enumerate(self.model):
            if profile:
                in_shape = x.shape
                
            if m.f != -1:
                x = [x if j == -1 else pyramid_feats[j] for j in m.f]
            x = m(x)
            
            if idx == 0:
                pyramid_feats['h0'] = x
            if idx == 4:
                pyramid_feats['h1'] = x  
            if idx == 7:
                pyramid_feats['d1'] = x
            if idx == 10:
                pyramid_feats['d2'] = x
            if idx == 13:
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
            if a is not None and not isinstance(a, str):
                valid_a_n += 1
                valid_a_idx_lst.append(i)
                new_anchors.append(a)
        assert len(set([len(na) for na in new_anchors])) == 1, 'Invalid anchors param with different shape for different fpn layer'
        return new_anchors, valid_a_idx_lst, valid_a_n


if __name__ == '__main__':
    pyramid_feats = [torch.rand(8,96,80,80), torch.rand(8,192,40,40), torch.rand(8,384,20,20)]
    head = YoloV5Head(variant='s', nc=80, in_fpn_feats=pyramid_feats, anchors=ANCHORS1_2)
    output = head(pyramid_feats)
    print(*[v.shape for v in output], sep='\n')