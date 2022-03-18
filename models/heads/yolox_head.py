#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__).replace('models/heads', ''))
from tactics.yolox_loss import IOUloss

YOLOX_VARIANT = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        't': {'depth_multiple': 0.33, 'width_multiple': 0.375},
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'l': {'depth_multiple': 1.0,  'width_multiple': 1.0},
        'x': {'depth_multiple': 1.33, 'width_multiple': 1.25}
    }


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class YOLOXHead(nn.Module):
    def __init__(
        self,
        variant='l',
        nc=80,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        width = YOLOX_VARIANT[variant]['width_multiple']

        self.n_anchors = 1
        self.stride = torch.from_numpy(np.array(strides))
        self.num_classes = nc
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(in_channels[i]),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(in_channels[i]),
                            out_channels=int(in_channels[i]),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(in_channels[i]),
                            out_channels=int(in_channels[i]),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(in_channels[i]),
                            out_channels=int(in_channels[i]),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(in_channels[i]),
                            out_channels=int(in_channels[i]),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[i]),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[i]),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[i]),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def forward(self, xin):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)
            
        if self.training:
            return outputs, x_shifts, y_shifts, expanded_strides, origin_preds
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k].to(output.device)
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        # if grid.device != output.device:
        #     print()
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid


def get_args_parser():
    parser = argparse.ArgumentParser(description='yolo5 training', add_help=False)
    parser.add_argument('--weights', type=str, default='/data/pretrainedModelZoo/yolo5/yolov5x.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--autoanchor', type=int, default=1, help='auto anchor for training')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--train_path', default='/data/datasets/detData/expData/trainData', help='trainData dir')
    parser.add_argument('--test_path', default='/data/datasets/detData/expData/valData', help='valData dir')
    parser.add_argument('--gpu0_bs', default=0, type=int, help='batch_size of gpu0, 0 for average distribution')
    
    parser.add_argument('--fp16', action='store_true', help='float16 training')
    parser.add_argument('--input_size', type=int, default=640, help='train image input size')
    parser.add_argument('--bkbo_variant', default='yolo5-x', type=str, help='bkbo_variant')
    parser.add_argument('--head_variant', default='x', type=str, help='head_variant')
    
    # for yolox
    parser.add_argument('--data_num_workers', default=4, type=int, help='')
    parser.add_argument('--multiscale_range', default=5, type=int, help='')
    parser.add_argument('--random_size', default=(14, 26), nargs='+', help='')
    parser.add_argument('--train_dir', default='/data/datasets/detData/expData/trainData', type=str, help='')
    parser.add_argument('--val_dir', default='/data/datasets/detData/expData/valData', type=str, help='')
    parser.add_argument('--train_ann', default="instances_trainData.json", type=str, help='')
    parser.add_argument('--val_ann', default="instances_valData.json", type=str, help='')
    parser.add_argument('--mosaic_prob', default=1.0, type=float, help='')
    parser.add_argument('--mixup_prob', default=1.0, type=float, help='')
    parser.add_argument('--hsv_prob', default=1.0, type=float, help='')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='')
    parser.add_argument('--degrees', default=10.0, type=float, help='')
    parser.add_argument('--translate', default=0.1, type=float, help='')
    parser.add_argument('--mosaic_scale', default=(0.1, 2), help='')
    parser.add_argument('--mixup_scale', default=(0.5, 1.5), help='')
    parser.add_argument('--shear', default=2.0, type=float, help='')
    parser.add_argument('--perspective', default=0.0, type=float, help='')
    parser.add_argument('--enable_mixup', default=True, type=bool, help='')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='')
    parser.add_argument('--max_epoch', default=300, type=int, help='')
    parser.add_argument('--warmup_lr', default=0, type=int, help='')
    parser.add_argument('--basic_lr_per_img', default=0.01 / 64.0, type=float, help='')
    parser.add_argument('--scheduler', default="yoloxwarmcos", type=str, help='')
    parser.add_argument('--no_aug_epochs', default=15, type=int, help='')
    parser.add_argument('--min_lr_ratio', default=0.05, type=float, help='')
    parser.add_argument('--ema', default=True, type=bool, help='')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
    parser.add_argument('--momentum', default=0.9, type=float, help='')
    parser.add_argument('--print_interval', default=10, type=int, help='')
    parser.add_argument('--eval_interval', default=10, type=int, help='')
    parser.add_argument('--test_size', default=(640, 640), help='')
    parser.add_argument('--test_conf', default=0.01, type=float, help='')
    parser.add_argument('--nmsthre', default=0.65, type=float, help='')
    return parser


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    ## 获取并更新参数
    parser = argparse.ArgumentParser('Nfnets dist training script', parents=[get_args_parser()])
    args = parser.parse_args()

    head = YOLOXHead(variant='l', nc=80, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False).cuda()
    head.use_l1 = True
    bs = args.batch_size = 4
    
    bkbo_feats = [torch.rand(bs, 256, 80, 80).cuda(), torch.rand(bs, 512, 40, 40).cuda(), torch.rand(bs, 1024, 20, 20).cuda()]
    head.eval()
    outputs, x_shifts, y_shifts, expanded_strides, origin_preds  = head(bkbo_feats)
    print(outputs[0].shape)
    print(x_shifts[0].shape)
    print(y_shifts[0].shape)
    print(expanded_strides[0].shape)
    print(origin_preds[0].shape)
