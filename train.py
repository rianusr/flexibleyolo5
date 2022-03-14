import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from datetime import datetime

import torch
from torch.cuda import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tactics.plots import plot_labels
from tactics.callbacks import Callbacks
from tactics.bdp import BalancedDataParallel
from tactics.datasetsX import DataPrefetcher
from tactics.autoanchor import check_anchors
from tactics.datasets import create_dataloader
from tactics.dataloaderX import create_yolox_dataloader
from tactics.torch_utils import ModelEMA, select_device, de_parallel
from tactics.yolo5_train import yolo5_train_setup, get_yolo5_optimizer, train_one_epoch_with_yolo5_head
from tactics.yolox_train import yolox_train_setup, get_yolox_optimizer, train_one_epoch_with_yolox_head
from tactics.general import check_img_size, increment_path, colorstr, strip_optimizer, intersect_dicts, check_dataset, LOGGER
from models.yolo import Model
from models.flexibleYolo5 import build_model

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_train', action='store_true', help='training for coco datasets')
    
    parser.add_argument('--pretrained', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yamls/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--lr-scaler', action='store_true', help='LR scale with ratio: batch_size / 64')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    parser.add_argument('--bkbo_variant', type=str, default='yolo5-s', help='backbone variant')
    parser.add_argument('--head_variant', type=str, default='yolo5-s', help='det_head variant')
    parser.add_argument('--im_name', type=str, default='images', help='JPEGImages | images')
    parser.add_argument('--gpu0_bs', type=int, default=0, help='bs of gpu0')
    parser.add_argument('--no_aug', action='store_true', help='no_aug')
    parser.add_argument('--plots', action='store_true', help='plots')
    parser.set_defaults(plots=True)
    
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='pin_memory')
    parser.add_argument('--no_aug_epochs', type=int, default=15, help='bs of gpu0')
    
    parser.add_argument('--stages_save', type=int, default=50, help='save best model every stage')
    
    parser.add_argument('--original_model_benchmark_test', action='store_true', help='using origin yolo5 model for benchmark test')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def set_resume(args, ckpt, ema, optimizer):
    # Optimizer
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        args.best_fitness = ckpt['best_fitness']

    # EMA
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']

    # Epochs
    start_epoch = ckpt['epoch'] + 1
    assert start_epoch > 0, f'{args.pretrained} training to {args.epochs} epochs is finished, nothing to resume.'
    if args.epochs < start_epoch:
        logger.info(f"{args.pretrained} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {args.epochs} more epochs.")
        args.epochs += ckpt['epoch']  # finetune additional epochs

    del ckpt
    return ema, optimizer


def build_dataloader(args):
    if args.yolox_H:
        train_loader, dataset, val_loader = create_yolox_dataloader(args, args.batch_size, WORLD_SIZE > 1)
        return train_loader, dataset, val_loader
    # Trainloader
    train_loader, dataset = create_dataloader(
        args.train_path, args.imgsz, args.batch_size // WORLD_SIZE, args.grid_size, args.single_cls,
        hyp=args.hyp, augment=True, cache=args.cache, rect=args.rect, rank=LOCAL_RANK,
        workers=args.workers, image_weights=args.image_weights, quad=args.quad,
        prefix=colorstr('train: '), shuffle=True, im_name=args.im_name, 
        js_name='instances_trainData.json', coco_format_train=args.coco_train)
    # dataset.class_ids = [0] if args.single_cls else list(range(args.nc))
    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            args.val_path, args.imgsz, args.batch_size // WORLD_SIZE * 2, args.grid_size, args.single_cls,
            hyp=args.hyp, cache=None if args.noval else args.cache, rect=False, rank=-1, workers=args.workers, 
            pad=0.5, prefix=colorstr('val: '), im_name=args.im_name, js_name='instances_valData.json')[0]
    return train_loader, dataset, val_loader


def model_save(args, epoch, ema_model, model, optimizer, is_best, results):
    #! model save
    final_epoch = epoch + 1 == args.epochs
    if (not args.nosave) or (final_epoch and not args.evolve):  # if save
        ckpt = {'epoch': epoch,
                'best_fitness': args.best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema_model.ema).half(),
                'updates': ema_model.updates,
                'optimizer': optimizer.state_dict(),
                'date': datetime.now().isoformat()}

        #! Save last
        torch.save(ckpt, args.last)
        #! Save best
        if is_best:
            torch.save(ckpt, args.best)
        #! Save stage best
        if (epoch + 1) % args.stages_save == 0:
            stage_best = args.w_dir / f'best_stage{epoch+1}.pt'
            os.system(f"cp {args.best} {stage_best}")
            strip_optimizer(stage_best)
        del ckpt
    return results


def save_run_settings(args):
    with open(args.save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(args.hyp, f, sort_keys=False)
    with open(args.save_dir / 'args.yaml', 'w') as f:
        k_w_args = deepcopy(vars(args))
        for k,v in k_w_args.items():
            if not isinstance(v, (int, str, bool, float)):
                k_w_args[k] = str(v)
        k_w_args.pop('hyp')
        yaml.safe_dump(k_w_args, f, sort_keys=False)


def build_original_yolo5_model(args, device):
    ckpt = None
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)  # load checkpoint
    model = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=args.nc, anchors=args.hyp.get('anchors')).to(device)  # create
    if ckpt:
        exclude = ['anchor'] if (args.cfg or args.hyp.get('anchors')) and not args.resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    return model, ckpt


def main(args, callbacks=Callbacks()):
    args.yolox_H = 'yolox' in args.head_variant.lower()
    if not os.path.exists(args.pretrained):
        logger.warning(f'{args.pretrained} not Exists! Training from scratch!')
        args.pretrained = ''
    args.is_distributed = WORLD_SIZE > 1

    #! saving dir
    if args.yolox_H:
        args.project = os.path.abspath(args.project) + f'_yoloxH/{args.bkbo_variant.lower()}__{args.head_variant.lower()}'
    else:
        args.project = os.path.abspath(args.project) + f'_yolo5H/{args.bkbo_variant.lower()}__{args.head_variant.lower()}'
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
    
    if RANK in [-1, 0]:
        logger.remove(handler_id=None)              ## 关闭logger的sys.stderr，可以输出到文件，但不会在终端显示 
        logger.add(args.save_dir / 'runtime.log')
    args.w_dir = args.save_dir / 'weights'  # weights dir
    args.w_dir.mkdir(parents=True, exist_ok=True)
    args.last, args.best = args.w_dir / 'last.pt', args.w_dir / 'best.pt'
    args.nbs = 64 ## for yolo5 # nominal batch size

    device = select_device(args.device, batch_size=args.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert args.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not args.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not args.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Hyperparameters
    if not os.path.exists(args.hyp):
        logger.error('Hyp file: {args.hyp} not exists, training with "hyp.scratch.yaml"')
        args.hyp = ROOT / 'data/hyps/hyp.scratch.yaml'
    with open(args.hyp, errors='ignore') as f:
        args.hyp = yaml.safe_load(f)  # load hyps dict
    
    if args.yolox_H:
        args.hyp['lr0'] = args.hyp['basic_lr_per_img'] * args.batch_size
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in args.hyp.items()))
    
    if RANK in [-1, 0]: ## print info
        print(args)
        ##! Save run settings [optional]
        save_run_settings(args)
    
    ##! dataset info
    data_dict = check_dataset(args.data)
    args.train_path = data_dict['train']
    args.val_path   = data_dict['val']
    args.nc         = data_dict['nc']
    args.names      = data_dict['names']
    
    if args.single_cls:
        args.names = data_dict['names'] = ['item']
        args.nc    = data_dict['nc'] = 1
    if args.yolox_H:
        # assert 'background' in args_dict['classes'], '"background" must provided and must put it into index 0'
        args.names.insert(0, 'background')
        args.nc += 1
        
    args.data_dict = data_dict
    assert len(args.names) == args.nc, f'{len(args.names)} names found for nc={args.nc} dataset!'  # check
    
    ##! build model
    if args.original_model_benchmark_test:
        model, ckpt = build_original_yolo5_model(args, device)
    else:
        freeze=[]
        model, ckpt = build_model(args.nc, args.imgsz, args.bkbo_variant, args.head_variant, args.hyp, device, 
                                  pretrained=args.pretrained, freeze=freeze, anchors=args.hyp['anchors_fpn'])
    model.nc    = args.nc      # attach number of classes to model
    model.hyp   = args.hyp     # attach hyperparameters to model
    model.names = args.names
    
    ##! Image size
    args.grid_size = max(int(model.stride.max()), 32)  # grid size (max stride)
    args.imgsz = check_img_size(args.imgsz, args.grid_size, floor=args.grid_size * 2)  # verify imgsz is gs-multiple
    
    ##! optimizer
    if args.yolox_H:
        optimizer = get_yolox_optimizer(args, model) 
    else:
        optimizer = get_yolo5_optimizer(args, model) 
        
    # yolo5 EMA
    if not args.yolox_H:
        ema_model = ModelEMA(model) if RANK in [-1, 0] else None
    
    args.start_epoch, args.best_fitness = 0, 0.0
    if args.resume and ckpt:
        ema_model, optimizer = set_resume(args, ckpt, ema_model, optimizer)
    
    args.use_cuda = device.type != 'cpu'
    ##! DP or BDP mode 
    if args.use_cuda and torch.cuda.device_count() > 1:
        if args.gpu0_bs > 0:
            logger.info('>>>>>>>>>  Use BalancedDataParallel mode!!')
            model = BalancedDataParallel(args.gpu0_bs, model, dim=0).to(device)
        else:
            logger.info('>>>>>>>>>  Use DataParallel mode!!')
            model = torch.nn.DataParallel(model)
    
    # SyncBatchNorm
    if args.sync_bn and args.use_cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
    
    ##! build dataloder
    train_loader, dataset, val_loader = build_dataloader(args)
    prefetcher = DataPrefetcher(train_loader)
    
    if not args.resume and not args.yolox_H:
        labels = np.concatenate(dataset.labels, 0)
        if args.plots:
            plot_labels(labels, args.names, args.save_dir)

        # Anchors
        if not args.noautoanchor:
            check_anchors(args, dataset, model=model, thr=args.hyp['anchor_t'], imgsz=args.imgsz)
    model.half().float()  # pre-reduce anchor precision
        
    # DDP mode
    if args.use_cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    
    # yolox EMA
    if args.yolox_H:
        ema_model = ModelEMA(model, args.hyp['yolox_ema_decay']) if RANK in [-1, 0] else None
    # Start training
    # number of warmup iterations, max(3 epochs, 1k iterations)
    args.warmup_num = max(round(args.hyp['warmup_epochs'] * len(train_loader)), 1000)  
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    scaler = amp.GradScaler(enabled=args.use_cuda)
    logger.info(f'Image sizes {args.imgsz} train, {args.imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', args.save_dir)}\n"
                f'Starting training for {args.epochs} epochs...')

    ## before train setup
    if args.yolox_H:
        lr_scheduler, evaluator = yolox_train_setup(args, train_loader, val_loader, RANK)
    else:
        model, compute_loss, lr_scheduler, lf = yolo5_train_setup(args, model, optimizer, dataset, device)
    
    ##! start training
    for epoch in range(args.start_epoch, args.epochs):  
        # epoch start ------------------------------------------------------------------
        model.train()
        if args.yolox_H:
            is_best, results = train_one_epoch_with_yolox_head( args, epoch, 
                                                                model, ema_model, 
                                                                prefetcher, train_loader, val_loader, 
                                                                optimizer, scaler, lr_scheduler, 
                                                                evaluator, device, RANK)
        else:
            is_best, results = train_one_epoch_with_yolo5_head( args, epoch, 
                                                                model, ema_model, 
                                                                dataset, train_loader, val_loader,
                                                                optimizer, scaler, lr_scheduler, lf, 
                                                                compute_loss, callbacks, device, RANK, WORLD_SIZE)
        if RANK in [-1, 0]:
            model_save(args, epoch, ema_model, model, optimizer, is_best, results)
        # epoch end ---------------------------------------------------------------------
        
    ##! end training
    torch.cuda.empty_cache()
    if RANK in [-1, 0]:
        strip_optimizer(args.best)
        strip_optimizer(args.last)
    if WORLD_SIZE > 1 and RANK == 0:
        logger.info('Destroying process group... ')
        dist.destroy_process_group() 


if __name__ == '__main__':
    args = parse_args()
    main(args)
