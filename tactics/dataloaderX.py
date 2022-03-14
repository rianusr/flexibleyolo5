import uuid
import random
import argparse
import numpy as np
from loguru import logger
from contextlib import contextmanager

import torch
import torch.distributed as dist

from .datasetsX import (
    COCODataset,
    TrainTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    get_local_rank,
    worker_init_reset_seed,
)


@contextmanager
def wait_for_the_master(local_rank: int):
    """
    Make all processes waiting for the master to do some task.
    """
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()


def get_data_loader(args, batch_size, is_distributed, cache_img=False):
    if isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 2:
        img_size = args.imgsz
    elif isinstance(args.imgsz, int):
        img_size = (args.imgsz, args.imgsz)
    else:
        raise ValueError(f'Image size error! Input image size: {args.imgsz}')
    
    local_rank = get_local_rank()
    with wait_for_the_master(local_rank):
        dataset = COCODataset(
            data_dir=args.train_path,
            json_file='instances_trainData.json',
            name='images',
            img_size=img_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=args.hyp['flip_prob'],
                hsv_prob=args.hyp['hsv_prob']),
            cache=cache_img,
        )

    dataset = MosaicDetection(
        dataset,
        mosaic=not args.no_aug,
        img_size=img_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=args.hyp['flip_prob'],
            hsv_prob=args.hyp['hsv_prob']),
        degrees=args.hyp['degrees'],
        translate=args.hyp['translate'],
        mosaic_scale=args.hyp['mosaic_scale'],
        mixup_scale=args.hyp['mixup_scale'],
        shear=args.hyp['shear'],
        perspective=args.hyp['perspective'],
        enable_mixup=args.hyp['enable_mixup'],
        mosaic_prob=args.hyp['mosaic_prob'],
        mixup_prob=args.hyp['mixup_prob'],
    )


    if is_distributed:
        batch_size = batch_size // dist.get_world_size()

    sampler = InfiniteSampler(len(dataset), seed=0)

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        mosaic=not args.no_aug,
    )

    dataloader_kwargs = {"num_workers": args.workers, "pin_memory": args.pin_memory}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(dataset, **dataloader_kwargs)

    return train_loader, dataset

def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str, default='for yolox dataloader', help='training for coco datasets')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_eval_loader(args, batch_size, is_distributed, cache_img=False):
    if isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 2:
        img_size = args.imgsz
    elif isinstance(args.imgsz, int):
        img_size = (args.imgsz, args.imgsz)
    else:
        raise ValueError(f'Image size error! Input image size: {args.imgsz}')
    local_rank = get_local_rank()
    with wait_for_the_master(local_rank):
        dataset = COCODataset(
            data_dir=args.val_path,
            json_file='instances_valData.json',
            name='images',
            img_size=img_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=0.0,
                hsv_prob=0.0),
            cache=cache_img,
        )
    if is_distributed:
        batch_size = batch_size // dist.get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    val_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return val_loader

def create_yolox_dataloader(args, batch_size, is_distributed, cache_img=False):
    train_loader, dataset = get_data_loader(args, batch_size, is_distributed, cache_img=cache_img)
    val_loader = get_eval_loader(args, batch_size * 2, is_distributed)
    return train_loader, dataset, val_loader


if __name__ == '__main__':
    args = parse_args()
    args.train_path = '/data/datasets/ga/id_4237_gesture_ll/trainData'
    args.val_path = '/data/datasets/ga/id_4237_gesture_ll/valData'
    args.train_ann = 'instances_trainData.json'
    args.val_ann = 'instances_valData.json'
    # args.name   = 'images'
    # args.input_size = (640, 640)
    # args.hsv_prob   = 1.0
    # args.flip_prob = 0.5
    # args.degrees = 10.0
    # args.translate = 0.1
    # args.mosaic_scale = (0.1, 2)
    # args.mixup_scale = (0.5, 1.5)
    # args.shear = 2.0
    # args.perspective = 0.0
    # args.enable_mixup = True
    # args.mosaic_prob    = 1.0
    # args.mixup_prob = 1.0
    # args.seed = None
    # args.data_num_workers = 4
    
    
    # train_loader = get_data_loader(args, 32, False, no_aug=False, cache_img=False)
    # logger.info("init prefetcher, this might take one minute or less...")
    # prefetcher = DataPrefetcher(train_loader)
    # data_type = torch.float32
    # for i in range(len(train_loader)):
    #     inps, targets = prefetcher.next()
    #     inps = inps.to(data_type)
    #     targets = targets.to(data_type)
    #     targets.requires_grad = False
    #     # inps, targets = preprocess(inps, targets, (640, 640))
    #     print(targets.shape)
    #     print(inps.shape)
    #     exit()
        
    val_loader = get_eval_loader(args, 32, False, testdev=False, legacy=False)
    for data in val_loader:
        print()