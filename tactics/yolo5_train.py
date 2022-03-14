import math
import random
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler

import val  # for end-of-epoch mAP
from tactics.metrics import fitness
from tactics.loss import ComputeLoss
from tactics.torch_utils import EarlyStopping, de_parallel
from tactics.general import colorstr, labels_to_class_weights, labels_to_image_weights, one_cycle, LOGGER


def yolo5_train_setup(args, model, optimizer, dataset, device):
    """[summary]
    """
    # Model attributes
    if args.original_model_benchmark_test:
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    else:
        nl = de_parallel(model).head.model[-1].nl
    args.hyp['box'] *= 3 / nl  # scale to layers
    args.hyp['cls'] *= args.nc / 80 * 3 / nl  # scale to classes and layers
    args.hyp['obj'] *= (args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    args.hyp['label_smoothing'] = args.label_smoothing
    model.class_weights = labels_to_class_weights(dataset.labels, args.nc).to(device) * args.nc  # attach class weights
    
    args.last_opt_step = -1
    args.maps = np.zeros(args.nc)  # mAP per class
    
    # stopper = EarlyStopping(patience=args.patience)
    compute_loss = ComputeLoss(de_parallel(model), original_model_benchmark_test=args.original_model_benchmark_test)  # init loss class
    
    ##! Scheduler
    if args.linear_lr:
        lf = lambda x: (1 - x / (args.epochs - 1)) * (1.0 - args.hyp['lrf']) + args.hyp['lrf']  # linear
    else:
        lf = one_cycle(1, args.hyp['lrf'], args.epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler.last_epoch = args.start_epoch - 1  # do not move
    return model, compute_loss, scheduler, lf


def get_yolo5_optimizer(args, model):
    # Optimizer
    args.accumulate = max(round(args.nbs / args.batch_size), 1)  # accumulate loss before optimizing
    args.hyp['weight_decay'] *= args.batch_size * args.accumulate / args.nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {args.hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if args.optimizer == 'Adam':
        optimizer = Adam(g0, lr=args.hyp['lr0'], betas=(args.hyp['yolo5_momentum'], 0.999))  # adjust beta1 to momentum
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=args.hyp['lr0'], betas=(args.hyp['yolo5_momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=args.hyp['lr0'], momentum=args.hyp['yolo5_momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': args.hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    del g0, g1, g2
    return optimizer


def epoch_eval_yolo5(args, epoch, evalmodel, val_loader, callbacks=None):
    results = (0, 0, 0, 0, 0, 0, 0)
    # Calculate mAP
    results, args.maps, _ = val.run(args.data_dict,
                                    batch_size=args.batch_size * 2,
                                    imgsz=args.imgsz,
                                    model=evalmodel,
                                    single_cls=args.single_cls,
                                    dataloader=val_loader,
                                    save_dir=args.save_dir,
                                    plots=False,
                                    save_json=False,
                                    callbacks=callbacks,
                                    compute_loss=False)
    # Update best mAP
    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    if fi > args.best_fitness:
        args.best_fitness = fi
    val_info = f'【epoch-val】:{epoch+1:3d}\tP: {results[0]:.5f}\tR: {results[1]:.5f}\tmAP@.5: {results[2]:.5f}\tmAP@.5:.95: {results[3]:.5f}'
    logger.info(val_info)
    return args.best_fitness == fi, results


def train_one_epoch_with_yolo5_head(args, epoch, model, ema_model, dataset, train_loader, val_loader, optimizer, scaler, scheduler, lf, compute_loss, callbacks, device, rank, world_size):
    nb = len(train_loader)
    gs = args.grid_size
    # Update image weights (optional, single-GPU only)
    if args.image_weights:
        cw = model.class_weights.cpu().numpy() * (1 - args.maps) ** 2 / args.nc  # class weights
        iw = labels_to_image_weights(dataset.labels, nc=args.nc, class_weights=cw)  # image weights
        dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

    # Update mosaic border (optional)
    # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
    # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

    mloss = torch.zeros(4, device=device)  # mean losses
    if rank != -1:
        train_loader.sampler.set_epoch(epoch)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'total', 'box', 'obj', 'cls', 'labels', 'img_size'))
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    optimizer.zero_grad()
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= args.warmup_num:
            xi = [0, args.warmup_num]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            args.accumulate = max(1, np.interp(ni, xi, [1, args.nbs / args.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [args.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [args.hyp['warmup_momentum'], args.hyp['yolo5_momentum']])

        # Multi-scale
        if args.multi_scale:
            sz = random.randrange(args.imgsz * 0.5, args.imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward
        with amp.autocast(enabled=args.use_cuda):
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if rank != -1:
                loss *= world_size  # gradient averaged between devices in DDP mode
            if args.quad:
                loss *= 4.

        # Backward
        scaler.scale(loss).backward()

        # Optimize
        if ni - args.last_opt_step >= args.accumulate:
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema_model:
                ema_model.update(model)
            args.last_opt_step = ni

        # Log
        if rank in [-1, 0]:
            loss_items = torch.cat((loss, loss_items)).detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            iter_info = ('%10s' * 2 + '%10.4g' * 6) % (f'{epoch+1}/{args.epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(iter_info)
            callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, args.plots, args.sync_bn)
        # end batch ------------------------------------------------------------------------------------------------
        
    # Scheduler
    scheduler.step()
    
    is_best, results = False, None
    if rank in [-1, 0]:
        epoch_info = f'【epoch-train】:{epoch+1:3d}\tgpu_mem: {mem}\ttotal: {mloss[0]:.5f}\tbox: {mloss[1]:.5f}\tobj: {mloss[2]:.5f}\tcls: {mloss[3]:.5f}\timgsz: {imgs.shape[-1]}'
        logger.info(epoch_info)
        callbacks.run('on_train_epoch_end', epoch=epoch)
        ema_model.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        
        ## epoch eval
        if val_loader:
            if ema_model:
                evalmodel = ema_model.ema
            else:
                evalmodel = de_parallel(model)
            final_epoch = epoch + 1 == args.epochs
            if not args.noval or final_epoch:
                is_best, results = epoch_eval_yolo5(args, epoch, evalmodel, val_loader, callbacks=None)
    return is_best, results