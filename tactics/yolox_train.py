import time
import os, sys
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn

from .lr_scheduler import LRScheduler

sys.path.append(os.path.dirname(__file__).replace('tactics', ''))
from tactics.general import LOGGER
from tactics.metrics import ap_per_class
from tactics.datasetsX import postprocess
from tactics.yolox_loss import compute_yolox_loss
from tactics.torch_utils import de_parallel, time_synchronized


def yolox_train_setup(args, train_loader, val_loader, RANK):
    if isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 2:
        img_size = args.imgsz
    elif isinstance(args.imgsz, int):
        img_size = (args.imgsz, args.imgsz)
    else:
        raise ValueError(f'Image size error! Input image size: {args.imgsz}')
    scheduler = LRScheduler(
            args.hyp['scheduler'],
            args.hyp['lr0'],
            len(train_loader),
            args.epochs,
            warmup_epochs=args.hyp['warmup_epochs'],
            warmup_lr_start=args.hyp['warmup_lr_start'],
            no_aug_epochs=args.hyp['no_aug_epochs'],
            min_lr_ratio=args.hyp['min_lr_ratio'],
        )
    evaluator = None
    if RANK in [0, -1]:
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=img_size,
            confthre=args.hyp['confthre'],
            nmsthre=args.hyp['nmsthre'],
            num_classes=args.nc,
            testdev=False,
            )
    return scheduler, evaluator


def get_yolox_optimizer(args, model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(pg0, lr=args.hyp['lr0'], momentum=args.hyp['yolox_momentum'], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": args.hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    return optimizer


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, max_det=300):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader  = dataloader
        self.img_size    = img_size
        self.confthre    = confthre
        self.nmsthre     = nmsthre
        self.num_classes = num_classes
        self.testdev     = testdev
        self.max_det     = max_det

    def evaluate(self, model, half=False, decoder=None, nc=81):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []

        inference_time = 0
        nms_time = 0
        
        ##! for yolo5 format logging
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        stats = []
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        seen = 0
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        pbar = tqdm(self.dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(pbar):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                ##! for yolo5 format logging
                stats, seen = self.cal_metrics(targets, outputs, info_imgs, stats, seen)
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))
        
        # Compute metrics       ##! for yolo5 format logging
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names) #, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)    
        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        # logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))    
        return mp, mr, map50, map

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def xyxy2xywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes
    
    def process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        labels = labels.to(detections.device)
        
        def box_iou(box1, box2):
            # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
            """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                box1 (Tensor[N, 4])
                box2 (Tensor[M, 4])
            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
            """

            def box_area(box):
                # box = 4xn
                return (box[2] - box[0]) * (box[3] - box[1])

            area1 = box_area(box1.T)
            area2 = box_area(box2.T)

            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
            return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
        
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
    def scale_coords(self, img0_shape, coords, ratio, pad):
        coords[:, :4] = self.xywh2xyxy(coords[:, :4])
        coords[:, :4] /= ratio
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= ratio
        self.clip_coords(coords, img0_shape)
        return coords
    
    def cal_metrics(self, targets, outputs, info_imgs, stats, seen):
        iouv = torch.linspace(0.5, 0.95, 10).to('cuda:0')  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        
        #? outputs_scales
        for (target, output, img_h, img_w) in zip(targets, outputs, info_imgs[0], info_imgs[1]):
            seen += 1
            if output is None:
                continue
            ##! process labels
            labels = target[target[:, 0] > 0, :]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            if len(outputs) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            output = output.cpu()
            if self.max_det > 0:    ## 取 max_det 数量
                pred = output[:self.max_det, :]
                
            pred = output.clone()
            bboxes = pred[:, 0:4]
            self.clip_coords(bboxes, self.img_size)

            cls = pred[:, 6]
            scores = pred[:, 4] * pred[:, 5]
            predn = torch.zeros((scores.shape[0], 6)).to('cuda:0')
            predn[:, :4] = bboxes
            predn[:, 4] = scores
            predn[:, 5] = cls
            # Evaluate
            if nl:
                tbox = self.xywh2xyxy(labels[:, 1:5])  # target boxes
                # tbox /= scale
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = self.process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), predn[:, 4].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)
        return stats, seen
    
    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale
            bboxes = self.xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # try:
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list


def epoch_eval_yolox(args, epoch, evalmodel, evaluator):
    P, R, ap50, ap50_95 = evaluator.evaluate(evalmodel, args.fp16, nc=args.nc)
    args.best_fitness = max(args.best_fitness, ap50_95)
    val_info = f'【epoch-val】:{epoch+1:3d}\tP: {P:.5f}\tR: {R:.5f}\tmAP@.5: {ap50:.5f}\tmAP@.5:.95: {ap50_95:.5f}'
    logger.info(val_info)
    return args.best_fitness == ap50_95, (P, R, ap50, ap50_95)


def train_one_epoch_with_yolox_head(args, epoch, model, ema_model, prefetcher, train_loader, val_loader, 
                                    optimizer, scaler, lr_scheduler, evaluator, device, RANK):
    ##! step 1 判断是否需要关闭 mosaic, 如果需要关闭，则head需要设置 head.use_l1 = True, eval_interval = 1
    if args.no_aug or epoch + 1 == args.epochs - args.no_aug_epochs:
        de_parallel(model).head.use_l1 = True
        
    data_type = torch.float16 if args.fp16 else torch.float32
    amp_training = args.fp16

    ## step 2
    cur_iter = 0
    mloss = torch.zeros(5, device=device)  # mean losses
    
    pbar = enumerate(range(len(train_loader)))
    LOGGER.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'total', 'iou', 'l1', 'obj', 'cls', 'labels', 'img_size'))
    if RANK in [-1, 0]:
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for cur_iter, _ in pbar:
        args.progress_in_iter = epoch * len(train_loader) + cur_iter
        inps, targets = prefetcher.next()
        inps = inps.to(data_type)
        targets = targets.to(data_type)
        targets.requires_grad = False
        
        with torch.cuda.amp.autocast(enabled=amp_training):
            outputs, x_shifts, y_shifts, expanded_strides, origin_preds = model(inps)
            lossOuts, loss_items = compute_yolox_loss(inps, x_shifts, y_shifts, expanded_strides, 
                                          targets, torch.cat(outputs, 1), origin_preds, 
                                          use_l1=False, num_classes=args.nc)
        loss = lossOuts['total_loss']
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ema_model:
            ema_model.update(model)
            
        lr = lr_scheduler.update_lr(args.progress_in_iter + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        ##! yolo5 format log
        if RANK in [-1, 0]:
            mloss = (mloss * cur_iter + loss_items) / (cur_iter + 1)        # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 7) % (f'{epoch+1}/{args.epochs}', mem, *mloss, targets.shape[0], inps.shape[-1]))

    is_best, results = False, None
    if RANK in [-1, 0]:
        ##! record to runtime.log
        epoch_info = f'【epoch-train】:{epoch+1:3d}\tgpu_mem: {mem}\ttotal: {mloss[0]:.5f}\tiou: {mloss[1]:.5f}\tl1: {mloss[2]:.5f}\tobj: {mloss[3]:.5f}\tcls: {mloss[4]:.5f}\timgsz: {inps.shape[-1]}'
        logger.info(epoch_info)
        
        ##! epoch eval
        if val_loader:
            if ema_model:
                evalmodel = ema_model.ema
            else:
                evalmodel = de_parallel(model)
            final_epoch = epoch + 1 == args.epochs
            if not args.noval or final_epoch:
                is_best, results = epoch_eval_yolox(args, epoch, evalmodel, evaluator)
    return is_best, results