import os
import sys
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from lxml import etree
from loguru import logger

import torch
from tactics.datasetsX import postprocess, LoadImagesX
from tactics.torch_utils import select_device, time_sync
from tactics.datasets import LoadImages
from tactics.general import LOGGER, non_max_suppression, scale_coords, increment_path

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

_COLORS = np.array([
    0.000, 1.000, 0.000, 0.850, 0.325, 0.098, 0.000, 0.000, 1.000, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 
    0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0]
    ).astype(np.float32).reshape(-1, 3)

NAMES = ['background', 'down', 'left', 'others', 'right', 'up', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img


def writeXmlForBoxes(xmlFile, boxes):
    with open(xmlFile, 'w') as wf:
        wf.write('<annotation>\n')
        wf.write('</annotation>\n')
    
    output_parse = etree.parse(xmlFile)
    for control, ann_out in enumerate(output_parse.iter('annotation')):
        if control >= 1:
            break
        for box in boxes:
            score = 0.0
            if len(box) == 5:
                name, xmin, ymin, xmax, ymax = box[:5]
            elif len(box) == 6:
                name, xmin, ymin, xmax, ymax, score = box[:6]            
            
            obj_elem = etree.Element('object')
            ann_out.append(obj_elem)

            obj_elem.append(etree.Element('name'))
            obj_elem.find('name').text = name
            
            obj_elem.append(etree.Element('score'))
            obj_elem.find('score').text = str(score)

            obj_elem.append(etree.Element('bndbox'))
            obj_elem.find('bndbox').append(etree.Element('xmin'))
            obj_elem.find('bndbox').append(etree.Element('ymin'))
            obj_elem.find('bndbox').append(etree.Element('xmax'))
            obj_elem.find('bndbox').append(etree.Element('ymax'))

            obj_elem.find('bndbox').find('xmin').text = str(xmin)
            obj_elem.find('bndbox').find('ymin').text = str(ymin)
            obj_elem.find('bndbox').find('xmax').text = str(xmax)
            obj_elem.find('bndbox').find('ymax').text = str(ymax)
    output_parse.write(xmlFile)


def writeXml(img_shape, boxes, scores, cls_ids, class_names, conf, xml_file):
    boxes_info = []
    height, width = img_shape[:2]
    for i in range(len(boxes)):
        box = boxes[i]
        name = class_names[int(cls_ids[i])]
        score = np.around(scores[i].item(), 8)
        if score < conf:
            continue
        x0 = max(0, int(box[0]))
        y0 = max(0, int(box[1]))
        x1 = min(int(box[2]), width)
        y1 = min(int(box[3]), height)
        boxes_info.append([name, x0, y0, x1, y1, score])
    writeXmlForBoxes(xml_file, boxes_info)


def visual(output, img_info, save_result, vis_folder, xml_folder, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, NAMES)
    
    xml_file = os.path.join(xml_folder, img_info["file_name"].rsplit('.', 1)[0] + '.xml')
    writeXml(img_info, bboxes, scores, cls, NAMES, cls_conf, xml_file)
    if save_result:
        save_file_name = os.path.join(vis_folder, img_info["file_name"])
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, vis_res)
    return vis_res


def postprocess_yolo5(args, pred, imgsz, img_shape):
    outputs = []
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, agnostic=args.agnostic_nms, max_det=args.max_det)
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(imgsz, det[:, :4], img_shape).round()
            outputs.append(det)
    if not outputs:
        return None, None, None
    outputs = torch.vstack(outputs) 
    return outputs[:, :4], outputs[:, 4], outputs[:, 5]


def postprocess_yolox(args, outputs, imgsz, img_shape, decoder=None):
    ratio = min(imgsz[0] / img_shape[0], imgsz[1] / img_shape[1])
    if decoder is not None:
        outputs = decoder(outputs, dtype=outputs.type())
    outputs = postprocess(outputs, args.num_classes, args.conf_thres, args.iou_thres, class_agnostic=args.agnostic_nms)[0]
    if outputs is None:
        return None, None, None
    outputs = outputs.cpu()
    bboxes = outputs[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    return bboxes, scores, cls


def parse_config(args, config_file):
    with open(config_file, 'r') as rf:
        configs           = json.load(rf)
        args.names        = configs['classes']
        args.anchors      = configs['anchors']
        args.stride       = configs['stride']
        args.head_type    = configs.get('head_type', 'yolo5_head')
        args.agnostic_nms = configs.get('agnostic_nms', True)


def get_infer_model(args):
    if args.onnx_infer:
        config_file = os.path.dirname(args.weight) + f'/config.json'
        import onnxruntime
        # create session
        model = onnxruntime.InferenceSession(args.weight)
        args.onnx_input_name = model.get_inputs()[0].name
        parse_config(args, config_file)
    elif args.weight.endswith('.pt') or args.weight.endswith('.pth'):
        model = torch.load(args.weight, map_location=args.device)['model']
        args.head_type = str(model.head.__class__).split('.')[2]
        args.names     = model.module.names if hasattr(model, 'module') else model.names
        args.stride    = model.stride.cpu().numpy()
        if 'yolo5' in args.head_type:
            args.anchors = model.head.model[-1].anchors.cpu().numpy()     ## 实际训练时使用的anchor，尺度分别按照stride进行缩放
        else:
            args.anchors = []
        model = model.half() if args.fp16 else model.float()
        model.to(args.device)
        model.eval()
    args.num_classes = len(args.names)
    return model


def inference(args, model, img):
    t1 = time_sync()
    if args.onnx_infer:
        img = np.ascontiguousarray(img).astype(np.float32)
        if 'yolo5' in args.head_type:
            img /= 255.0
        img = img[np.newaxis, ...]
        t2 = time_sync()
        results = model.run([], {args.onnx_input_name: img})
        pred = torch.from_numpy(results[0])
        t3 = time_sync()
    else:
        img = torch.from_numpy(img).to(args.device)
        img = img.half() if args.fp16 else img.float()  # uint8 to fp16/32
        if 'yolo5' in args.head_type:
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        with torch.no_grad():
            pred = model(img)
            if isinstance(pred, tuple):
                pred = pred[0]
        t3 = time_sync()
    return pred, t1, t2, t3


def speed_assessment(args, model, dataset, times=100):
    dt, seen = [0.0, 0.0, 0.0], 0
    for idx in range(times):
        print(f'speed_assessment: >>>> {idx+1}')
        for _, im, _, _, _ in tqdm(dataset):
            seen += 1
            #! inference
            _, t1, t2, t3 = inference(args, model, im)
            ## time-consuming add up
            dt[0] += t2 - t1
            dt[1] += t3 - t2
            dt[2] += time_sync() - t3
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms PostProcess per image at shape {(1, 3, *args.imgsz)}' % t)


def detect_run(args, times=100):
    if args.weight.endswith('.onnx'):
        logger.info('Flexibleyolo5 inference with onnx model!')
        args.onnx_infer = True
    elif args.weight.endswith('.pt') or args.weight.endswith('.pth'):
        logger.info('Flexibleyolo5 inference with torch model!')
        args.onnx_infer = False
    else:
        raise ValueError(f'Invalid model file: {args.weight}')
    
    args.device = select_device(args.device)
    #! Load model
    model = get_infer_model(args)

    #! load dataset
    if args.head_type == 'yolo5_head':
        dataset = LoadImages(args.source, img_size=args.imgsz, stride=args.stride, auto=False)
    else:
        dataset = LoadImagesX(args.source, img_size=args.imgsz)
    
    if args.speed_assessment:
        LOGGER.critical(f'Speed_assessment mode: assessment for {times} times!')
        speed_assessment(args, model, dataset, times=times)
        LOGGER.info(f'Speed_assessment Done!')
        return 

    #! outputs save dir
    save_dir = increment_path(Path(args.output) / args.head_type, exist_ok=False, mkdir=True)
    vis_dir  = save_dir / 'visual_res'
    xml_dir  = save_dir / 'Annotations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f'     Visual output: {vis_dir}')
    LOGGER.info(f'Annotations output: {xml_dir}')
    
    #! Run inference
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0, _, _ in tqdm(dataset):
        img_name = os.path.basename(path)
        
        seen += 1
        #! inference
        pred, t1, t2, t3 = inference(args, model, im)
        
        #! postprosess
        if args.head_type == 'yolo5_head':
            bboxes, scores, cls = postprocess_yolo5(args, pred, args.imgsz, im0.shape)
        else:
            bboxes, scores, cls = postprocess_yolox(args, pred, args.imgsz, im0.shape)
        
        dt[0] += t2 - t1
        dt[1] += t3 - t2
        dt[2] += time_sync() - t3
        
        if args.draw_boxes:
            if bboxes is None:
                vis_res = im0
                img_out = os.path.join(vis_dir, '0', img_name)
                if not os.path.exists(os.path.dirname(img_out)):
                    os.makedirs(os.path.dirname(img_out))
            else:
                vis_res = vis(im0, bboxes, scores, cls, args.conf_thres, args.names)
                img_out = os.path.join(vis_dir, img_name)
            cv2.imwrite(img_out, vis_res)
        
        if args.save_xml and bboxes is not None:
            xml_file = img_out = os.path.join(xml_dir, f"{img_name.rsplit('.', 1)[0]}.xml")
            writeXml(im0.shape, bboxes, scores, cls, args.names, args.conf_thres, xml_file)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms PostProcess per image at shape {(1, 3, *args.imgsz)}' % t)
    
    
def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='detect source')
    parser.add_argument('--output', type=str, default=ROOT / 'runs/detect', help='detect source')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='detect img size')
    parser.add_argument('--weight', type=str, default='', help='model file for detect')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--agnostic_nms', action='store_true', help='training for coco datasets')
    parser.add_argument('--max_det', type=int, default=300, help='keep max det while inference')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='training for coco datasets')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='training for coco datasets')
    parser.add_argument('--draw_boxes', action='store_true', default=True, help='draw_boxes')
    parser.add_argument('--save_xml', action='store_true', default=True, help='save_xml')
    parser.add_argument('--speed_assessment', action='store_true', help='speed_assessment')
    
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    detect_run(args)
