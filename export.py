import os
import sys
import json
import time
import torch
import argparse

import torch.nn as nn

sys.path.append(os.path.dirname(__file__).replace('models', ''))  # to run '$ python *.py' files in subdirectories
from models.common import Conv
from models.experimental import attempt_load
from tactics.model_utils import replace_module
from tactics.activations import Hardswish, SiLU

ANCHOR_DEFAULT=[
    [10, 13, 16, 30, 33, 23], 
    [30, 61, 62, 45, 59, 119], 
    [116, 90, 156, 198, 373, 326]]

FLEXIBLE_YOLO5_EXPORT_IDS = {
    'yolo5-x_yolo5-x':[696, 966, 1236],
    'yolo5-l_yolo5-l':[579, 849, 1119],
    'yolo5-m_yolo5-m':[462, 732, 1002],
    'yolo5-s_yolo5-s':[345, 615, 885],
    'yolo5-t_yolo5-t':[345, 615, 885],
    'yolo5-z_yolo5-p':[334, 604, 874],
    'yolo5-z_yolo5-z':[323, 593, 863],
    'yolo5-e_yolo5-e':[323, 593, 863],
}


def update_model_yolox(model):
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    return model


def update_model_yolo5(model):
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.head.model[-1].export = True  # set Detect() layer export=True
    return model 


def update_model(model):
    head_type = str(model.head.__class__).split('.')[2]
    if 'yolox' in head_type:
        opset_version = 11
        return update_model_yolox(model), opset_version, head_type
    else:
        opset_version = 12
        return update_model_yolo5(model), opset_version, head_type


def get_configs(args, model, head_type):
    model_config = {}

    labels = model.names
    stride = model.stride.numpy()
    anchors_train = []
    if 'yolo5' in head_type:
        anchors = model.head.model[-1].anchors.numpy()     ## 实际训练时使用的anchor，尺度分别按照stride进行缩放
        for idx, anc in enumerate(anchors):
            anc_t = anc * stride[idx]
            anc_t = anc_t.tolist()
            anchors_train.append(anc_t)
    model_config["stride"]  = stride.tolist()
    model_config["classes"] = labels
    model_config["anchors"] = anchors_train
    model_config["img_size"] = args.img_size
    model_config["head_type"] = head_type
    model_config["agnostic_nms"] = args.agnostic_nms
    config_file = os.path.dirname(args.weight) + f'/config.json'
    with open(config_file, 'w') as wf:
        json.dump(model_config, wf)
        print(f'Configs saved into: {config_file}')
    
    output_ids = FLEXIBLE_YOLO5_EXPORT_IDS[args.variant]

    ##? add anchors and stride
    model_info = ''
    if args.default_anchor:
        #! only add out_ids 
        model_ids_str = '-'.join(list(map(str, output_ids)))
        model_info += f'-{model_ids_str}' 
    else:
        #! add model_id-stride-anchor
        for idx, (stri, anch) in enumerate(zip(*[stride, anchors_train])):
            model_info += f'-{output_ids[idx]}_{stri}_'                     ## xx means output_id
            anc_str = ''
            for anc in anch:
                anc_s_l = [f"{a:<.0f}".rstrip('0') for a in anc]
                tmp_str = '_'.join(anc_s_l)
                anc_str = anc_str + tmp_str + '_'
            model_info += f'{anc_str[:-1]}'
    return model_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./yolov5s.pt', help='weight path')  # from yolov5/models/
    parser.add_argument('--variant', type=str, default='yolo5-t_yolo5-t', help='variant of model')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--agnostic-nms', action='store_true', help='agnostic nms')
    parser.add_argument('--print-onnx', action='store_true', help='print a human readable model')
    
    parser.add_argument('--default_anchor', action='store_true', help='default anchor --> ANCHOR_DEFAULT')  # from yolov5/models/
    args = parser.parse_args()

    print(args)
    t = time.time() 
    #! Load PyTorch model
    model = attempt_load(args.weight, map_location=torch.device('cpu'))  # load FP32 model
    #! Input
    img = torch.zeros(args.batch_size, 3, *args.img_size)  
    #! Update model
    model, opset_version, head_type = update_model(model)
    model_info = get_configs(args, model, head_type)
    try:
        import onnx
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = os.path.join(os.path.dirname(args.weight), os.path.basename(args.weight).replace('.pt', f'{model_info}.onnx'))  # filename
        torch.onnx.export(model, img, f, opset_version=opset_version, input_names=['images'], output_names=['output'])

        #! Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        if args.print_onnx:
            print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)