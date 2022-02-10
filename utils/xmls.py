import os
import numpy as np
from lxml import etree
from easydict import EasyDict as edict

import torch

all_colors = [
    (49, 23, 241),
    (4, 206, 42),
    (237, 216, 50),
    (231, 19, 124),
    (55, 132, 251),
    (89, 221, 254),
    (27, 95, 143),
    (132, 202, 230),
    (12, 72, 29),
    (81, 39, 131),
    (58, 92, 253),
    (197, 6, 208),
    (243, 125, 21)]

def write_res_2_xml(res_dict_lst, xml_file):
    '''
    res_dict: [{name:0, bndbox:{xmin:0, ymin:0,xmax:0,ymax:0}}, ...]
    xml_file: <str> path_to_xml/xml_name.xml
    '''
    if not os.path.exists(os.path.dirname(xml_file)):
        os.makedirs(os.path.dirname(xml_file))
        print('Make new xml dir: {}'.format(os.path.dirname(xml_file)))

    with open(xml_file, 'w') as wf:
        wf.write('<annotation>\n')
        wf.write('</annotation>\n')

    xml_parse = etree.parse(xml_file)
    # annotation = xml_parse.find('annotation')
    for _, annotation in enumerate(xml_parse.iter('annotation')):
        # 构建每个的object模块
        for i in range(len(res_dict_lst)):
            annotation.append(etree.Element('object'))

        for idx, object in enumerate(annotation.iter('object')):
            name = res_dict_lst[idx]["name"]
            bndbox = res_dict_lst[idx]["bndbox"]
            score = -1
            if 'score' in res_dict_lst[idx]:
                score = res_dict_lst[idx]['score']

            if int(float(score)) >= 0:
                object.append(etree.Element('score'))
                object.find('score').text = str(score)

            object.append(etree.Element('name'))
            object.find('name').text = name
            object.append(etree.Element('bndbox'))
            object.find('bndbox').append(etree.Element('xmin'))
            object.find('bndbox').append(etree.Element('ymin'))
            object.find('bndbox').append(etree.Element('xmax'))
            object.find('bndbox').append(etree.Element('ymax'))

            object.find('bndbox').find('xmin').text = str(bndbox['xmin'])
            object.find('bndbox').find('ymin').text = str(bndbox['ymin'])
            object.find('bndbox').find('xmax').text = str(bndbox['xmax'])
            object.find('bndbox').find('ymax').text = str(bndbox['ymax'])

    xml_parse.write(xml_file)
    
def convert_coco_2_idt(bboxes, w, h, xml_file):
    single_res = []
    for bbox in bboxes:
        name, x_c, y_c, box_w, box_h, score = bbox
        center_box = np.array(list(map(float, [x_c, y_c, box_w, box_h])))
        center_box[0] *= w
        center_box[2] *= w
        center_box[1] *= h
        center_box[3] *= h
        xmax = int((center_box[2] + 2 * center_box[0]) / 2)
        ymax = int((center_box[3] + 2 * center_box[1]) / 2)
        xmin = int((2 * center_box[0] - center_box[2]) / 2)
        ymin = int((2 * center_box[1] - center_box[3]) / 2)
        single_res.append(edict({"name": name,
                                 "width": w,
                                 "height": h,
                                 "score": score,
                                 "bndbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}))
    write_res_2_xml(single_res, xml_file)
    
def little_box_base_iou(pred, thres=0.7):
    pred_numpy = pred[0].cpu().numpy()
    pred_len = len(pred_numpy)
    suppressed = np.ones(pred_len)
    for i in range(pred_len):
        if suppressed[i] == 0:
            continue
        for j in range(i+1, pred_len):
            if suppressed[j] == 0:
                continue
            box_A = pred_numpy[i]
            box_B = pred_numpy[j]
            area_A = (box_A[3] - box_A[1]) * (box_A[2] - box_A[0])
            area_B = (box_B[3] - box_B[1]) * (box_B[2] - box_B[0])
            
            inner_xmin = max(box_A[0], box_B[0])
            inner_ymin = max(box_A[1], box_B[1])
            inner_xmax = min(box_A[2], box_B[2])
            inner_ymax = min(box_A[3], box_B[3])
            inner_w = max(inner_xmax - inner_xmin, 0)
            inner_h = max(inner_ymax - inner_ymin, 0)
            inner_area = inner_w * inner_h
            iou = inner_area / min(area_A, area_B)
            if iou>thres:
                idx = i if area_A > area_B else j
                suppressed[idx] = 0
    pred = [pred[0][torch.from_numpy(suppressed).bool()]]
    return pred

def get_xml_elem_dict(im0, xyxy, names, cls, conf):
    xmin, ymin, xmax, ymax = [np.round(x.item()) for x in xyxy]
    pred_name = names[int(cls.item())]                              
    return edict({"name": pred_name,
                    "width": im0.shape[1], 
                    "height": im0.shape[0],
                    "score": conf.item(),
                    "bndbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}})