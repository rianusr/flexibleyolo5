#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by rianusr
# Create on 2021/1/14

import os
import cv2
import json
import codecs
import numpy as np
from tqdm import tqdm
from glob import glob
from easydict import EasyDict as edict

coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif']
class Labels2Jsons():
    def __init__(self, label_path, image_path, is_train, classes=[], img_formats=img_formats):
        self.label_path  = label_path
        self.image_path  = image_path
        self.is_train    = is_train
        self.classes     = classes
        self.img_formats = img_formats
        
        self.coco = edict({
            'images':[],
            'type':'instances',
            'annotations':[],
            'categories':[],
        })
        self.category_set = dict()
        self.image_set = set()

        self.image_id = 20200000000
        self.annotation_id = 0
        self.voc_txt_lines = []
        self.category_ids_set = set()
        self.category_ids_list = []
        self.initAllCatItem(label_path)
        self.parseLabels()

    def addImgItem(self, file_name, size):
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name

        im_name = file_name.strip().split('.j')[0]
        self.voc_txt_lines.append(im_name)

        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, image_id, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        annotation_item['segmentation'].append(seg)

        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)

    def initAllCatItem(self, label_path):
        print('Initializing all categories ...')
        # in case train and val category_id misalign
        for f in os.listdir(label_path):
            if not f.endswith('.txt'):
                continue
            label_file = os.path.join(label_path, f)
            with open(label_file, 'r') as rf:
                l_data = rf.readlines()
                for line in l_data:
                    object_name = line.split(' ')[0]
                    # if self.classes:
                    #     object_name = self.classes[int(object_name)]
                    self.category_ids_set.add(object_name)
        self.category_ids_list = list(self.category_ids_set)
        
        self.category_ids_list.sort(key=lambda x:int(x))
        category_item_id = 0
        for category_id in self.category_ids_list:
            if self.classes:
                obj_name = self.classes[int(category_id)]
            else:
                obj_name = str(category_id)
            
            category_item_id += 1
            category_item = edict({
                'supercategory':'none',
                'id':category_item_id,
                'name':obj_name
            })

            self.coco['categories'].append(category_item)
            self.category_set[obj_name] = category_item_id

    def parseLabels(self):
        labels = [l for l in os.listdir(self.label_path) if l.endswith('.txt')]
        images = [im for im in os.listdir(self.image_path) if im.rsplit('.', 1)[-1] in self.img_formats]
        assert self.data_check(images, labels), "image files not match with label files!"
        print(f'Parsing labels for {len(images)} images!')
        
        for f in tqdm(os.listdir(self.label_path)):
            if not f.endswith('.txt'):
                continue
            label_file = os.path.join(self.label_path, f)
            file_name = f.replace('.txt', '.jpg')
            assert file_name not in self.image_set
            img_file = os.path.join(self.image_path, file_name)
            h, w, c = cv2.imread(img_file).shape
            size = edict({'width' : w, 'height' : h, 'depth' : c})        

            current_image_id = self.addImgItem(file_name, size)
            with open(label_file, 'r') as rf:
                label_data = rf.readlines()
                for ld in label_data:
                    object_name, xmin, xmax, ymin, ymax = self.parse_label_line(ld, w, h)
                    # current_image_id = self.addImgItem(file_name, size)
                    current_category_id = self.category_set[object_name]
                    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
                    self.addAnnoItem(current_image_id, current_category_id, bbox)

    def parse_label_line(self, line, w, h):
        object_name, x_c, y_c, box_w, box_h = line.split(' ')[:5]
        if self.classes:
            object_name = self.classes[int(object_name)]
        center_box = np.array(list(map(float, [x_c, y_c, box_w, box_h])))
        center_box *= [w, h, w, h]

        xmax = int((center_box[2] + 2 * center_box[0]) / 2)
        ymax = int((center_box[3] + 2 * center_box[1]) / 2)
        xmin = int((2 * center_box[0] - center_box[2]) / 2)
        ymin = int((2 * center_box[1] - center_box[3]) / 2)
        return object_name, xmin, xmax, ymin, ymax
    
    def exportCocoJsonFile(self, json_file):
        json.dump(self.coco, open(json_file, 'w'))
        print('Json file saved into: {}'.format(json_file))

    def exportClassesFile(self, outputTxt):
        outfile = codecs.open(outputTxt, 'w', encoding='utf-8')
        outfile.write(str(self.category_ids_list) + '\n')
        print('Classes file saved into: {}'.format(outputTxt))
    
    def data_check(self, images, labels):
        images.sort()
        labels.sort()
        img_names = [im.rsplit('.', 1)[0] for im in images]
        lbl_names = [lb.rsplit('.', 1)[0] for lb in labels]
        return img_names == lbl_names
    
    def __call__(self, json_file=None, class_file=None):
        if json_file is None:
            if self.is_train:
                json_name = 'instances_trainData.json'
            else:
                json_name = 'instances_valData.json'
            json_file = os.path.join(self.image_path.replace('/images', '/annotations'), json_name)
            if not os.path.exists(os.path.dirname(json_file)):
                os.makedirs(os.path.dirname(json_file))
        self.exportCocoJsonFile(json_file)
        if class_file is not None:
            self.exportClassesFile(class_file)


def check():
    all_labs = []
    labels = '/data/datasets/cocoDatasets/tinyCoco/trainData/labels'
    for txt_file in glob(labels + '/*.txt'):
        with open(txt_file, 'r') as rf:
            data = rf.readlines()
            for line in data:
                obj_id = line.split(' ')[0]
                if obj_id not in all_labs:
                    all_labs.append(obj_id)
    print(len(all_labs))


def check_jsons():
    js_file1 = '/data/datasets/cocoDatasets/tinyCoco/trainData/annotations/instances_trainData.json'
    js_file2 = '/data/datasets/cocoDatasets/tinyCoco/trainData/instances_trainData.json'
    
    with open(js_file1, 'r') as rf1:
        js_data1 = json.load(rf1)
    
    with open(js_file2, 'r') as rf2:
        js_data2 = json.load(rf2)
    
    js_str1 = str(js_data1)
    js_str2 = str(js_data2)

    word_count1 = {}
    word_count2 = {}
    for ite in js_str1:
        if ite not in word_count1:
            word_count1[ite] = 1
        else:
            word_count1[ite] += 1
    
    for ite in js_str2:
        if ite not in word_count2:
            word_count2[ite] = 1
        else:
            word_count2[ite] += 1
    word_count1 = sorted(word_count1.items())
    word_count2 = sorted(word_count2.items())
    for idx, item in enumerate(word_count1):
        if item != word_count2[idx]:
            print(item, word_count2[idx])
    if word_count1 == word_count2:
        print('Two jsons are same!')
    else:
        print('Two different jsons!')
    

if __name__ == "__main__":
    # train_label_path = '../datasets/coco128/labels/train2017'
    # train_image_path = '../datasets/coco128/images/train2017'
    # val_label_path = '../datasets/coco128/labels/train2017'
    # val_image_path = '../datasets/coco128/images/train2017'
    # classes = coco_names
    train_image_path = '/data/datasets/cocoDatasets/tinyCoco/trainData/images'
    train_label_path = '/data/datasets/cocoDatasets/tinyCoco/trainData/labels'
    val_image_path = '/data/datasets/cocoDatasets/tinyCoco/valData/images'
    val_label_path = '/data/datasets/cocoDatasets/tinyCoco/valData/labels'
    classes = ['down', 'left', 'others', 'right', 'up']
    # Labels2Jsons(train_label_path, train_image_path, classes=coco_names, is_train=True)()
    Labels2Jsons(val_label_path, val_image_path, classes=coco_names, is_train=False)()
