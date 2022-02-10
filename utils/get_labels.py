import os
from glob import glob
from tqdm import tqdm
from lxml import etree


coco_nc = 80  # number of classes
coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names


def get_all_files(data_dir, postfix='jpg'):
    all_files = []
    for root, _, _ in os.walk(data_dir):
        # if '/trainData' not in root and 'valData' not in root and 'testData' not in root:
        #     continue
        files = glob(root + '/*.{}'.format(postfix))
        all_files.extend(files)
    return all_files


def get_classes(data_dir, postfix='xml'):
    all_xmls = get_all_files(data_dir, postfix=postfix)
    classes = []
    for xml in tqdm(all_xmls):
        annotation = etree.parse(xml)
        for object in annotation.iter('object'):
            name = object.find('name').text
            if name not in classes:
                classes.append(name)
    classes.sort()
    return classes