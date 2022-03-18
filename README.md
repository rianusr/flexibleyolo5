### [Declaration]
> This repo is heavily based on [official_yolo5](https://github.com/ultralytics/yolov5) and [official_yolox](https://github.com/MegEngine/YOLOX). If you are taking an interest in `anchor-based detection >> `[official_yolo5](https://github.com/ultralytics/yolov5) or `anchor-free detection >> `[official_yolox](https://github.com/MegEngine/YOLOX), please feel free to follow official codeRepo to get more innovative and timely updates!

---------------------------------------
## Flexible yolo5 
This repo aims to make yolo5 more flexible for not only easy to `change backbones`, and also have `two heads` (yolo5_head and yolox_head) for your choice! Besides, the new updates support `flexible anchors`, such as:
```
# default yolo5 anchors
ANCHORS      = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

# flexible anchors
ANCHORS1_0   = [[10, 13, 16, 30], None, None]  # means: only put anchors on d1 layer, and two anchors for each map point
ANCHORS2_0_2 = [[10, 13], None, [116, 90]]     # means: put anchors on d1、d3 layers, and only one anchor for each map point
ANCHORS3     = [[10, 13], [30,  61], [116, 90]]# means: put one anchor for each map point for all(d1、d2 and d3) layers

# set up flexible anchors
You can setup your own anchors by modify the param "anchors_fpn" on hyp file (default: data/hyps/hyp.scratch.yaml)!
```
And the following backbones and heads are supported in this repo！

### Backbones

| NetWork          | Variant                  |Model                |
|     :--------:   | :-------:                |:-------:            |
| yolo5            | n、s、m、l、x              |   conv              |
| resnext          | p、n、m、t、s、l、h、g      |  conv              | 
| coatnet          | p、n、m、t、s、l、h、g      |   conv+transformer  |
| convnext         | p、n、m、t、s、l、h、g      |   conv              |
| uniformer        | p、n、m、t、s、l、h、g      |   transformer       |
| swin_transformer | p、n、m、t、s、l、h、g      |   transformer       |
| swin_mlp         | p、n、m、t、s、l、h、g      |   mlp               |

### Heads

| HeadType         | Variant       |
|     :--------:   | :-------:     |
| yolo5            | n、s、m、l、x  |
| yolox            | n、s、m、l、x  | 


## Getting Started
### Train
```bash
python train.py \
    --bkbo_variant ${NetWork}-${Variant} \
    --head_variant ${HeadType}-${Variant} \
    --pretrained path_to_pretrain \
    --data coco.yaml \
    --batch-size 128 \
    --gpu0_bs 40

eg.
# training with yolo5 head
python train.py --bkbo_variant resnext-s --head_variant yolo5-n --data coco.yaml --batch-size 64 --gpu0_bs 10
# training with yolox head
python train.py --bkbo_variant resnext-s --head_variant yolox-s --data coco.yaml --batch-size 48 --gpu0_bs 10

# To quickly verify, automaticly download coco128 datasets(<6M)
python train.py --bkbo_variant yolo5-n --head_variant yolo5-n --batch-size 64
```

### Detect
```
python detect.py --weight path_to_trained_weight --source path_to_img_file
                                                          path_to_img_dir
```

### Export onnx
```
python export.py --weight path_to_trained_weight \
                 --img-size 640 640 \
                 --print-onnx       #! whether to print a human readable model
```

## Params
### `--gpu0_bs`: batch_size of gpu-idx-0
- The default vaule for gpu0_bs is 0, means DataParallel-train while torch.cuda.device_count() > 1.
- If you set gpu0_bs greater than 0, means BalancedDataParallel-train while torch.cuda.device_count() > 1.
    - Please make sure that `(total-batch-size - gpu0_bs) % (torch.cuda.device_count() - 1) = 0`

### `--original_model_benchmark_test`: build original yolo5 detect model for training
Training with param `--original_model_benchmark_test` to build official yolo5 model, and set param `--cfg` for different variants!
```
eg.
python train.py --original_model_benchmark_test --cfg models/yamls/yolov5n.yaml --batch-size 64 
```

## ToDo:
- [x] Add yolox head!
- [ ] Complete the model test!
