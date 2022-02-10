## Flexible yolo5 
[Declaration]This repo is heavily based on [official_yolo5](https://github.com/ultralytics/yolov5). If you
are taking an interest in `anchor-based detection >> `[official_yolo5](https://github.com/ultralytics/yolov5), please feel free to follow official codeRepo to get more innovative and timely updates!

This repo aims to make yolo5 more flexible to change backbones! 
And the following backbones are supported in this repo！

### Backbones

| NetWork          | Variant                                       |Model                |
|     :--------:   | :-------:                                     |:-------:            |
| yolo5            | n、s、m、l、x                                  |   conv              |
| resnext          | 50、101、152                                   |  conv              | 
| coatnet          | 0、1、2、3、4                                  |   conv+transformer  |
| convnext         | t、s、b、l、xl                                 |   conv              |
| uniformer        | s、p、b、ls                                    |   transformer       |
| swin_transformer | tiny_det、small_det、base_det、 large_det      |   transformer       |
| swin_mlp         | tinyc24_det、tinyc12_det、tinyc6_det、base_det |   mlp               |
|                  |                                               |                     |


### Heads

| HeadType         | Variant       |
|     :--------:   | :-------:     |
| yolo5            | n、s、m、l、x  |
|                  |               | 

---------------------------------------
## Getting Started
### Train
```bash
python train.py --bkbo_variant ${NetWork}-${Variant} --data coco.yaml --batch-size 128 --gpu0_bs 40 --pretrained path_to_pretrain \
    --head_variant yolo5-n
                        -s
                        -m
                        -l
                        -x

eg.
python train.py --bkbo_variant resnext-50 --head_variant yolo5-n --data coco.yaml --batch-size 30 --gpu0_bs 10

# To quickly verify, automaticly download coco128 datasets(<6M)
python train.py --bkbo_variant yolo5-n --head_variant yolo5-n --batch-size 64
```

### Detect
```
python detect.py --weights path_to_trained_weight --source img.jpg             
                                                           path/*.jpg  # glob
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
- [ ] Add yolox head!
- [ ] Complete the model test!
