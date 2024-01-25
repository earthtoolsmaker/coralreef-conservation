# Experiments Logbook

I keep track of all experiments I run for the coral reefs challenge in this
experiments logbook.

Datasets uploaded on the GPU instance:

- `/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets`: per region dataset - can be used for training yolov8 - it handles mislabelled datapoints by removing them from the set
- `/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2`: handles mislabelled datatpoints and empty masks (black masks) - ~18% were empty masks, mostly in `SEAVIEW_ATL`

## YOLOv8

### Object Detection

#### All regions but SEAVIEW_PAC_AUS

##### Baseline

###### Parameters

- epochs: `20`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `object/train`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8m.pt epochs=20
```

###### Performance

```sh
Validating runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
Model summary (fused): 218 layers, 25840918 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all        410       7337      0.646      0.535      0.574      0.402
            hard_coral        410       6143      0.685      0.566      0.623      0.453
            soft_coral        410       1194      0.606      0.503      0.526      0.351
Speed: 0.1ms preprocess, 3.4ms inference, 0.0ms loss, 1.3ms postprocess per image
```

##### Large Model + big imgsz + close mosaic

###### Take Away

TODO

###### Parameters

- epochs: `200`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `40`
- data: handled label mismatch
- all other parameters are left as default
- train run: `object/trainXXX`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x.pt epochs=200 imgsz=1024 close_mosaic=40
```

###### Performance

```sh
Validating runs/detect/train2/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
Model summary (fused): 268 layers, 68125494 parameters, 0 gradients, 257.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  15%|█▌        | 2/13 [00:01<00:08,  1.27it/s]WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 13/13 [00:20<00:00,  1.58s/it]
                   all        410       7337      0.668       0.57      0.603      0.434
            hard_coral        410       6143      0.658      0.611       0.64      0.477
            soft_coral        410       1194      0.678      0.529      0.565       0.39
Speed: 0.3ms preprocess, 21.6ms inference, 0.0ms loss, 6.8ms postprocess per image
```

### Instance Segmentation

#### All regions but SEAVIEW_PAC_AUS

##### Baseline

###### Parameters

- epochs: `20`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train5`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8m-seg.pt epochs=20
```

###### Performance

```sh
Validating runs/segment/train5/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R
                   all        410       7337      0.663      0.521      0.575      0.409       0.66      0.519      0.567      0.357
            hard_coral        410       6143      0.697      0.552      0.626      0.457      0.694       0.55       0.62      0.405
            soft_coral        410       1194      0.628       0.49      0.524      0.361      0.625      0.487      0.514      0.309
Speed: 0.1ms preprocess, 4.3ms inference, 0.0ms loss, 2.5ms postprocess per image
```

##### Baseline without empty masks

###### Parameters

- epochs: `20`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train23`
- dataset: `benthic_datasets2`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8m-seg.pt epochs=20
```

###### Performance

```sh
Validating runs/segment/train23/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        349       7359      0.657      0.557      0.607       0.43      0.664      0.551        0.6      0.388
            hard_coral        349       5921      0.704      0.552      0.634       0.47      0.711      0.545      0.632      0.417
            soft_coral        349       1438      0.611      0.561       0.58       0.39      0.617      0.556      0.567      0.359
Speed: 0.2ms preprocess, 4.5ms inference, 0.0ms loss, 1.5ms postprocess per image
```

##### Large Model + long training time

###### Take Away

Long training time (`80` epochs):
The model can keep learning for a very large amount of epochs, 80 still seems 
too little as the learning curves still show that there is learning potential.
The default epochs value from yolov8 training is set to `100`, try to use 20% 
more (= `120`) for the final model.

Large Model (`x` size)
The model has better performance with more parameters. as there are no 
constraints for the model in terms of RAM usage, model size, we should use the 
big model size `x` for the selected model.

###### Parameters

- epochs: `80`
- model size: `x`
- all other parameters are left as default
- train run: `segment/train`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=80
```

###### Performance

```sh
Validating runs/segment/train/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        410       7337      0.701      0.552      0.601      0.438      0.703      0.549      0.598      0.386
            hard_coral        410       6143      0.696      0.596      0.639      0.473      0.694       0.59      0.633      0.421
            soft_coral        410       1194      0.706      0.508      0.563      0.402      0.712      0.509      0.563      0.351
Speed: 0.1ms preprocess, 10.3ms inference, 0.0ms loss, 2.0ms postprocess per image
```

##### Large Model + Long long training time

###### Take Away

###### Parameters

- epochs: `200`
- model size: `x`
- all other parameters are left as default
- train run: `segment/train18`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=200
```

###### Performance

```sh
```

##### Large Model + large imgsz

###### Take Away

using larger `imgsz` lead to faster learning (it takes less epochs to get better
losses). Using a too large `imgsz` leads to memory issue as the GPU is constrained.
Use a large `imgsz` for training the final model.

###### Parameters

- epochs: `20`
- model size: `x`
- imgsz: `1024`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train4`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=20 imgsz=1024
```

###### Performance

```sh
Validating runs/segment/train4/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        410       7337      0.667      0.547      0.607      0.452      0.671      0.551      0.605       0.41
            hard_coral        410       6143      0.685      0.588      0.655      0.501      0.689      0.592      0.656      0.465
            soft_coral        410       1194       0.65      0.507       0.56      0.403      0.654       0.51      0.554      0.356
Speed: 0.5ms preprocess, 26.5ms inference, 0.0ms loss, 2.0ms postprocess per image
```

##### Augmentation - degrees

###### Take Away

Increasing the data augmentation `degree` parameter does not seem to lead to 
better performance (compared to the baseline which uses by default a `degree = 0`)

###### Parameters

- epochs: `20`
- model size: `m`
- data augmentation degrees: `180`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train7`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=20 degrees=180
```

###### Performance

```sh
Validating runs/segment/train7/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P
                   all        410       7337      0.618      0.512      0.557      0.395      0.622      0.515      0.553      0.357
            hard_coral        410       6143      0.664      0.558      0.615      0.448      0.664      0.558      0.611      0.408
            soft_coral        410       1194      0.571      0.466      0.498      0.341       0.58      0.473      0.494      0.305
Speed: 0.2ms preprocess, 4.3ms inference, 0.0ms loss, 3.2ms postprocess per image
```

##### Large Model + large imgsz + long training time

###### Take Away

###### Parameters

- epochs: `120`
- model size: `x`
- imgsz: `1024`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train13`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=120 imgsz=1024
```

###### Performance

```sh
Validating runs/segment/train13/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        410       7337       0.68       0.58      0.616      0.455      0.684      0.586      0.616      0.416
            hard_coral        410       6143      0.677      0.619      0.659      0.502      0.682      0.628      0.665      0.473
            soft_coral        410       1194      0.683       0.54      0.572      0.408      0.687      0.545      0.567      0.359
Speed: 0.4ms preprocess, 26.7ms inference, 0.0ms loss, 1.8ms postprocess per image
```


##### Large Model + big imgsz + Long training time + large close_mosaic

###### Take Away

###### Parameters

- epochs: `200`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `40`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train14`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=200 imgsz=1024 close_mosaic=40
```

###### Performance

```sh
Validating runs/segment/train14/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P
                   all        410       7337       0.73      0.556      0.613      0.456      0.733      0.559       0.61       0.42
            hard_coral        410       6143      0.712        0.6      0.644      0.489      0.714      0.601      0.644      0.461
            soft_coral        410       1194      0.747      0.513      0.583      0.423      0.753      0.518      0.577      0.379
Speed: 0.4ms preprocess, 26.4ms inference, 0.0ms loss, 1.5ms postprocess per image
```

##### Large Model + big imgsz + Long training time + large close_mosaic + data augmentation

###### Take Away

###### Parameters

- epochs: `200`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `40`
- flipud: `0.5`
- degrees: `45`
- translate: `0.2`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train15`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=200 imgsz=1024 close_mosaic=40 flipud=0.5 degrees=45 translate=0.2
```

###### Performance

```sh
Validating runs/segment/train15/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        410       7337      0.686      0.613      0.647      0.489      0.688      0.618      0.645       0.45
            hard_coral        410       6143      0.672      0.652      0.682      0.526      0.675      0.658      0.683      0.495
            soft_coral        410       1194      0.699      0.575      0.613      0.452        0.7      0.578      0.607      0.405
Speed: 0.4ms preprocess, 26.4ms inference, 0.0ms loss, 1.6ms postprocess per image
```

##### Large Model + big imgsz + Very Long training time + large close_mosaic

###### Take Away

Stopped training after 153 epochs because no improvement in metrics.

###### Parameters

- epochs: `1000`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `150`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train17`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=1000 imgsz=1024 close_mosaic=150
```

###### Performance

```sh
Validating runs/segment/train17/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        410       7337      0.708      0.566      0.607      0.456      0.717      0.567      0.604      0.419
            hard_coral        410       6143      0.712      0.603      0.637      0.488      0.718      0.603      0.639       0.46
            soft_coral        410       1194      0.705      0.529      0.578      0.423      0.716       0.53      0.568      0.379
```

#### Region specific - SEAFLOWER_BOLIVAR

##### Baseline

###### Parameters

- epochs: `20`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train8`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8m-seg.pt epochs=20
```

###### Performance

```sh
Validating runs/segment/train8/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P
                   all         50        760      0.572      0.419      0.415      0.287      0.548      0.402      0.401      0.249
            hard_coral         50        708      0.571      0.492       0.49      0.355      0.581      0.497      0.489      0.333
            soft_coral         50         52      0.573      0.346       0.34      0.219      0.514      0.308      0.314      0.164
Speed: 0.1ms preprocess, 3.9ms inference, 0.0ms loss, 1.2ms postprocess per image
```

##### Longer training time

###### Take Away

It seems that we can train for even longer and the model should keep improving.

###### Parameters

- epochs: `80`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train9`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8m-seg.pt epochs=80
```

###### Performance

```sh
Validating runs/segment/train9/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all         50        760      0.765      0.412      0.466      0.331      0.746      0.408      0.458      0.293
            hard_coral         50        708      0.681      0.459      0.493      0.368      0.693      0.469      0.502      0.337
            soft_coral         50         52       0.85      0.365      0.439      0.295      0.799      0.346      0.413      0.248
Speed: 0.1ms preprocess, 3.9ms inference, 0.0ms loss, 0.7ms postprocess per image
```

##### Longer Longer training time

###### Take Away

Not performing better than the `80` epochs model. Performing worse actually.

###### Parameters

- epochs: `120`
- model size: `m`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train10`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8m-seg.pt epochs=120
```

###### Performance

```sh
Validating runs/segment/train10/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P
                   all         50        760       0.69       0.44      0.456      0.339      0.677      0.434      0.452      0.292
            hard_coral         50        708      0.685      0.494      0.495      0.377      0.695      0.503      0.506      0.345
            soft_coral         50         52      0.696      0.385      0.418      0.301      0.659      0.365      0.398      0.239
Speed: 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.7ms postprocess per image
```


##### Large model

###### Take Away

Overall better performances than the baseline model.

###### Parameters

- epochs: `20`
- model size: `x`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train11`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8x-seg.pt epochs=20
```

###### Performance

```sh
Validating runs/segment/train11/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P
                   all         50        760      0.592      0.414      0.413      0.267      0.605      0.414      0.407      0.246
            hard_coral         50        708      0.661      0.501      0.519      0.383      0.672      0.501       0.52      0.345
            soft_coral         50         52      0.523      0.327      0.306      0.151      0.537      0.327      0.295      0.147
Speed: 0.1ms preprocess, 9.1ms inference, 0.0ms loss, 1.1ms postprocess per image
```

##### Large Model + big imgsz

###### Take Away

Overall much better performance than the baseline. Took some time before
learning and not jumping around. Is the learning rate too high at the beginning
of the training?

###### Parameters

- epochs: `20`
- model size: `x`
- imgsz: `1024`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train12`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8x-seg.pt epochs=20 imgsz=1024
```

###### Performance

```sh
Validating runs/segment/train12/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all         50        760      0.557      0.474      0.442      0.324      0.562      0.478      0.441      0.278
            hard_coral         50        708      0.592      0.524      0.511      0.382      0.602      0.532      0.517      0.366
            soft_coral         50         52      0.522      0.423      0.372      0.266      0.522      0.423      0.366      0.191
Speed: 0.3ms preprocess, 22.7ms inference, 0.0ms loss, 0.9ms postprocess per image
```

##### Large Model + big imgsz + Long training time

###### Take Away

###### Parameters

- epochs: `120`
- model size: `x`
- imgsz: `1024`
- data: handled label mismatch
- all other parameters are left as default
- train run: `segment/train16`
- dataset: `benthic_datasets`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets/SEAFLOWER_BOLIVAR/data.yaml model=yolov8x-seg.pt epochs=120 imgsz=1024
```

###### Performance

```sh
Validating runs/segment/train16/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all         50        760      0.681       0.51      0.506      0.375      0.676      0.513      0.494      0.333
            hard_coral         50        708      0.605       0.52      0.473      0.364      0.603      0.527      0.476      0.346
            soft_coral         50         52      0.758        0.5      0.539      0.387       0.75        0.5      0.512       0.32
Speed: 0.3ms preprocess, 22.9ms inference, 0.0ms loss, 0.7ms postprocess per image
```

##### Baseline + longer training time

###### Take Away


Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 78, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

###### Parameters

- epochs: `200`
- model size: `m`
- all other parameters are left as default
- train run: `segment/train24`
- dataset: `benthic_datasets2`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8m-seg.pt epochs=200
```

###### Performance

```sh
Validating runs/segment/train24/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        349       7359      0.679       0.55      0.593      0.421      0.689      0.555      0.597       0.38
            hard_coral        349       5921      0.678      0.574      0.614      0.446      0.683      0.574      0.613      0.396
            soft_coral        349       1438      0.679      0.526      0.571      0.396      0.695      0.535       0.58      0.363
Speed: 0.1ms preprocess, 4.6ms inference, 0.0ms loss, 1.2ms postprocess per image
```

##### Large Model + long training time + large imgsz

###### Take Away

Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 89, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

###### Parameters

- epochs: `200`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `40`
- all other parameters are left as default
- train run: `segment/train25`
- dataset: `benthic_datasets2`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=200 imgsz=1024 close_mosaic=40
```

###### Performance

```sh
Validating runs/segment/train25/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:16<00:00,  1.50s/it]
                   all        349       7359      0.712      0.594      0.632       0.47      0.715      0.601      0.636      0.439
            hard_coral        349       5921      0.698      0.607      0.646      0.489      0.701      0.613      0.649      0.458
            soft_coral        349       1438      0.726      0.581      0.618      0.451      0.729      0.588      0.623      0.421
Speed: 0.4ms preprocess, 26.0ms inference, 0.0ms loss, 2.0ms postprocess per image
```

##### Large Model + long training time + large imgsz + data augmentation

###### Take Away

Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 107, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

###### Parameters

- epochs: `200`
- model size: `x`
- imgsz: `1024`
- data augmentation
  - degrees: `180`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- close_mosaic: `40`
- dataset: `benthic_datasets2`
- train run: `segment/train26`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=200 imgsz=1024 close_mosaic=40 flipud=0.5 degrees=45 translate=0.2
```

###### Performance

```sh
Validating runs/segment/train26/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:16<00:00,  1.53s/it]
                   all        349       7359      0.721      0.598      0.659      0.502      0.727      0.604      0.662      0.472
            hard_coral        349       5921      0.716      0.608      0.673      0.522      0.723      0.614      0.678      0.491
            soft_coral        349       1438      0.726      0.588      0.646      0.482      0.732      0.594      0.647      0.452
Speed: 0.4ms preprocess, 26.0ms inference, 0.0ms loss, 2.1ms postprocess per image
```

##### Large Model + large imgsz

###### Take Away

###### Parameters

- epochs: `120`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `35`
- all other parameters are left as default
- train run: `segment/train27`
- dataset: `benthic_datasets2`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=120 imgsz=1024 close_mosaic=35
```

###### Performance

```sh
Validating runs/segment/train27/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:16<00:00,  1.52s/it]
                   all        349       7359       0.72      0.568      0.634      0.467      0.721       0.58      0.639      0.435
            hard_coral        349       5921      0.699      0.597      0.652      0.492      0.692      0.605      0.652       0.46
            soft_coral        349       1438      0.742      0.539      0.616      0.441       0.75      0.555      0.626       0.41
Speed: 0.9ms preprocess, 26.0ms inference, 0.0ms loss, 1.5ms postprocess per image
```

##### Large Model + long training time + large imgsz + data augmentation

###### Take Away

Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 79, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

###### Parameters

- epochs: `140`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `35`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/train28`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=140 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=35
```

###### Performance

```sh
Validating runs/segment/train28/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:16<00:00,  1.51s/it]
                   all        349       7359      0.734      0.612      0.672       0.51      0.731      0.626      0.675      0.477
            hard_coral        349       5921      0.721      0.617      0.684      0.531      0.716       0.63      0.686      0.499
            soft_coral        349       1438      0.747      0.608      0.661       0.49      0.747      0.621      0.664      0.455
```

##### Large Model + long training time + large imgsz + data augmentation #2

###### Take Away

###### Parameters

- epochs: `101`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `21`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/train29`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=101 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=21
```

###### Performance


```sh
Validating runs/segment/train29/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:16<00:00,  1.54s/it]
                   all        349       7359      0.711       0.62      0.672      0.508      0.722      0.625      0.675      0.479
            hard_coral        349       5921      0.702      0.628      0.686      0.531      0.711      0.631      0.687      0.501
            soft_coral        349       1438      0.721      0.612      0.659      0.485      0.732      0.618      0.662      0.457
Speed: 0.4ms preprocess, 26.2ms inference, 0.0ms loss, 1.8ms postprocess per image
```

##### Large Model + long training time + large imgsz + data augmentation #2

###### Take Away


###### Parameters

- epochs: `116`
- model size: `x`
- imgsz: `1024`
- close_mosaic: `35`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/trainXXX`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8x-seg.pt epochs=116 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=35
```

###### Performance

```sh
Validating runs/segment/train30/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (Quadro RTX 8000, 48593MiB)
YOLOv8x-seg summary (fused): 295 layers, 71722582 parameters, 0 gradients, 343.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:17<00:00,  1.57s/it]
                   all        349       7359      0.727       0.61      0.668      0.505      0.717      0.626      0.668      0.475
            hard_coral        349       5921      0.709      0.617      0.681      0.528        0.7      0.632      0.682      0.498
            soft_coral        349       1438      0.744      0.604      0.655      0.482      0.734      0.619      0.654      0.452
Speed: 0.4ms preprocess, 26.0ms inference, 0.0ms loss, 2.6ms postprocess per image
```


##### Medium Model + long training time + large imgsz + data augmentation

###### Take Away

###### Parameters

- epochs: `100`
- model size: `m`
- imgsz: `1024`
- close_mosaic: `35`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/train31`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8m-seg.pt epochs=100 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=35 device=1
```

###### Performance

```sh
Validating runs/segment/train31/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:1 (Quadro RTX 8000, 48586MiB)
YOLOv8m-seg summary (fused): 245 layers, 27223542 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        349       7359      0.709        0.6       0.65      0.487      0.714      0.605      0.655       0.46
            hard_coral        349       5921      0.716      0.606       0.67      0.513      0.719      0.609      0.672      0.484
            soft_coral        349       1438      0.703      0.593      0.629      0.461      0.709        0.6      0.638      0.436
Speed: 0.3ms preprocess, 10.8ms inference, 0.0ms loss, 2.6ms postprocess per image
```

##### Small Model + long training time + large imgsz + data augmentation

###### Take Away

###### Parameters

- epochs: `100`
- model size: `s`
- imgsz: `1024`
- close_mosaic: `35`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/train32`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8s-seg.pt epochs=100 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=35 device=1
```

###### Performance

```sh
Validating runs/segment/train32/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:1 (Quadro RTX 8000, 48586MiB)
YOLOv8s-seg summary (fused): 195 layers, 11780374 parameters, 0 gradients, 42.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        349       7359      0.702        0.6      0.638      0.471      0.706      0.605      0.641      0.445
            hard_coral        349       5921      0.695      0.612      0.653      0.495      0.701      0.618      0.658      0.471
            soft_coral        349       1438      0.709      0.589      0.623      0.447      0.711      0.591      0.624      0.419
Speed: 0.6ms preprocess, 5.2ms inference, 0.0ms loss, 2.2ms postprocess per image
```

##### Nano Model + long training time + large imgsz + data augmentation

###### Take Away

###### Parameters

- epochs: `100`
- model size: `n`
- imgsz: `1024`
- close_mosaic: `35`
- data augmentation
  - degrees: `45`
  - flipud: `0.5`
  - translate: `0.2`
- all other parameters are left as default
- dataset: `benthic_datasets2`
- train run: `segment/train33`

###### Training command

```sh
yolo train data=/data/ai-for-coral-reefs-2/yolov8_ready/benthic_datasets2/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=1024 flipud=0.5 degrees=45 translate=0.2 close_mosaic=35 device=1
```

###### Performance

```sh
Validating runs/segment/train33/weights/best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.9.18 torch-2.1.1+cu121 CUDA:1 (Quadro RTX 8000, 48586MiB)
YOLOv8n-seg summary (fused): 195 layers, 3258454 parameters, 0 gradients, 12.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|████████
                   all        349       7359      0.681      0.581      0.632      0.462      0.692       0.58       0.63      0.435
            hard_coral        349       5921      0.667      0.594      0.641      0.482       0.68      0.597      0.646      0.458
            soft_coral        349       1438      0.695      0.567      0.623      0.442      0.704      0.564      0.614      0.411
Speed: 0.4ms preprocess, 2.8ms inference, 0.0ms loss, 2.2ms postprocess per image
```
