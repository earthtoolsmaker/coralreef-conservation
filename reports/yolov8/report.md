# Final Report

Group: Supervised team - YOLOv8 instance segmentation

Abstrat: TODO

## Intro

Marine biologists studying coral reefs spend a vast chunk of their time
manually processing the data they recover from research dives. The goal of this
challenge is to develop an image segmentation pipeline to speed up the analysis
of this data. This will help conservationists and researchers better protect
and understand these crucial ocean ecosystems.
By using computer vision to segment coral reefs in benthic imagery, we can
measure long-term growth or loss of coral cover in marine protected areas.

## The challenge

The goal of this subgroup is to automate the image segmentation of underwater
imagery.
The provided dataset contains images of a wide range of static benthic
organisms such as coral, algae and bare substrate that are commonly found on
tropical coral reefs. The final model should accurately differentiate between
hard and soft corals, and eventually among a group of several biologically
functional entities.

One key usecase of the model is to estimate benthich coverage of coral groups.

## Data

The data provided by ReefSupport is made available on a publically hosted [Google
Cloud bucket](https://console.cloud.google.com/storage/browser/rs_storage_open).
They provide two different types of datasets:

- __Point labels or sparse labels__: random points in an image are classified. A
typical image would contain between 50 and 100 point labels.
- __Mask labels or dense labels__: detailed segmentations masks are provided for
hard and soft corals.

Given the nature of the computer vision task, dense labels are required.
ReefSupport provides a high quality dataset dense labels dataset using images
from the `SEAVIEW`, `TETES` and `SEAFLOWER` datasets.

| Dataset   | Region      | Number of dense labels |
| --------- | ----------- | ---------------------- |
| SEAFLOWER | BOLIVAR     | 246                    |
| SEAFLOWER | COURTOWN    | 241                    |
| SEAVIEW   | ATL         | 705                    |
| SEAVIEW   | IDN_PHL     | 466                    |
| SEAVIEW   | PAC_AUS     | 808                    |
| SEAVIEW   | PAC_USA     | 728                    |
| TETES     | PROVIDENCIA | 105                    |

### Low dense labels quality for `SEAVIEW_PAC_USA`

The dense labels in `SEAVIEW/PAC_USA` are causing issues with the data
modelling - we had to exclude them from the training set. The labeler for that
dataset unfortunately created large masks for almost all corals present in the
image instead of individual mask per entity.

### Image size and quality

The images from `SEAFLOWER` and `TETES` are much larger (5-10x more pixels)
than the ones from `SEAVIEW`. We did not know if this would be an issue for the
data modelling stage.

### Empty masks

Some empty individual masks (all black pixels) are discovered throughout the
datasets - filtering these out yielded greater performance.
For instance, 532 empty masks in `SEAVIEW_PAC_USA` and 328 empty masks in
`SEAVIEW_ATL`.

### Mismatched sparse and dense labels

An analysis was run by the unsupervised group to compare the sparse and dense
labels.
They compared the dense labels provided by ReefSupport with the available point
labels for these same images.
They identified 1054 dense labels that contained more than 10 mismatched label
points that were excluded from the training and evaluation sets. This
translates to 10% sparse labels error with 100 points and 20% sparse labels
error with 50 points.

It was decided to exclude these dense labels from the training and evaluation
sets.

## Data Preprocessing

In order to levarage the YOLOv8 ecosytem, one needs to preprocess the provided raw
datasets into a format that can be understood.

The training script expects a specific folder structure as seen below:

```sh
.
data.yaml
├── train
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
```

The `data.yaml` file looks like the following:

```yaml
train: ./train/images
val: ./val/images
nc: 2
names:
  - hard_coral
  - soft_coral
```

Each label file in `train/labels` and `val/labels` is a filename with the same
name as its associated image.

Each line represents an instance of a class with a defined contour. It has the
following format:

```txt
class_number x1 y1 x2 y2 x3 y3 ... xk yk
class_number x1 y1 x2 y2 x3 y3 ... xj yj
```

Where the coordinates `x` and `y` are normalized to the image width and height
accordingly. Therefore, they always lie in the range `[0,1]`.

Example:

```txt
1 0.617 0.359 0.114 0.173 0.322 0.654
0 0.094 0.386 0.156 0.236 0.875 0.134
```

## Data Modelling

### Baseline

A __baseline__ model was quickly established in order to measure the
effectiveness of this approach, while evaluating the potential performance
gains that one could still make.

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | m                       |
| data                | All regions but PAC_USA |
| epochs              | 5                       |
| imgsz               | 640                     |
| close_mosaic        | 10                      |
| degrees             | 0                       |
| flipud              | 0                       |
| translate           | 0.1                     |

### Best models

In this section we present the best models we were able to finetune. It took
hundreds of hours of GPU time, training on the provided FruitpunchAI GPU server
to find decent hyperparameter combinations.
As we do not know the hardware or economic constraints of ReefSupport,
we decided to provide a wide range of possible models that could fit
different configurations: from embedding on edge to do real time video
stream segmentation, to high end GPU with higher IoU performance, and
everything in between.

Below is a summary of the different model size performance on the COCO-SEG
dataset provided by Ultralytics:

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

#### xlarge

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | x                       |
| data                | All regions but PAC_USA |
| epochs              | 140                     |
| imgsz               | 1024                    |
| close_mosaic        | 35                      |
| degrees             | 45                      |
| flipud              | 0.5                     |
| translate           | 0.2                     |

#### large

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | l                       |
| data                | All regions but PAC_USA |
| epochs              | 120                     |
| imgsz               | 1024                    |
| close_mosaic        | 35                      |
| degrees             | 45                      |
| flipud              | 0.5                     |
| translate           | 0.2                     |

#### medium

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | m                       |
| data                | All regions but PAC_USA |
| epochs              | 100                     |
| imgsz               | 1024                    |
| close_mosaic        | 35                      |
| degrees             | 45                      |
| flipud              | 0.5                     |
| translate           | 0.2                     |

#### small

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | s                       |
| data                | All regions but PAC_USA |
| epochs              | 100                     |
| imgsz               | 1024                    |
| close_mosaic        | 35                      |
| degrees             | 45                      |
| flipud              | 0.5                     |
| translate           | 0.2                     |

#### nano

| Hyperparameter Name | Hyperparameter Value    |
| ------------------- | ----------------------- |
| Model Size          | n                       |
| data                | All regions but PAC_USA |
| epochs              | 100                     |
| imgsz               | 1024                    |
| close_mosaic        | 35                      |
| degrees             | 45                      |
| flipud              | 0.5                     |
| translate           | 0.2                     |

## Conclusions

## Bibliography
