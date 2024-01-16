# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # YOLOv8 Inference Notebook
#
# In this notebook, we look at how good the different fine tuned YOLOv8 models perform quantitatively and qualitatively.

# %% [markdown]
# ## Imports

# %%
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import ultralytics
import yaml
from PIL import Image
from tqdm.notebook import tqdm, trange
from ultralytics import YOLO

# %% [markdown]
# ## Global Variables

# %%
FRUITPUNCH_CHALLENGE_PROJECT_ROOT = Path(
    "/home/chouffe/fruitpunch/challenges/coralreefs2/"
)

YOLOV8_MODELS_ROOT_PATH = FRUITPUNCH_CHALLENGE_PROJECT_ROOT / "models/yolov8/"
CACHE_PATH = FRUITPUNCH_CHALLENGE_PROJECT_ROOT / ".cache"

CLASS_TO_COLOR_MAPPING = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 0, 0),
}


# TODO: get this from data.yaml
LABEL_TO_CLASS_MAPPING: dict[str, int] = {"soft_coral": 1, "hard_coral": 0, "other": 2}
CLASS_TO_LABEL_MAPPING: dict[int, str] = {
    v: k for k, v in LABEL_TO_CLASS_MAPPING.items()
}


Polygon = np.ndarray
AnnotationEntry = dict
Annotation = list[AnnotationEntry]
Prediction = list[dict]
# %% [markdown]
# ## Utilities and helpers


# %%
def yaml_content(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def image_filename_to_image_filepath(
    train_data_root_path: Path, filename: str, split: str = "val"
) -> Path:
    return train_data_root_path / split / "images" / filename


def label_filename_to_label_filepath(
    train_data_root_path: Path, filename: str, split: str = "val"
) -> Path:
    return train_data_root_path / split / "labels" / filename


def slurp(filepath: Path) -> str:
    with open(filepath, "r") as f:
        return f.read()


def load_image_filepath(image_filepath: Path) -> np.ndarray:
    return cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)


def get_prediction_as_image(prediction) -> PIL.Image:
    im_array = prediction.plot(labels=False)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    return im


def denormalize_polygon(polygon: Polygon, W: int, H: int) -> Polygon:
    """`polygon`: numpy array of shape (_,2) containing the polygon x,y
    coordinates.

    `W`: int - width of the image / mask
    `H`: int - height of the image / mask

    returns a numpy array of shape `polygon.shape` with coordinates that are
    denormalized and between 0 and W or H.
    """
    copy = np.copy(polygon)
    copy = copy.astype(np.float16)
    copy[:, 0] *= W
    copy[:, 1] *= H
    copy = copy.astype(np.int32)

    return copy


def partition(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def draw_polygon(img: np.ndarray, polygon: Polygon, color) -> np.ndarray:
    return cv2.fillPoly(img, pts=[polygon], color=color)


def draw_bounding_box(img: np.ndarray, bbox, color, thickness: int = 3) -> np.ndarray:
    start_point, end_point = bbox
    return cv2.rectangle(img, start_point, end_point, color, thickness)


def bounding_box(points) -> tuple[tuple[float, float], tuple[float, float]]:
    x_coordinates, y_coordinates = zip(*points)
    return (
        (min(x_coordinates), min(y_coordinates)),
        (max(x_coordinates), max(y_coordinates)),
    )


def parse_annotation_line(
    annotation_line: str, orig_img: np.ndarray
) -> AnnotationEntry:
    H, W, _ = orig_img.shape
    elements = annotation_line.split(" ")
    object_class = int(elements[0])
    polygon_normalized = np.array(list(partition([float(e) for e in elements[1:]], 2)))
    polygon = denormalize_polygon(polygon_normalized, W=W, H=H)
    return {
        "class": object_class,
        "polygon_normalized": polygon_normalized,
        "polygon": polygon,
    }


def parse_annotation_raw(annotation_raw: str, orig_img: np.ndarray) -> Annotation:
    if annotation_raw == "":
        return []
    else:
        try:
            annotation_lines = annotation_raw.split("\n")
            result = [
                parse_annotation_line(line, orig_img)
                for line in annotation_lines
                if line.strip()
            ]
            return result
        except:
            print(f"Parsing error with annotation_raw: {annotation_raw}")
            print(f"Returning empty annotation")
            return []


def generate_annotation_as_nd_array(
    annotation: Annotation,
    orig_img: np.ndarray,
    class_to_color_mapping: dict = CLASS_TO_COLOR_MAPPING,
    annotation_overlay_segment_alpha: float = 0.45,
    thickness: int = 2,
) -> np.ndarray:
    img = orig_img.copy()
    overlay_segment = orig_img.copy()
    for annotation_entry in annotation:
        color = class_to_color_mapping.get(annotation_entry["class"])
        overlay_segment = draw_polygon(
            overlay_segment, annotation_entry["polygon"], color
        )
        bbox = bounding_box(annotation_entry["polygon"])
        img = draw_bounding_box(img, bbox, color, thickness=thickness)
    img = cv2.addWeighted(
        overlay_segment,
        annotation_overlay_segment_alpha,
        img,
        1 - annotation_overlay_segment_alpha,
        0,
        img,
    )
    return img


def generate_annotation_raw_as_nd_array(
    annotation_raw: str,
    orig_img: np.ndarray,
    class_to_color_mapping: dict = CLASS_TO_COLOR_MAPPING,
) -> np.ndarray:
    annotation = parse_annotation_raw(annotation_raw, orig_img)
    return generate_annotation_as_nd_array(
        annotation=annotation,
        orig_img=orig_img,
        class_to_color_mapping=class_to_color_mapping,
    )


def generate_prediction_results_as_nd_array(
    prediction: Prediction,
    orig_img: np.ndarray,
    class_to_color_mapping: dict = CLASS_TO_COLOR_MAPPING,
    annotation_overlay_segment_alpha: float = 0.45,
    thickness: int = 2,
) -> np.ndarray:
    img = orig_img.copy()
    overlay_segment = orig_img.copy()
    for prediction_entry in prediction:
        color = class_to_color_mapping.get(prediction_entry["class"])
        overlay_segment = draw_polygon(
            overlay_segment, prediction_entry["polygon"], color
        )
        bbox = bounding_box(prediction_entry["polygon"])
        img = draw_bounding_box(img, bbox, color, thickness=thickness)
    img = cv2.addWeighted(
        overlay_segment,
        annotation_overlay_segment_alpha,
        img,
        1 - annotation_overlay_segment_alpha,
        0,
        img,
    )
    return img


def get_prediction_data(results_raw: ultralytics.engine.results.Results) -> Prediction:
    r = []
    for i, cls in enumerate(results_raw.boxes.cls):
        # FIXME:
        H, W = results_raw.masks.orig_shape
        # W, H = results_raw.masks.orig_shape
        polygon_normalized = results_raw.masks.xyn[i]
        polygon = denormalize_polygon(polygon=polygon_normalized, W=W, H=H)
        conf = float(results_raw.boxes.conf[i])
        entry = {
            "class": int(cls),
            "conf": conf,
            "polygon_normalized": polygon_normalized,
            "polygon": polygon,
        }
        r.append(entry)
    return r


def img_array_to_pil_img(img_array: np.ndarray) -> PIL.Image.Image:
    """Converts an `img_array` into a PIL image ready for displaying it."""
    return Image.fromarray(img_array[..., ::-1])


def display_annotation_prediction(
    orig_img: np.ndarray,
    annotation: Annotation,
    prediction: Prediction,
    save_filepath: Optional[Path],
):
    """Display the annotation and prediction results side by side."""
    img_annotation = img_array_to_pil_img(
        generate_annotation_as_nd_array(annotation, orig_img)
    )
    img_prediction = img_array_to_pil_img(
        generate_prediction_results_as_nd_array(prediction, orig_img)
    )

    _f, _axs = plt.subplots(1, 2, figsize=(20, 20))

    plt.subplot(1, 2, 1)
    plt.imshow(img_annotation)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_prediction)
    plt.title("Prediction")
    plt.axis("off")

    if save_filepath:
        print(f"saving mask comparison in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()

    plt.close()


def polygon_to_boolean_mask(W: int, H: int, polygon: Polygon) -> np.ndarray:
    """Turn a polygon into a boolean mask.

    `W`: width of the image
    `H`: height of the image
    `polygon`: list of x-y coordinates (floats)

    Note:
    It assumes that the polygon can be closed and will fill the inside.
    """
    shape = (W, H, 3)
    img = np.zeros(shape, dtype=np.int32)
    color = (255, 255, 255)  # white
    image_mask = draw_polygon(img, polygon, color=color)
    mask = np.all(image_mask == list(color), axis=2)
    return mask


def polygon_to_boolean_mask2(W: int, H: int, polygon: Polygon) -> np.ndarray:
    """Turn a polygon into a boolean mask.

    `W`: width of the image
    `H`: height of the image
    `polygon`: list of x-y coordinates (floats)

    Note:
    It assumes that the polygon can be closed and will fill the inside.
    """
    shape = (H, W, 3)
    img = np.zeros(shape, dtype=np.int32)
    color = (255, 255, 255)  # white
    image_mask = draw_polygon(img, polygon, color=color)
    mask = np.all(image_mask == list(color), axis=2)
    return mask


def iou(bmask1, bmask2):
    """Returns the iou metric (float between 0 and 1) of the Intersection over
    Union of bmask1 and bmask2."""
    intersection_mask = bmask1 & bmask2
    union_mask = bmask1 | bmask2
    N = np.sum(intersection_mask)
    D = np.sum(union_mask)
    epsilon = 1
    return N / (D + epsilon)  # Avoid divide by zero


def dice(bmask1, bmask2):
    """Returns the dice coefficient (float between 0 and 1)  of bmask1 and
    bmask2."""
    intersection_mask = bmask1 & bmask2
    N = 2 * np.sum(intersection_mask)
    D = np.sum(bmask1) + np.sum(bmask2)
    epsilon = 1
    return N / (D + epsilon)  # Avoid divide by zero


# FIXME: remove 2 suffix
# FIXME: invert W and H in the function body here
def generate_class_boolean_masks2(
    annotation: Annotation,
    prediction: Prediction,
    W: int,
    H: int,
):
    """
    Returns a dict with the following shape:
    {
      'other': {'annotation': bmask, 'prediction': bmask},
      'soft_coral': {'annotation': bmask, 'prediction': bmask},
      'hard_coral': {'annotation': bmask, 'prediction': bmask},
    }
    """
    classes_prediction = [e["class"] for e in prediction]
    classes_annotation = [e["class"] for e in annotation]
    soft_coral_polygons_annotation = [
        annotation[i]["polygon"] for i, cls in enumerate(classes_annotation) if cls == 1
    ]
    soft_coral_polygons_prediction = [
        prediction[i]["polygon"] for i, cls in enumerate(classes_prediction) if cls == 1
    ]
    hard_coral_polygons_annotation = [
        annotation[i]["polygon"] for i, cls in enumerate(classes_annotation) if cls == 0
    ]
    hard_coral_polygons_prediction = [
        prediction[i]["polygon"] for i, cls in enumerate(classes_prediction) if cls == 0
    ]
    soft_coral_masks_annotation = [
        polygon_to_boolean_mask(H, W, polygon)
        for polygon in soft_coral_polygons_annotation
    ]
    soft_coral_masks_prediction = [
        polygon_to_boolean_mask(H, W, polygon)
        for polygon in soft_coral_polygons_prediction
    ]
    hard_coral_masks_annotation = [
        polygon_to_boolean_mask(H, W, polygon)
        for polygon in hard_coral_polygons_annotation
    ]
    hard_coral_masks_prediction = [
        polygon_to_boolean_mask(H, W, polygon)
        for polygon in hard_coral_polygons_prediction
    ]

    bmask_soft_coral_annotation = reduce(
        lambda m1, m2: m1 | m2,
        soft_coral_masks_annotation,
        np.zeros((H, W), dtype=bool),
    )
    bmask_soft_coral_prediction = reduce(
        lambda m1, m2: m1 | m2,
        soft_coral_masks_prediction,
        np.zeros((H, W), dtype=bool),
    )
    bmask_hard_coral_annotation = reduce(
        lambda m1, m2: m1 | m2,
        hard_coral_masks_annotation,
        np.zeros((H, W), dtype=bool),
    )
    bmask_hard_coral_prediction = reduce(
        lambda m1, m2: m1 | m2,
        hard_coral_masks_prediction,
        np.zeros((H, W), dtype=bool),
    )
    bmask_other_annotation = ~(
        bmask_soft_coral_annotation | bmask_hard_coral_annotation
    )
    bmask_other_prediction = ~(
        bmask_soft_coral_prediction | bmask_hard_coral_prediction
    )

    return {
        "soft_coral": {
            "annotation": bmask_soft_coral_annotation,
            "prediction": bmask_soft_coral_prediction,
        },
        "hard_coral": {
            "annotation": bmask_hard_coral_annotation,
            "prediction": bmask_hard_coral_prediction,
        },
        "other": {
            "annotation": bmask_other_annotation,
            "prediction": bmask_other_prediction,
        },
    }


def generate_mask_overlay(
    orig_img: np.ndarray, bmask: np.ndarray, alpha: float = 0.3
) -> np.ndarray:
    """Generates an overlay on top of `orig_img` (make a copy) to better
    highligh the boolean mask `bmask`"""
    copy = orig_img.copy()
    img = np.stack([bmask, bmask, bmask], axis=2) * copy
    overlay = orig_img.copy()
    return cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img,
    )


def display_mask_comparison(
    orig_img: np.ndarray,
    boolean_mask1: np.ndarray,
    boolean_mask2: np.ndarray,
    class_mask: str = "soft coral",
    save_filepath: Optional[Path] = None,
) -> None:
    """Displays as images the comparison between the boolean_masks.

    `boolean_mask1` is
    the ground truth whereas `boolean_mask2` is the prediction.
    """
    f, axs = plt.subplots(1, 4, figsize=(20, 20))

    plt.subplot(1, 4, 1)
    plt.imshow(boolean_mask1)
    plt.title(f"Ground Truth for {class_mask} ")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(generate_mask_overlay(orig_img, boolean_mask1))
    plt.title(f"Ground Truth for {class_mask} ")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(boolean_mask2)
    plt.title(f"Prediction for {class_mask}")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(generate_mask_overlay(orig_img, boolean_mask2))
    plt.title(f"Prediction for {class_mask} ")
    plt.axis("off")

    if save_filepath:
        print(f"saving mask comparison in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()
    plt.close()


def confusion_matrix_to_precision_recall_f1(
    confusion_matrix: np.ndarray,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
) -> dict:
    num_classes = confusion_matrix.shape[0]
    assert num_classes == 3, "should be soft corals, hard corals and others."

    metrics = ["precision", "recall", "f1"]
    result = {m: {} for m in metrics}

    for c in range(0, num_classes):
        tp = confusion_matrix[c, c]
        fp = np.sum([confusion_matrix[i, c] for i in set(range(0, 3)).difference({c})])
        fn = np.sum([confusion_matrix[c, i] for i in set(range(0, 3)).difference({c})])
        epsilon = 0.0001
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        result["precision"][class_to_label_mapping.get(c)] = precision
        result["recall"][class_to_label_mapping.get(c)] = recall
        result["f1"][class_to_label_mapping.get(c)] = f1

    # Computing the means
    for m in metrics:
        result[m]["mean"] = (
            np.sum(
                [result[m][class_to_label_mapping.get(i)] for i in range(num_classes)]
            )
            / num_classes
        )

    return result


def get_evaluation_metrics(
    annotation: Annotation,
    prediction: Prediction,
    W: int,
    H: int,
    confusion_matrix: np.ndarray,
) -> dict:
    class_boolean_masks = generate_class_boolean_masks2(
        annotation=annotation,
        prediction=prediction,
        W=W,
        H=H,
    )

    dice_soft = dice(
        class_boolean_masks["soft_coral"]["annotation"],
        class_boolean_masks["soft_coral"]["prediction"],
    )
    dice_hard = dice(
        class_boolean_masks["hard_coral"]["annotation"],
        class_boolean_masks["hard_coral"]["prediction"],
    )
    dice_other = dice(
        class_boolean_masks["other"]["annotation"],
        class_boolean_masks["other"]["prediction"],
    )
    iou_soft = iou(
        class_boolean_masks["soft_coral"]["annotation"],
        class_boolean_masks["soft_coral"]["prediction"],
    )
    iou_hard = iou(
        class_boolean_masks["hard_coral"]["annotation"],
        class_boolean_masks["hard_coral"]["prediction"],
    )
    iou_other = iou(
        class_boolean_masks["other"]["annotation"],
        class_boolean_masks["other"]["prediction"],
    )
    freq_annotation = Counter([e["class"] for e in annotation])
    freq_prediction = Counter([e["class"] for e in prediction])

    result_instances = {
        "instances": {
            "prediction": {
                "soft_coral": freq_prediction[LABEL_TO_CLASS_MAPPING.get("soft_coral")],
                "hard_coral": freq_prediction[LABEL_TO_CLASS_MAPPING.get("hard_coral")],
            },
            "annotation": {
                "soft_coral": freq_annotation[LABEL_TO_CLASS_MAPPING.get("soft_coral")],
                "hard_coral": freq_annotation[LABEL_TO_CLASS_MAPPING.get("hard_coral")],
            },
        },
    }

    result_iou = {
        "iou": {
            "other": iou_other,
            "soft_coral": iou_soft,
            "hard_coral": iou_hard,
            "mean": (iou_soft + iou_hard + iou_other) / 3,
        },
    }

    result_dice = {
        "dice": {
            "other": dice_other,
            "soft_coral": dice_soft,
            "hard_coral": dice_hard,
            "mean": (dice_soft + dice_hard + dice_other) / 3,
        },
    }

    result_precision_recall_f1 = confusion_matrix_to_precision_recall_f1(
        confusion_matrix=confusion_matrix
    )

    result = {
        **result_instances,
        **result_iou,
        **result_dice,
        **result_precision_recall_f1,
    }

    return result


def get_matches_summary(
    annotation: Annotation,
    prediction: Prediction,
    orig_img: np.ndarray,
    iou_threshold: float = 0.25,
) -> dict:
    """
    Returns a dict with the following keys:
    - `pairwise_iou`: list of dicts - pairwise iou scores between all masks in `annotation` and `prediction`
    - `matches`: dict of `all` matches, the ones that correspond to the right class (`good`) and the `misclassified` ones.
    - `spurious_extra_prediction`: set of prediction indices that corresponds to predicted corals that are not in annotation.
    - `missed_annotation`: set of annotation indices that corresponds to the segments that were not predicted.
    """
    # FIXME: invert W and H here
    W, H, _ = orig_img.shape
    H, W, _ = orig_img.shape
    bmasks_annotation = [
        polygon_to_boolean_mask(W, H, e["polygon"]) for e in annotation
    ]
    bmasks_prediction = [
        polygon_to_boolean_mask(W, H, e["polygon"]) for e in prediction
    ]
    pairwise_iou = [
        {
            "idx_prediction": idx_prediction,
            "idx_annotation": idx_annotation,
            "class_prediction": prediction[idx_prediction]["class"],
            "class_annotation": annotation[idx_annotation]["class"],
            "iou": iou(bm_a, bm_p),
        }
        for idx_prediction, bm_p in enumerate(bmasks_prediction)
        for idx_annotation, bm_a in enumerate(bmasks_annotation)
    ]
    all_matches = [e for e in pairwise_iou if e["iou"] > iou_threshold]
    missclassified_matches = [
        e for e in all_matches if e["class_prediction"] != e["class_annotation"]
    ]
    good_matches = [
        e for e in all_matches if e["class_prediction"] == e["class_annotation"]
    ]
    spurious_extra_prediction = set(range(len(prediction))) - {
        e["idx_prediction"] for e in all_matches
    }
    missed_annotation = set(range(len(annotation))) - {
        e["idx_annotation"] for e in all_matches
    }
    return {
        "pairwise_iou": pairwise_iou,
        "matches": {
            "all": all_matches,
            "good": good_matches,
            "misclassified": missclassified_matches,
        },
        "spurious_extra_prediction": spurious_extra_prediction,
        "missed_annotation": missed_annotation,
    }


def display_evaluation_metrics(
    metrics: dict,
    save_filepath: Optional[Path] = None,
) -> None:
    if save_filepath:
        print(f"saving metrics in {save_filepath}")
        save_filepath.write_text(json.dumps(metrics, indent=2))
    else:
        print()
        print("Evaluation Metrics")
        print("------------------")
        if "instances" in metrics:
            print("Instances")
            print(f"    annotation")
            print(
                f"         soft coral: {metrics['instances']['annotation']['soft_coral']}"
            )
            print(
                f"         hard coral: {metrics['instances']['annotation']['hard_coral']}"
            )
            print(f"    prediction")
            print(
                f"         soft coral: {metrics['instances']['prediction']['soft_coral']}"
            )
            print(
                f"         hard coral: {metrics['instances']['prediction']['hard_coral']}"
            )

        print("Dice")
        print(f"    mean:       {metrics['dice']['mean']:.2}")
        print(f"    soft coral: {metrics['dice']['soft_coral']:.2f}")
        print(f"    hard coral: {metrics['dice']['hard_coral']:.2f}")
        print(f"    other:      {metrics['dice']['other']:.2f}")
        print("IoU")
        print(f"    mean:       {metrics['iou']['mean']:.2}")
        print(f"    soft coral: {metrics['iou']['soft_coral']:.2f}")
        print(f"    hard coral: {metrics['iou']['hard_coral']:.2f}")
        print(f"    other:      {metrics['iou']['other']:.2f}")


def display_matches_summary(
    matches_summary: dict,
    annotation: Annotation,
    prediction: Prediction,
    orig_img: np.ndarray,
    class_to_color_mapping: dict = CLASS_TO_COLOR_MAPPING,
    bbox_thickness: int = 4,
    save_filepath: Optional[Path] = None,
) -> None:
    H, W, _ = orig_img.shape
    masks_missed_annotation = [
        polygon_to_boolean_mask2(W=W, H=H, polygon=annotation[idx]["polygon"])
        for idx in matches_summary["missed_annotation"]
    ]
    mask_missed_annotation = reduce(
        lambda m1, m2: m1 | m2, masks_missed_annotation, np.zeros((H, W), dtype=bool)
    )
    masks_spurious_extra_prediction = [
        polygon_to_boolean_mask2(W=W, H=H, polygon=prediction[idx]["polygon"])
        for idx in matches_summary["spurious_extra_prediction"]
    ]
    mask_spurious_extra_prediction = reduce(
        lambda m1, m2: m1 | m2,
        masks_spurious_extra_prediction,
        np.zeros((H, W), dtype=bool),
    )

    img_missed_annotation = generate_mask_overlay(orig_img, mask_missed_annotation)
    for idx in matches_summary["missed_annotation"]:
        color = class_to_color_mapping.get(annotation[idx]["class"])
        bbox = bounding_box(annotation[idx]["polygon"])
        img_missed_annotation = draw_bounding_box(
            img_missed_annotation, bbox, color, thickness=bbox_thickness
        )

    img_spurious_extra_prediction = generate_mask_overlay(
        orig_img, mask_spurious_extra_prediction
    )
    for idx in matches_summary["spurious_extra_prediction"]:
        color = class_to_color_mapping.get(prediction[idx]["class"])
        bbox = bounding_box(prediction[idx]["polygon"])
        img_spurious_extra_prediction = draw_bounding_box(
            img_spurious_extra_prediction, bbox, color, thickness=bbox_thickness
        )

    f, axs = plt.subplots(1, 4, figsize=(20, 20))

    plt.subplot(1, 4, 1)
    plt.imshow(mask_missed_annotation)
    plt.title(f"Missed elements")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img_array_to_pil_img(img_missed_annotation))
    plt.title(f"Missed elements")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_spurious_extra_prediction)
    plt.title(f"Spurious predictions")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_array_to_pil_img(img_spurious_extra_prediction))
    plt.title(f"Spurious predictions")
    plt.axis("off")

    if save_filepath:
        print(f"saving matches summary in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()
    plt.close()


def pixel_location_to_class(
    x: int,
    y: int,
    class_boolean_masks: dict,
    mode: str = "annotation",
    label_to_class_mapping: dict[str, int] = LABEL_TO_CLASS_MAPPING,
) -> int:
    default_class = 0
    if class_boolean_masks["other"][mode][x, y]:
        return label_to_class_mapping.get("other", default_class)
    elif class_boolean_masks["soft_coral"][mode][x, y]:
        return label_to_class_mapping.get("soft_coral", default_class)
    elif class_boolean_masks["hard_coral"][mode][x, y]:
        return label_to_class_mapping.get("hard_coral", default_class)
    else:
        return label_to_class_mapping.get("other", default_class)


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    n_classes: int = 3,
) -> np.ndarray:
    cf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, p in zip(y_true, y_pred):
        cf_matrix[a][p] += 1
    return cf_matrix


def make_confusion_matrix_from_class_boolean_masks(class_boolean_masks) -> np.ndarray:
    W, H = class_boolean_masks["other"]["annotation"].shape
    y_true = [
        pixel_location_to_class(x, y, class_boolean_masks, mode="annotation")
        for x in range(W)
        for y in range(H)
    ]
    y_pred = [
        pixel_location_to_class(x, y, class_boolean_masks, mode="prediction")
        for x in range(W)
        for y in range(H)
    ]
    assert len(y_true) == len(y_pred), "y_true and y_pred should have same length"
    cf_matrix = confusion_matrix(y_true, y_pred)
    return cf_matrix


def plot_confusion_matrix(
    cf_matrix: np.ndarray,
    normalize=False,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
    save_path: Optional[Path] = None,
) -> None:
    """Given a confusion matrix `cf_matrix`, it displays it as a pyplot
    graph."""
    cm = cf_matrix.copy()
    ax = None
    if normalize:
        epsilon = 0.0001
        cm = cm.astype(np.float64) / (cm.sum(axis=1)[:, np.newaxis] + epsilon)
        ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues")
    else:
        ax = sns.heatmap(cm, annot=True, cmap="Blues")

    #  ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ## Ticket labels - List must be in alphabetical order
    labels = sorted([class_to_label_mapping[i] for i in range(3)])
    labels = [class_to_label_mapping[i] for i in range(3)]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save the confusion matrix
    if save_path:
        figname = "confusion_matrix_normalized" if normalize else "confusion_matrix"
        fp = save_path / f"{figname}.png"
        print(f"saving confusion matrix in {fp}")
        plt.savefig(str(fp))
    else:
        plt.show()
    plt.close()


def display_detailed_qualitative_evaluation(
    annotation: Annotation,
    prediction: Prediction,
    orig_img: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    H, W, _ = orig_img.shape
    class_boolean_masks = generate_class_boolean_masks2(
        annotation=annotation,
        prediction=prediction,
        W=W,
        H=H,
    )
    cf_matrix = make_confusion_matrix_from_class_boolean_masks(class_boolean_masks)
    plot_confusion_matrix(cf_matrix, normalize=True, save_path=save_path)
    plot_confusion_matrix(cf_matrix, normalize=False, save_path=save_path)

    metrics = get_evaluation_metrics(
        annotation=annotation,
        prediction=prediction,
        W=W,
        H=H,
        confusion_matrix=cf_matrix,
    )

    display_evaluation_metrics(
        metrics, save_filepath=None if not save_path else save_path / "metrics.json"
    )

    display_annotation_prediction(
        orig_img,
        annotation,
        prediction,
        save_filepath=None
        if not save_path
        else save_path / "annotation_and_prediction.png",
    )
    display_mask_comparison(
        orig_img,
        class_boolean_masks["soft_coral"]["annotation"],
        class_boolean_masks["soft_coral"]["prediction"],
        class_mask="soft coral",
        save_filepath=None
        if not save_path
        else save_path / "mask_comparison_soft_coral_annotation_vs_prediction.png",
    )
    display_mask_comparison(
        orig_img,
        class_boolean_masks["hard_coral"]["annotation"],
        class_boolean_masks["hard_coral"]["prediction"],
        class_mask="hard coral",
        save_filepath=None
        if not save_path
        else save_path / "mask_comparison_hard_coral_annotation_vs_prediction.png",
    )
    matches_summary = get_matches_summary(annotation, prediction, orig_img)
    display_matches_summary(
        matches_summary,
        annotation,
        prediction,
        orig_img,
        save_filepath=None if not save_path else save_path / "matches_summary.png",
    )


def get_annotation_and_prediction(model, image_filepath: Path) -> dict:
    # Prediction
    results = model.predict(image_filepath)
    prediction = get_prediction_data(results[0])
    orig_img = results[0].orig_img

    # Loading ground truth
    stem = image_filepath.stem
    label_filename = f"{stem}.txt"
    train_data_root_path = image_filepath.parent.parent.parent
    label_filepath = label_filename_to_label_filepath(
        train_data_root_path=train_data_root_path, filename=label_filename
    )
    annotation_raw = slurp(label_filepath)
    annotation = parse_annotation_raw(annotation_raw, orig_img)

    return {
        "annotation": annotation,
        "prediction": prediction,
        "orig_img": orig_img,
    }


def image_filepath_to_annotation(
    image_filepath: Path, orig_img: np.ndarray
) -> Annotation:
    # Loading ground truth
    stem = image_filepath.stem
    label_filename = f"{stem}.txt"
    train_data_root_path = image_filepath.parent.parent.parent
    label_filepath = label_filename_to_label_filepath(
        train_data_root_path=train_data_root_path, filename=label_filename
    )
    annotation_raw = slurp(label_filepath)
    annotation = parse_annotation_raw(annotation_raw, orig_img)
    return annotation


def get_annotations_and_predictions(
    model, image_filepaths: list[Path], batch_size: int = 16
) -> list[dict]:
    print(
        f"Predictions for {len(image_filepaths)} image_filepaths with batch_size {batch_size}"
    )
    result = []
    for i in trange(0, len(image_filepaths), batch_size):
        batch_image_filepaths = image_filepaths[i : i + batch_size]
        results_inference = model.predict(batch_image_filepaths)
        orig_imgs = [
            load_image_filepath(image_filepath)
            for image_filepath in batch_image_filepaths
        ]
        predictions = [
            get_prediction_data(results_inference[i])
            for i in range(len(batch_image_filepaths))
        ]
        annotations = [
            image_filepath_to_annotation(image_filepath, orig_imgs[i])
            for i, image_filepath in enumerate(batch_image_filepaths)
        ]
        # TODO: remove orig_img from return to save memory
        # batch_results = [{'annotation': annotation, 'prediction': prediction, 'W': orig_img.shape[0], 'H': orig_img.shape[1]} for annotation, prediction, orig_img in zip(annotations, predictions, orig_imgs)]

        batch_results = [
            {
                "annotation": annotation,
                "prediction": prediction,
                "W": orig_img.shape[1],
                "H": orig_img.shape[0],
            }
            for annotation, prediction, orig_img in zip(
                annotations, predictions, orig_imgs
            )
        ]
        result.extend(batch_results)
    return result


def analyze_prediction_from(
    model, image_filepath: Path, save_path: Optional[Path] = None
) -> None:
    if save_path:
        if not os.path.isdir(save_path):
            print(f"Creating folder {save_path}")
            os.makedirs(save_path)

    d = get_annotation_and_prediction(model, image_filepath)

    if save_path:
        fp = save_path / "orig_img.jpg"
        print(f"Persisting orig_img in {fp}")
        cv2.imwrite(str(fp), d["orig_img"])

    display_detailed_qualitative_evaluation(
        annotation=d["annotation"],
        prediction=d["prediction"],
        orig_img=d["orig_img"],
        save_path=save_path,
    )


def average_confusion_matrix(ys: list[dict]) -> np.ndarray:
    # Average the confusion matrix
    n_classes = 3
    cf_matrix = np.zeros((n_classes, n_classes))
    for y in ys:
        cf_matrix += y["confusion_matrix"]
    return cf_matrix


def accumulate_confusion_matrices(
    cf_matrices: list[np.ndarray], n_classes=3
) -> np.ndarray:
    cf_matrix = np.zeros((n_classes, n_classes))
    if not cf_matrices:
        return cf_matrix
    else:
        # Accumulate the confusion matrices
        for cm in cf_matrices:
            cf_matrix += cm
        return cf_matrix


# REPL
cf_matrix = np.array([[240, 1.2, 23], [4, 110, 7.2], [27, 13, 1200]])


def confusion_matrix_to_dice_scores(
    confusion_matrix: np.ndarray, class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING
) -> dict:
    """Returns the dice scores as a dictionnary of labels to scores given the
    `conf_matrix`."""
    num_classes = confusion_matrix.shape[0]
    dice_scores = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives
        tp = confusion_matrix[i, i]
        # False positives - sum of column for class i, minus TP
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False negatives - sum of row for class i, minus TP
        fn = np.sum(confusion_matrix[i, :]) - tp

        # IoU for class i
        dice = 2 * tp / (2 * tp + fp + fn)
        dice_scores[i] = dice

    return {
        **{class_to_label_mapping.get(i): dice_scores[i] for i in range(num_classes)},
        "mean": np.sum(dice_scores) / num_classes,
    }


def confusion_matrix_to_iou_scores(
    confusion_matrix: np.ndarray, class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING
) -> dict:
    """Returns the IoU scores as a dictionnary of labels to scores given the
    `conf_matrix`."""
    num_classes = confusion_matrix.shape[0]
    iou_scores = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives
        tp = confusion_matrix[i, i]
        # False positives - sum of column for class i, minus TP
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False negatives - sum of row for class i, minus TP
        fn = np.sum(confusion_matrix[i, :]) - tp

        # IoU for class i
        iou = tp / (tp + fp + fn)
        iou_scores[i] = iou

    return {
        **{class_to_label_mapping.get(i): iou_scores[i] for i in range(num_classes)},
        "mean": np.sum(iou_scores) / num_classes,
    }


def confusion_matrix_to_evaluation_metrics(
    confusion_matrix: np.ndarray,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
) -> dict:
    result_iou_scores = confusion_matrix_to_iou_scores(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    result_dice_scores = confusion_matrix_to_dice_scores(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    result_precision_recall_f1 = confusion_matrix_to_precision_recall_f1(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    return {
        "iou": {**result_iou_scores},
        "dice": {**result_dice_scores},
        **result_precision_recall_f1,
    }


def display_model_evaluation(
    evaluation_metrics,
    confusion_matrix,
    save_path: Optional[Path] = None,
) -> None:
    display_evaluation_metrics(
        evaluation_metrics,
        save_filepath=None if not save_path else save_path / "metrics.json",
    )
    print("")
    plot_confusion_matrix(confusion_matrix, normalize=False, save_path=save_path)
    plot_confusion_matrix(confusion_matrix, normalize=True, save_path=save_path)


def _handle_one(e) -> dict:
    class_boolean_masks = generate_class_boolean_masks2(
        annotation=e["annotation"],
        prediction=e["prediction"],
        W=e["W"],
        H=e["H"],
    )

    cf_matrix = make_confusion_matrix_from_class_boolean_masks(class_boolean_masks)

    evaluation_metrics = get_evaluation_metrics(
        annotation=e["annotation"],
        prediction=e["prediction"],
        W=e["W"],
        H=e["H"],
        confusion_matrix=cf_matrix,
    )

    return {
        "evaluation_metrics": evaluation_metrics,
        "confusion_matrix": cf_matrix,
    }


def from_cache_or_run(cache_filepath: Path, thunk):
    """Gets the data from `cache_filepath` or run the `thunk` and persists the
    computation to `cache_filepath`.

    It uses pickle to serialize the result of the computation.
    """
    if os.path.exists(cache_filepath):
        print(f"Loading cache {cache_filepath}")
        with open(cache_filepath, "rb") as f:
            return pickle.load(f)
    else:
        result = thunk()
        print(f"Persisting cache {cache_filepath}")
        with open(cache_filepath, "wb") as f:
            pickle.dump(result, f)
            print(f"Done persisting {cache_filepath}")
        return result


def evaluate_model(
    model,
    model_train_number: int,
    image_filepaths: list[Path],
    batch_size: int = 10,
    cache=True,
    cache_path: Path = CACHE_PATH,
    cv_task: str = "segment",
    save_path: Optional[Path] = None,
) -> dict:
    """Runs some evaluation on the `model` and the validation images
    `image_filepaths`. Display the mean iou and dice scores. Display the
    confusion matrix of per pixel class accuracy. Returns a dict with all the
    derived data.

    If `cache` is set to True, it will cache intermediate results for faster recomputing.
    """

    # Cache filepaths
    s = "".join([str(fp) for fp in image_filepaths])
    digest = hashlib.sha1(s.encode()).hexdigest()
    cache_path_run = (
        cache_path / cv_task / f"model_number_{model_train_number}" / digest
    )
    annotations_and_predictions_filepath = (
        cache_path_run / "annotations_and_prediction.pkl"
    )
    evaluation_metrics_filepath = cache_path_run / "evaluation_metrics.pkl"
    model_evaluation_result_filepath = cache_path_run / "evaluate_model.pkl"

    if cache and not os.path.isdir(annotations_and_predictions_filepath.parent):
        print(f"Creating cache folder {annotations_and_predictions_filepath.parent}")
        os.makedirs(annotations_and_predictions_filepath.parent)

    # Annotations and Predictions
    xs = from_cache_or_run(
        annotations_and_predictions_filepath,
        lambda: get_annotations_and_predictions(
            model, image_filepaths, batch_size=batch_size
        ),
    )

    print(f"\n\nGenerating metrics...")
    ys = []
    with mp.Pool(mp.cpu_count() - 2) as pool:
        ys = from_cache_or_run(
            evaluation_metrics_filepath,
            lambda: list(tqdm(pool.imap(_handle_one, xs), total=len(xs))),
        )

    conf_matrix = accumulate_confusion_matrices([y["confusion_matrix"] for y in ys])

    result = from_cache_or_run(
        model_evaluation_result_filepath,
        lambda: {
            "evaluation_metrics": confusion_matrix_to_evaluation_metrics(conf_matrix),
            "confusion_matrix": conf_matrix,
        },
    )

    display_model_evaluation(
        evaluation_metrics=result["evaluation_metrics"],
        confusion_matrix=result["confusion_matrix"],
        save_path=save_path,
    )

    return result


def display_model_training_overview(
    model_root_dir: Path, save_filepath: Optional[Path] = None
) -> None:
    """Displays the training graphs from a YOLOv8 train run using the
    `model_root_dir`."""
    nrows, ncols = (4, 2)
    f, axs = plt.subplots(nrows, ncols, figsize=(20, 30))

    # Turning all axes off
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].set_axis_off()

    plt.subplot(nrows, 1, 1)
    plt.imshow(Image.open(model_root_dir / "results.png"))
    plt.title(f"Training results")
    plt.axis("off")

    plt.subplot(nrows, 2, 3)
    plt.imshow(Image.open(model_root_dir / "MaskPR_curve.png"))
    plt.title(f"MaskPR curve")

    plt.subplot(nrows, 2, 4)
    plt.imshow(Image.open(model_root_dir / "MaskF1_curve.png"))
    plt.title(f"MaskF1 curve")

    plt.subplot(nrows, 2, 5)
    plt.imshow(Image.open(model_root_dir / "val_batch0_labels.jpg"))
    plt.title(f"val_batch0_labels")

    plt.subplot(nrows, 2, 6)
    plt.imshow(Image.open(model_root_dir / "val_batch0_pred.jpg"))
    plt.title(f"val_batch0_pred")

    plt.subplot(nrows, 2, 7)
    plt.imshow(Image.open(model_root_dir / "val_batch1_labels.jpg"))
    plt.title(f"val_batch1_labels")

    plt.subplot(nrows, 2, 8)
    plt.imshow(Image.open(model_root_dir / "val_batch1_pred.jpg"))
    plt.title(f"val_batch1_pred")

    if save_filepath:
        print(f"saving model training overview in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()

    plt.close()


def get_validation_set(train_data_root_path: Path) -> dict:
    validation_image_filenames = os.listdir(train_data_root_path / "val" / "images")
    validation_image_filepaths = [
        image_filename_to_image_filepath(
            train_data_root_path=train_data_root_path, filename=f
        )
        for f in validation_image_filenames
    ]
    train_image_filenames = os.listdir(train_data_root_path / "train" / "images")

    validation_label_filenames = os.listdir(train_data_root_path / "val" / "labels")
    validation_label_filepaths = [
        label_filename_to_label_filepath(
            train_data_root_path=train_data_root_path, filename=f
        )
        for f in validation_label_filenames
    ]
    return {
        "validation_image_filepaths": validation_image_filepaths,
        "validation_label_filepaths": validation_label_filepaths,
    }


# %% [markdown]
# ## Model evaluation
# In this section, we evaluate the model performance.

# %% [markdown]
# ### Quantitative evaluation
#
# In this section, we evaluate the model performance on the full validation set and report the different metrics and confusion matrix.
# Note: the script is currenlty quite slow (especially on CPU) -


# %%
def get_validation_set(train_data_root_path: Path) -> dict:
    validation_image_filenames = os.listdir(train_data_root_path / "val" / "images")
    validation_image_filepaths = [
        image_filename_to_image_filepath(
            train_data_root_path=train_data_root_path, filename=f
        )
        for f in validation_image_filenames
    ]
    train_image_filenames = os.listdir(train_data_root_path / "train" / "images")

    validation_label_filenames = os.listdir(train_data_root_path / "val" / "labels")
    validation_label_filepaths = [
        label_filename_to_label_filepath(
            train_data_root_path=train_data_root_path, filename=f
        )
        for f in validation_label_filenames
    ]
    return {
        "validation_image_filepaths": validation_image_filepaths,
        "validation_label_filepaths": validation_label_filepaths,
    }


QuantitativeEvalResult = dict


def get_train_data_root_path(benthic_dataset_version: str) -> Path:
    if benthic_dataset_version == "benthic_datasets":
        return Path(
            "/home/chouffe/fruitpunch/challenges/coralreefs2/data/05_model_input/yolov8/v0/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA"
        )
    elif benthic_dataset_version == "benthic_datasets2":
        return Path(
            "/home/chouffe/fruitpunch/challenges/coralreefs2/data/05_model_input/yolov8/v1/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA"
        )
    else:
        return Path(
            "/home/chouffe/fruitpunch/challenges/coralreefs2/data/05_model_input/yolov8/v1/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA"
        )


def model_train_number_to_model_root_path(
    model_train_number: int, yolov8_models_root_path: Path = YOLOV8_MODELS_ROOT_PATH
) -> Path:
    return yolov8_models_root_path / "runs" / "segment" / f"train{model_train_number}"


def load_yolov8_model(model_train_number: int):
    model_root_path = model_train_number_to_model_root_path(
        model_train_number=model_train_number
    )
    model_weights_path = model_root_path / "weights" / "best.pt"
    print(f"Loading the model from {model_root_path}")
    model = YOLO(model_weights_path)
    return model


def quantitative_eval(
    model_train_number: int,
    N_sample: Optional[int] = None,
    batch_size: int = 5,
    save_path: Optional[Path] = None,
) -> QuantitativeEvalResult:
    if save_path:
        if not os.path.isdir(save_path):
            print(f"Creating folder {save_path}")
            os.makedirs(save_path)

    model_root_path = model_train_number_to_model_root_path(model_train_number)
    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    benthic_dataset_version = Path(args_yaml_content["data"]).parts[4]
    print(f"benthic_dataset_version: {benthic_dataset_version}")
    train_data_root_path = get_train_data_root_path(benthic_dataset_version)
    print(f"train_data_root_path: {train_data_root_path}")

    # Loading the model
    print(f"Loading the model from {model_root_path}")
    model = load_yolov8_model(model_train_number=model_train_number)
    model.info()

    display_model_training_overview(
        model_root_path,
        save_filepath=None if not save_path else save_path / "overview.png",
    )

    # Getting the validation set
    validation_set = get_validation_set(train_data_root_path=train_data_root_path)
    validation_image_filepaths = validation_set["validation_image_filepaths"]

    # Number of image_filepaths to evaluate the model on
    N = N_sample or len(validation_image_filepaths)

    result = evaluate_model(
        model=model,
        model_train_number=model_train_number,
        image_filepaths=validation_image_filepaths[:N],
        batch_size=batch_size,
        save_path=save_path,
    )

    return result


def qualitative_eval(
    model_train_number: int,
    random_seed: int = 0,
    N_samples: int = 10,
    save_path: Optional[Path] = None,
):
    # Loading the model
    model = load_yolov8_model(model_train_number=model_train_number)
    model.info()

    model_root_path = model_train_number_to_model_root_path(model_train_number)
    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    benthic_dataset_version = Path(args_yaml_content["data"]).parts[4]
    print(f"benthic_dataset_version: {benthic_dataset_version}")
    train_data_root_path = get_train_data_root_path(benthic_dataset_version)
    print(f"train_data_root_path: {train_data_root_path}")

    # Getting the validation set
    validation_set = get_validation_set(train_data_root_path=train_data_root_path)
    validation_image_filepaths = validation_set["validation_image_filepaths"]

    random.seed(random_seed)
    indices = random.sample(range(0, len(validation_image_filepaths)), N_samples)

    for idx in indices:
        image_filepath = validation_image_filepaths[idx]
        digest = hashlib.sha1(str(image_filepath).encode()).hexdigest()
        save_path_prediction = None if not save_path else save_path / digest
        analyze_prediction_from(
            model=model, image_filepath=image_filepath, save_path=save_path_prediction
        )


def full_evaluation(
    model_train_number: int,
    random_seed: int = 0,
    save_path_root: Optional[Path] = None,
    N_samples_quantitative: Optional[int] = None,
    N_samples_qualitative: int = 10,
) -> None:
    print(f"Running quantitative evaluation of model number {model_train_number}")
    save_path_quantitative = (
        None
        if not save_path_root
        else save_path_root
        / f"model_train_number_{model_train_number}"
        / "quantitative"
    )
    quantitative_eval(
        model_train_number=model_train_number,
        N_sample=N_samples_quantitative,
        save_path=save_path_quantitative,
    )
    print(f"Running qualitative evaluation of model number {model_train_number}")
    save_path_qualitative = (
        None
        if not save_path_root
        else save_path_root / f"model_train_number_{model_train_number}" / "qualitative"
    )
    qualitative_eval(
        model_train_number=model_train_number,
        random_seed=random_seed,
        N_samples=N_samples_qualitative,
        save_path=save_path_qualitative,
    )


# %%
save_path_root = Path(
    "/home/chouffe/fruitpunch/challenges/coralreefs2/models/yolov8/evaluation/"
)
N_samples_qualitative = 10
for model_train_number in [14, 28]:
    print(f"Evaluating model {model_train_number}")
    full_evaluation(
        model_train_number=model_train_number,
        N_samples_qualitative=N_samples_qualitative,
        save_path_root=save_path_root,
    )

# %%
full_evaluation(
    model_train_number=31,
    random_seed=0,
    # N_samples_quantitative=6,
    N_samples_qualitative=10,
    save_path_root=save_path_root,
)

# %%
quantitative_eval(
    model_train_number=28,
    save_path=save_path_root / "model_train_number_28/quantitative",
)
quantitative_eval(
    model_train_number=14,
    save_path=save_path_root / "model_train_number_14/quantitative",
)
