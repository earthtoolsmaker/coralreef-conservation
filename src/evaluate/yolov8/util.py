import json
import logging
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ultralytics
import yaml
from datatypes import Annotation, AnnotationEntry, Polygon, Prediction

# FIXME: use fully qualified names
from metrics import *
from PIL import Image

CLASS_TO_COLOR_MAPPING = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 0, 0),
}

LABEL_TO_CLASS_MAPPING: dict[str, int] = {"soft_coral": 1, "hard_coral": 0, "other": 2}
CLASS_TO_LABEL_MAPPING: dict[int, str] = {
    v: k for k, v in LABEL_TO_CLASS_MAPPING.items()
}


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


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


def get_prediction_as_image(prediction):
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
            logging.info(f"Parsing error with annotation_raw: {annotation_raw}")
            logging.info("Returning empty annotation")
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


def img_array_to_pil_img(img_array: np.ndarray):
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
        logging.info(f"saving mask comparison in {save_filepath}")
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
        logging.info(f"saving mask comparison in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()
    plt.close()


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
        logging.info(f"saving metrics in {save_filepath}")
        save_filepath.write_text(json.dumps(metrics, indent=2))
    else:
        logging.info("Evaluation Metrics")
        logging.info("------------------")
        if "instances" in metrics:
            logging.info("Instances")
            logging.info(f"    annotation")
            logging.info(
                f"         soft coral: {metrics['instances']['annotation']['soft_coral']}"
            )
            logging.info(
                f"         hard coral: {metrics['instances']['annotation']['hard_coral']}"
            )
            logging.info(f"    prediction")
            logging.info(
                f"         soft coral: {metrics['instances']['prediction']['soft_coral']}"
            )
            logging.info(
                f"         hard coral: {metrics['instances']['prediction']['hard_coral']}"
            )

        logging.info("Dice")
        logging.info(f"    mean:       {metrics['dice']['mean']:.2}")
        logging.info(f"    soft coral: {metrics['dice']['soft_coral']:.2f}")
        logging.info(f"    hard coral: {metrics['dice']['hard_coral']:.2f}")
        logging.info(f"    other:      {metrics['dice']['other']:.2f}")
        logging.info("IoU")
        logging.info(f"    mean:       {metrics['iou']['mean']:.2}")
        logging.info(f"    soft coral: {metrics['iou']['soft_coral']:.2f}")
        logging.info(f"    hard coral: {metrics['iou']['hard_coral']:.2f}")
        logging.info(f"    other:      {metrics['iou']['other']:.2f}")


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
        logging.info(f"saving matches summary in {save_filepath}")
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
