import hashlib
import logging
import os
import random
import shutil
from functools import reduce
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
import torchvision.transforms.functional as F
import yaml
from pandas.plotting import table
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Black (0,0,0): Others, Red (255,0,0): Hard Coral, Blue (0,0,255): Soft Coral
CLASS_COLOR_MAPPING = {0: [255, 0, 0], 1: [0, 0, 255], 2: [0, 0, 0]}
CLASS_NAME_MAPPING = {0: "Hard Coral", 1: "Soft Coral", 2: "Others"}
LABEL_NAME_MAPPING = {v: k for k, v in CLASS_NAME_MAPPING.items()}

# Reef Support dense masks have interior and contour colors that are different
REEF_SUPPORT_COLORS = {
    "K": [0, 0, 0],
    "R": [255, 0, 0],
    "Y": [255, 255, 0],
    "B": [0, 0, 255],
    "O": [255, 165, 0],
}
REEF_SUPPORT_COLOR_MAPPING = {
    "R": 0,
    "Y": 0,
    "B": 1,
    "O": 1,
    "K": 2,
}


def encode_mask_with_contours(
    mask,
    colors=REEF_SUPPORT_COLORS,
    color_mapping=REEF_SUPPORT_COLOR_MAPPING,
):
    """Convert dense ground truth masks with contours to label encoding
    format."""
    mask_encoded = np.zeros(mask.shape, dtype=int)
    for color_code, label in color_mapping.items():
        mask_encoded[np.all(mask == colors[color_code], axis=-1), :] = label
    return mask_encoded[:, :, :]


def image_filepath_to_mask_path(image_filepath: Path) -> Path:
    """Converts an image filepath from the dataset into its associated mask
    filepath."""
    return (
        image_filepath.parent.parent
        / "masks_stitched"
        / f"{image_filepath.stem}_mask.png"
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Loads a mask image."""
    mask = cv2.imread(str(mask_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask


def evaluate_sample(
    eval_functions: dict,
    num_classes: int,
    mask_torch: torch.Tensor,
    prediction_torch: torch.Tensor,
) -> dict:
    """Evaluates a single sample on the provided `eval_functions`."""
    eval_results = {}
    for metric_name in eval_functions.keys():
        eval_result = eval_functions[metric_name](prediction_torch, mask_torch)
        if "_class" in metric_name:
            for class_id in range(num_classes):
                eval_results[f"{metric_name}_{class_id}"] = eval_result[class_id].item()
        elif metric_name.startswith("confusion_matrix"):
            eval_results[metric_name] = [eval_result.numpy()]
        else:
            eval_results[metric_name] = eval_result.item()
    return eval_results


def resize(
    mask: np.ndarray,
    dim: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
):
    """Resize the mask to the provided `dim` using the interpolation method.

    `dim`: (W, H) format
    """
    return cv2.resize(mask, dsize=dim, interpolation=interpolation)


# def dense_mask_from_yolov8(
#     prediction_yolov8,
#     class_color_mapping=CLASS_COLOR_MAPPING,
# ) -> np.ndarray:
#     H, W = prediction_yolov8.orig_shape
#     # Black prediction
#     prediction = np.zeros((H, W, 3), dtype=int)
#     if not prediction_yolov8.masks:
#         return prediction
#     else:
#         prediction_yolov8_masks = prediction_yolov8.masks.data.to("cpu")
#         prediction_yolov8_masks_resized = F.resize(prediction_yolov8_masks, [H, W])
#         prediction_classes = prediction_yolov8.boxes.cls.to("cpu")
#         class_hard_coral = 0
#         class_soft_coral = 1
#         indices_hard = torch.where(prediction_classes == class_hard_coral)[0]
#         indices_soft = torch.where(prediction_classes == class_soft_coral)[0]

#         binary_mask_soft_coral = (
#             reduce(
#                 lambda m1, m2: torch.max(m1, m2),
#                 prediction_yolov8_masks_resized[indices_soft],
#                 torch.zeros((H, W)),
#             )
#             .numpy()
#             .astype("bool")
#         )

#         binary_mask_hard_coral = (
#             reduce(
#                 lambda m1, m2: torch.max(m1, m2),
#                 prediction_yolov8_masks_resized[indices_hard],
#                 torch.zeros((H, W)),
#             )
#             .numpy()
#             .astype("bool")
#         )

#         color_hard_coral = class_color_mapping.get(class_hard_coral)
#         color_soft_coral = class_color_mapping.get(class_soft_coral)
#         assert color_hard_coral == [255, 0, 0]
#         assert color_soft_coral == [0, 0, 255]
#         prediction[binary_mask_hard_coral] = color_hard_coral
#         prediction[binary_mask_soft_coral] = color_soft_coral
#         return prediction


# TODO: handle the ratio pad here: using this: https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.scale_image
def dense_mask_from_yolov8(
    prediction_yolov8,
    class_color_mapping=CLASS_COLOR_MAPPING,
) -> np.ndarray:
    H, W = prediction_yolov8.orig_shape
    if not prediction_yolov8.masks:
        return np.zeros((H, W, 3), dtype=int)
    else:
        _, h, w = prediction_yolov8.masks.data.shape

        predictions_masks = prediction_yolov8.masks.data.to("cpu").numpy()
        prediction_classes = prediction_yolov8.boxes.cls.to("cpu").numpy()

        class_hard_coral = 0
        class_soft_coral = 1
        indices_hard = np.where(prediction_classes == class_hard_coral)[0]
        indices_soft = np.where(prediction_classes == class_soft_coral)[0]

        binary_mask_hard_coral = reduce(
            lambda m1, m2: m1 | m2,
            predictions_masks[indices_hard].astype(bool),
            np.zeros((h, w), dtype=bool),
        )

        binary_mask_soft_coral = reduce(
            lambda m1, m2: m1 | m2,
            predictions_masks[indices_soft].astype(bool),
            np.zeros((h, w), dtype=bool),
        )

        prediction = np.zeros((h, w, 3), dtype="uint8")
        color_hard_coral = class_color_mapping.get(class_hard_coral)
        color_soft_coral = class_color_mapping.get(class_soft_coral)

        assert color_hard_coral == [255, 0, 0]
        assert color_soft_coral == [0, 0, 255]

        prediction[binary_mask_hard_coral] = color_hard_coral
        prediction[binary_mask_soft_coral] = color_soft_coral
        return resize(prediction, dim=(W, H))


def to_encoded_mask_torch(mask: np.ndarray) -> torch.Tensor:
    """Turn a numpy array containing the dense mask (red, blue, black colors)
    into a torch tensor that can be evaluated."""
    mask_encoded = encode_mask_with_contours(mask=mask)
    mask_torch = torch.from_numpy(mask_encoded[:, :, 0])
    return mask_torch


def from_yolov8(
    prediction_yolov8,
    class_color_mapping=CLASS_COLOR_MAPPING,
) -> torch.Tensor:
    """Returns a torch tensor with corresponds to the segmentation mask.

    Ready to be evaluated on.
    """
    prediction = dense_mask_from_yolov8(
        prediction_yolov8=prediction_yolov8,
        class_color_mapping=class_color_mapping,
    )
    return to_encoded_mask_torch(prediction)


def dense_mask_from_image_filepath(image_filepath: Path) -> np.ndarray:
    """Returns a dense mask from the image filepath."""
    mask_filepath = image_filepath_to_mask_path(image_filepath)
    mask = load_mask(mask_filepath)
    return mask


def from_image_filepath(image_filepath: Path) -> torch.Tensor:
    """Returns a mask from the image filepath - ready to be evaluated on."""
    mask = dense_mask_from_image_filepath(image_filepath)
    return to_encoded_mask_torch(mask)


def save_sample_mask_prediction(
    image_filepath: Path,
    mask: np.ndarray,
    prediction: np.ndarray,
    save_path: Path,
    save_orig_img: bool = False,
) -> None:
    """Saves mask, prediction and original image (from image_filepath) if
    save_orig_img is set to True.

    The save_path Path will be created if it does not exist yet.
    """
    if not os.path.isdir(save_path):
        logging.info(f"Creating folder {save_path}")
        os.makedirs(save_path)

    stem = image_filepath.stem
    cv2.imwrite(str(save_path / f"{stem}_ground_truth.png"), mask)
    cv2.imwrite(str(save_path / f"{stem}_prediction.png"), prediction)

    if save_orig_img:
        dest = save_path / image_filepath.name
        logging.info(f"Saving original image {image_filepath} in {dest}")
        shutil.copy(image_filepath, dest)


def process_and_evaluate_sample(
    eval_functions: dict,
    num_classes: int,
    image_filepath: Path,
    prediction_yolov8,
    save_path: Optional[Path] = None,
) -> dict:
    mask = dense_mask_from_image_filepath(image_filepath)
    mask_torch = to_encoded_mask_torch(mask)
    prediction = dense_mask_from_yolov8(prediction_yolov8)
    prediction_torch = to_encoded_mask_torch(prediction)

    if save_path:
        save_sample_mask_prediction(
            image_filepath=image_filepath,
            mask=mask,
            prediction=prediction,
            save_path=save_path,
            save_orig_img=False,
        )

    return evaluate_sample(eval_functions, num_classes, mask_torch, prediction_torch)


def process_and_evaluate_samples(
    eval_functions: dict,
    num_classes: int,
    samples: list[dict],
    save_path: Optional[Path] = None,
):
    eval_results_list = []
    logging.info("Evaluate samples")
    for sample in tqdm(samples):
        eval_results = process_and_evaluate_sample(
            eval_functions=eval_functions,
            num_classes=num_classes,
            image_filepath=sample["image_filepath"],
            prediction_yolov8=sample["prediction_yolov8"],
            save_path=save_path,
        )
        eval_results_list.append(eval_results)
    return eval_results_list


def evaluate_summary(eval_functions: dict, num_classes: int) -> dict:
    """Evaluate dataset level summary for all samples that were evaluated prior
    to this call."""
    eval_summary = {}
    logging.info("Evaluate summary")
    for metric_name in tqdm(eval_functions.keys()):
        eval_result = eval_functions[metric_name].compute()
        if "_class" in metric_name:
            for class_id in range(num_classes):
                eval_summary[f"{metric_name}_{class_id}"] = eval_result[class_id].item()
        elif metric_name.startswith("confusion_matrix"):
            eval_summary[metric_name] = [eval_result.numpy()]
        else:
            eval_summary[metric_name] = eval_result.item()
    return eval_summary


def to_pandas(confusion_matrix: np.ndarray, class_name_mapping=CLASS_NAME_MAPPING):
    return pd.DataFrame(
        confusion_matrix,
        index=class_name_mapping.values(),
        columns=class_name_mapping.values(),
    )


def yaml_content(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_yolov8_model(model_root_path: Path):
    model_weights_path = model_root_path / "weights" / "best.pt"
    logging.info(f"Loading the model from {model_root_path}")
    model = YOLO(model_weights_path)
    return model


def display_model_training_overview(
    model_root_dir: Path,
    save_filepath: Optional[Path] = None,
) -> None:
    """Displays the training graphs from a YOLOv8 train run using the
    `model_root_dir`."""
    nrows, ncols = (4, 2)
    f, axs = plt.subplots(nrows, ncols, figsize=(20, 30))

    # Turning all axes off
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].set_axis_off()

    # Image paths
    path_results = model_root_dir / "results.png"
    path_maskpr_curve = model_root_dir / "MaskPR_curve.png"
    path_maskf1_curve = model_root_dir / "MaskF1_curve.png"
    path_val_batch0_labels = model_root_dir / "val_batch0_labels.jpg"
    path_val_batch0_pred = model_root_dir / "val_batch0_pred.jpg"
    path_val_batch1_labels = model_root_dir / "val_batch1_labels.jpg"
    path_val_batch1_pred = model_root_dir / "val_batch1_pred.jpg"

    plt.subplot(nrows, 1, 1)

    if path_results.exists():
        plt.imshow(Image.open(path_results))
        plt.title("Training results")
        plt.axis("off")

    if path_maskpr_curve.exists():
        plt.subplot(nrows, 2, 3)
        plt.imshow(Image.open(path_maskpr_curve))
        plt.title("MaskPR curve")

    if path_maskf1_curve.exists():
        plt.subplot(nrows, 2, 4)
        plt.imshow(Image.open(path_maskf1_curve))
        plt.title("MaskF1 curve")

    if path_val_batch0_labels.exists():
        plt.subplot(nrows, 2, 5)
        plt.imshow(Image.open(path_val_batch0_labels))
        plt.title("val_batch0_labels")

    if path_val_batch0_pred.exists():
        plt.subplot(nrows, 2, 6)
        plt.imshow(Image.open(path_val_batch0_pred))
        plt.title("val_batch0_pred")

    if path_val_batch1_labels.exists():
        plt.subplot(nrows, 2, 7)
        plt.imshow(Image.open(path_val_batch1_labels))
        plt.title("val_batch1_labels")

    if path_val_batch1_pred.exists():
        plt.subplot(nrows, 2, 8)
        plt.imshow(Image.open(path_val_batch1_pred))
        plt.title("val_batch1_pred")

    if save_filepath:
        logging.info(f"saving model training overview in {save_filepath}")
        plt.savefig(str(save_filepath))
    else:
        plt.show()

    plt.close()


def image_filename_to_image_filepath(
    train_data_root_path: Path, filename: str, split: str = "val"
) -> Path:
    return train_data_root_path / split / "images" / filename


def label_filename_to_label_filepath(
    train_data_root_path: Path, filename: str, split: str = "val"
) -> Path:
    return train_data_root_path / split / "labels" / filename


def get_image_filepaths(
    data_config: dict,
    raw_images_root_path: Path,
    data_split="test",
) -> list[Path]:
    """data_split can be `val`, `test`, `train` or a list of them like so:

    ['test', 'val'].
    """
    if type(data_split) == list:
        image_filepaths = []
        for m in data_split:
            image_filepaths.extend(
                get_image_filepaths(
                    data_config=data_config,
                    raw_images_root_path=raw_images_root_path,
                    data_split=m,
                )
            )
        return image_filepaths
    else:
        image_filepaths = []
        config_key = f"{data_split}_dataset"
        for region in data_config[config_key]:
            for filename in data_config[config_key][region]:
                fp = raw_images_root_path / region / "images" / filename
                image_filepaths.append(fp)
        return image_filepaths


def export_dataframe_as_png(df: pd.DataFrame, save_path: Path) -> None:
    """Exports the pandas dataframe `df` as a png.

    Makes it easy to share in reports.
    """
    fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, df, loc="upper right", colWidths=[0.1] * len(df.columns))
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(10)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1.2, 1.2)  # change size table
    plt.savefig(str(save_path), transparent=False)
    plt.close()


def plot_confusion_matrix(
    cf_matrix: np.ndarray,
    normalize=False,
    class_name_mapping: dict = CLASS_NAME_MAPPING,
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

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ## Ticket labels - List must be in alphabetical order
    labels = sorted([class_name_mapping[i] for i in range(3)])
    labels = [class_name_mapping[i] for i in range(3)]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save the confusion matrix
    if save_path:
        figname = "confusion_matrix_normalized" if normalize else "confusion_matrix"
        fp = save_path / f"{figname}.png"
        logging.info(f"saving confusion matrix in {fp}")
        plt.savefig(str(fp))
    else:
        plt.show()
    plt.close()


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def write_config_yaml(
    save_path: Path,
    data_split,
    data_root_path: Path,
) -> None:
    """Writes the config.yaml file that describes the quantitative evaluation.

    data_split can be a list or a string.
    """

    yaml_filepath = data_root_path / "config.yaml"
    config_yaml_content = yaml_content(yaml_filepath)
    evaluation_dataset_size = 0
    if type(data_split) == list:
        for m in data_split:
            evaluation_dataset_size += config_yaml_content[f"{m}_dataset_size"]
    else:
        evaluation_dataset_size += config_yaml_content[f"{data_split}_dataset_size"]

    data = {
        "evaluation_data_split": data_split,
        "evaluation_dataset": f"{data_split}_dataset",
        "evaluation_dataset_size": evaluation_dataset_size,
        "dataset_config_path": str(yaml_filepath),
        "dataset_config": config_yaml_content,
    }

    with open(save_path, "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def write_eval_summary_yaml(
    save_path: Path,
    eval_summary: dict,
) -> None:
    """Writes the yaml file that describes the quantitative evaluation
    results."""

    # TODO: improve, replace class_0 by class name
    data = {**eval_summary}
    del data["confusion_matrix"]
    del data["confusion_matrix_normalized"]

    with open(save_path, "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def make_eval_functions(num_classes: int = 3):
    """Returns a dict of torchmetrics functions."""
    return dict(
        pa=torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=3,
        ),
        mpa=torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
            average="macro",
        ),
        pa_class=torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
            average="none",
        ),
        miou=torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
            average="macro",
        ),
        iou_class=torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
            average="none",
        ),
        mdice=torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        ),
        dice_class=torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="none",
        ),
        confusion_matrix=torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
        ),
        confusion_matrix_normalized=torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=num_classes,
            normalize="true",
        ),
    )


def qualitative_eval(
    model_root_path: Path,
    data_root_path: Optional[Path],
    raw_images_root_path: Path,
    data_split: str,
    save_path: Path,
    random_seed: int = 42,
    N_samples: int = 10,
) -> None:
    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    data_root_path = data_root_path or Path(args_yaml_content["data"]).parent
    data_config = yaml_content(data_root_path / "config.yaml")

    if save_path:
        if not os.path.isdir(save_path):
            logging.info(f"Creating folder {save_path}")
            os.makedirs(save_path)

    # Loading the model
    model = load_yolov8_model(model_root_path=model_root_path)

    image_filepaths = get_image_filepaths(
        data_config=data_config,
        data_split=data_split,
        raw_images_root_path=raw_images_root_path,
    )
    logging.info(f"image_filepaths to evaluate: {len(image_filepaths)}")

    batch = random.Random(random_seed).sample(image_filepaths, N_samples)
    logging.info(f"Generating predictions for {len(batch)} samples")
    for image_filepath in tqdm(batch):
        mask = dense_mask_from_image_filepath(image_filepath)
        prediction_yolov8 = model(image_filepath)[0]
        prediction = dense_mask_from_yolov8(prediction_yolov8)
        save_sample_mask_prediction(
            image_filepath=image_filepath,
            mask=mask,
            prediction=prediction,
            save_path=save_path,
            save_orig_img=True,
        )


def _rename_df_columns(df):
    """Small utility function to rename the df_summary dataframe columns,
    replacing the class_k with the actual class labels."""
    columns = list(df.columns)
    mapping = {}
    for c in columns:
        if c.endswith("class_0"):
            mapping[c] = c.replace("class_0", "hard")
        if c.endswith("class_1"):
            mapping[c] = c.replace("class_1", "soft")
        if c.endswith("class_2"):
            mapping[c] = c.replace("class_2", "other")
    return df.rename(columns=mapping)


def quantitative_eval(
    model_root_path: Path,
    data_root_path: Optional[Path],
    raw_images_root_path: Path,
    data_split: str,
    save_path: Path,
    batch_size: int = 5,
    save_predictions_path: Optional[Path] = None,
):
    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    data_root_path = data_root_path or Path(args_yaml_content["data"]).parent
    data_config = yaml_content(data_root_path / "config.yaml")

    if save_path:
        if not os.path.isdir(save_path):
            logging.info(f"Creating folder {save_path}")
            os.makedirs(save_path)

    # Loading the model
    logging.info(f"Loading the model from {model_root_path}")
    model = load_yolov8_model(model_root_path=model_root_path)

    if (model_root_path / "results.png").exists():
        display_model_training_overview(
            model_root_path,
            save_filepath=save_path / "overview.png",
        )

    image_filepaths = get_image_filepaths(
        data_config=data_config,
        data_split=data_split,
        raw_images_root_path=raw_images_root_path,
    )
    logging.info(f"image_filepaths to evaluate: {len(image_filepaths)}")

    write_config_yaml(
        save_path=save_path / "config.yaml",
        data_split=data_split,
        data_root_path=data_root_path,
    )

    predictions_yolov8 = []

    # We currently fix the batch size to 1 otherwise the prediction masks are
    # padded with black pixels and this is not handled properly yet in the code
    # logic
    batch_size = 1

    for i in tqdm(range(0, len(image_filepaths), batch_size)):
        batch = image_filepaths[i : i + batch_size]
        preds = model(batch)
        predictions_yolov8.extend(preds)

    assert len(predictions_yolov8) == len(
        image_filepaths
    ), "Should have equal number of preds and filepaths"

    samples = [
        {"image_filepath": fp, "prediction_yolov8": pred}
        for fp, pred in zip(image_filepaths, predictions_yolov8)
    ]

    num_classes = 3
    eval_functions = make_eval_functions(num_classes=num_classes)
    eval_results = process_and_evaluate_samples(
        eval_functions=eval_functions,
        num_classes=num_classes,
        samples=samples,
        save_path=None
        if not save_predictions_path
        else save_predictions_path / model_root_path.stem,
    )
    eval_summary = evaluate_summary(
        eval_functions=eval_functions,
        num_classes=num_classes,
    )

    write_eval_summary_yaml(
        save_path=save_path / "summary.yaml",
        eval_summary=eval_summary,
    )

    plot_confusion_matrix(
        cf_matrix=eval_summary["confusion_matrix"][0],
        normalize=True,
        class_name_mapping=CLASS_NAME_MAPPING,
        save_path=save_path,
    )
    plot_confusion_matrix(
        cf_matrix=eval_summary["confusion_matrix"][0],
        normalize=False,
        class_name_mapping=CLASS_NAME_MAPPING,
        save_path=save_path,
    )

    # Save eval_results and eval_summary
    df_results = pd.DataFrame.from_records(eval_results)
    df_results["image_filepath"] = [str(fp) for fp in image_filepaths]
    df_summary = pd.DataFrame.from_records(eval_summary)

    # Export the df_summary dataframe as a png to include in reports
    metrics = [
        "miou",
        "iou_class_0",
        "iou_class_1",
        "iou_class_2",
        "mdice",
        "dice_class_0",
        "dice_class_1",
        "dice_class_2",
    ]
    df_summary_metrics_selection = df_summary[metrics]
    df_summary_metrics_selection = _rename_df_columns(df_summary_metrics_selection)
    df_summary_metrics_selection = df_summary_metrics_selection.map("{0:.2f}".format)
    export_dataframe_as_png(df_summary_metrics_selection, save_path / "summary.png")

    df_confusion_matrix_normalized = to_pandas(
        eval_summary["confusion_matrix_normalized"][0]
    )
    df_confusion_matrix = to_pandas(eval_summary["confusion_matrix"][0])
    df_results.to_csv(save_path / "results.csv")
    df_summary.to_csv(save_path / "summary.csv")
    df_confusion_matrix_normalized.to_csv(save_path / "confusion_matrix_normalized.csv")
    df_confusion_matrix.to_csv(save_path / "confusion_matrix.csv")

    return {
        "eval_summary": eval_summary,
        "eval_results": eval_results,
    }


def full_eval(
    model_root_path: Path,
    data_root_path: Optional[Path],
    raw_images_root_path: Path,
    data_split: str,
    save_path: Path,
    random_seed: int = 42,
    batch_size: int = 5,
    N_samples: int = 10,
    save_predictions_path: Optional[Path] = None,
) -> None:
    """Runs a quantitative_eval and a qualitative_eval and persists the results
    of the run."""
    model_name = model_root_path.name
    s = f"{data_root_path}.{random_seed}.{data_split}"
    digest = hashlib.sha1(s.encode()).hexdigest()
    save_path_root = save_path / model_name / digest

    logging.info(f"quantitative evaluation of {model_name}")
    quantitative_eval(
        model_root_path=model_root_path,
        data_root_path=data_root_path,
        raw_images_root_path=raw_images_root_path,
        data_split=data_split,
        save_path=save_path_root / "quantitative",
        batch_size=batch_size,
        save_predictions_path=save_predictions_path,
    )
    logging.info(f"qualitative evaluation of {model_name}")
    qualitative_eval(
        model_root_path=model_root_path,
        data_root_path=data_root_path,
        raw_images_root_path=raw_images_root_path,
        data_split=data_split,
        save_path=save_path_root / "qualitative",
        random_seed=random_seed,
        N_samples=N_samples,
    )
