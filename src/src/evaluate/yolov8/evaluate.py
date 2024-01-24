"""Evaluate model performance.

This script allows the user to run qualitative and quantitative
performance evaluation on the validation sets. Some artifacts are
generated to make it easier for reporting.
"""
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datatypes import Annotation, Prediction
from PIL import Image
from tqdm import tqdm, trange
from ultralytics import YOLO
from util import *


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
    logging.info(
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
    model,
    image_filepath: Path,
    filename_to_region: dict,
    save_path: Optional[Path] = None,
) -> None:
    if save_path:
        if not os.path.isdir(save_path):
            logging.info(f"Creating folder {save_path}")
            os.makedirs(save_path)

    config_path = None if not save_path else save_path / "config.yaml"
    if config_path:
        with open(config_path, "w") as f:
            filename = image_filepath.name
            data = {
                "image_filename": filename,
                "region": filename_to_region[filename],
            }
            yaml.dump(
                data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False
            )
    d = get_annotation_and_prediction(model, image_filepath)

    if save_path:
        fp = save_path / "orig_img.jpg"
        logging.info(f"Persisting orig_img in {fp}")
        cv2.imwrite(str(fp), d["orig_img"])

    display_detailed_qualitative_evaluation(
        annotation=d["annotation"],
        prediction=d["prediction"],
        orig_img=d["orig_img"],
        save_path=save_path,
    )


def display_model_evaluation(
    evaluation_metrics,
    confusion_matrix,
    save_path: Optional[Path] = None,
) -> None:
    display_evaluation_metrics(
        evaluation_metrics,
        save_filepath=None if not save_path else save_path / "metrics.json",
    )
    logging.info("")
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
        logging.info(f"Loading cache {cache_filepath}")
        with open(cache_filepath, "rb") as f:
            return pickle.load(f)
    else:
        result = thunk()
        logging.info(f"Persisting cache {cache_filepath}")
        with open(cache_filepath, "wb") as f:
            pickle.dump(result, f)
            logging.info(f"Done persisting {cache_filepath}")
        return result


def evaluate_model(
    model,
    model_name: str,
    cache_path: Path,
    image_filepaths: list[Path],
    batch_size: int = 10,
    cache: bool = True,
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
    cache_path_run = cache_path / cv_task / model_name / digest
    annotations_and_predictions_filepath = (
        cache_path_run / "annotations_and_prediction.pkl"
    )
    evaluation_metrics_filepath = cache_path_run / "evaluation_metrics.pkl"
    model_evaluation_result_filepath = cache_path_run / "evaluate_model.pkl"

    if cache and not os.path.isdir(annotations_and_predictions_filepath.parent):
        logging.info(
            f"Creating cache folder {annotations_and_predictions_filepath.parent}"
        )
        os.makedirs(annotations_and_predictions_filepath.parent)

    # Annotations and Predictions
    xs = from_cache_or_run(
        annotations_and_predictions_filepath,
        lambda: get_annotations_and_predictions(
            model, image_filepaths, batch_size=batch_size
        ),
    )
    logging.info("Generating metrics...")
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
    plt.title("Training results")
    plt.axis("off")

    plt.subplot(nrows, 2, 3)
    plt.imshow(Image.open(model_root_dir / "MaskPR_curve.png"))
    plt.title("MaskPR curve")

    plt.subplot(nrows, 2, 4)
    plt.imshow(Image.open(model_root_dir / "MaskF1_curve.png"))
    plt.title("MaskF1 curve")

    plt.subplot(nrows, 2, 5)
    plt.imshow(Image.open(model_root_dir / "val_batch0_labels.jpg"))
    plt.title("val_batch0_labels")

    plt.subplot(nrows, 2, 6)
    plt.imshow(Image.open(model_root_dir / "val_batch0_pred.jpg"))
    plt.title("val_batch0_pred")

    plt.subplot(nrows, 2, 7)
    plt.imshow(Image.open(model_root_dir / "val_batch1_labels.jpg"))
    plt.title("val_batch1_labels")

    plt.subplot(nrows, 2, 8)
    plt.imshow(Image.open(model_root_dir / "val_batch1_pred.jpg"))
    plt.title("val_batch1_pred")

    if save_filepath:
        logging.info(f"saving model training overview in {save_filepath}")
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


def load_yolov8_model(model_root_path: Path):
    model_weights_path = model_root_path / "weights" / "best.pt"
    logging.info(f"Loading the model from {model_root_path}")
    model = YOLO(model_weights_path)
    return model


def write_config_yaml(
    path: Path,
    data_root_path: Path,
) -> None:
    """Writes the config.yaml file that describes the quantitative
    evaluation."""

    config_yaml_content = yaml_content(data_root_path / "config.yaml")

    data = {
        "val_dataset_size": config_yaml_content["val_dataset_size"],
        "val_dataset": config_yaml_content["val_dataset"],
    }

    with open(path / "config.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def quantitative_eval(
    model_root_path: Path,
    cache_path: Path,
    data_root_path: Optional[Path] = None,
    N_sample: Optional[int] = None,
    batch_size: int = 10,
    save_path: Optional[Path] = None,
) -> dict:
    digest = hashlib.sha1(str(data_root_path).encode()).hexdigest()

    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    data_root_path = data_root_path or Path(args_yaml_content["data"]).parent
    logging.info(f"data_root_path: {data_root_path}")

    if save_path:
        if not os.path.isdir(save_path / digest):
            logging.info(f"Creating folder {save_path / digest}")
            os.makedirs(save_path / digest)

    # Loading the model
    logging.info(f"Loading the model from {model_root_path}")
    model = load_yolov8_model(model_root_path=model_root_path)
    model.info()

    if (model_root_path / "results.png").exists():
        display_model_training_overview(
            model_root_path,
            save_filepath=None
            if not save_path
            else save_path / digest / "overview.png",
        )

    # Getting the validation set
    validation_set = get_validation_set(train_data_root_path=data_root_path)
    validation_image_filepaths = validation_set["validation_image_filepaths"]

    if save_path:
        logging.info("Writing config.yaml file")
        write_config_yaml(
            path=save_path / digest,
            data_root_path=data_root_path,
        )

    # Number of image_filepaths to evaluate the model on
    N = N_sample or len(validation_image_filepaths)

    result = evaluate_model(
        model=model,
        model_name=model_root_path.stem,
        cache_path=cache_path,
        image_filepaths=validation_image_filepaths[:N],
        batch_size=batch_size,
        save_path=None if not save_path else save_path / digest,
    )

    return result


def config_content_to_filename_region_mapping(config_yaml_content: dict):
    result = {}
    for region in config_yaml_content["val_dataset"].keys():
        for filename in config_yaml_content["val_dataset"][region]:
            result[filename] = region
    return result


def qualitative_eval(
    model_root_path: Path,
    data_root_path: Optional[Path] = None,
    random_seed: int = 0,
    N_samples: int = 10,
    save_path: Optional[Path] = None,
):
    # Loading the model
    model = load_yolov8_model(model_root_path=model_root_path)
    model.info()

    args_yaml_content = yaml_content(model_root_path / "args.yaml")
    data_root_path = data_root_path or Path(args_yaml_content["data"]).parent
    logging.info(f"data_root_path: {data_root_path}")
    config_yaml_content = yaml_content(data_root_path / "config.yaml")
    filename_to_region = config_content_to_filename_region_mapping(config_yaml_content)

    # Getting the validation set
    validation_set = get_validation_set(train_data_root_path=data_root_path)
    validation_image_filepaths = validation_set["validation_image_filepaths"]

    random.seed(random_seed)
    indices = random.sample(range(0, len(validation_image_filepaths)), N_samples)

    for idx in indices:
        image_filepath = validation_image_filepaths[idx]
        digest = hashlib.sha1(str(image_filepath).encode()).hexdigest()
        save_path_prediction = None if not save_path else save_path / digest

        analyze_prediction_from(
            model=model,
            image_filepath=image_filepath,
            save_path=save_path_prediction,
            filename_to_region=filename_to_region,
        )


def full_evaluation(
    model_root_path: Path,
    cache_path: Path,
    data_root_path: Optional[Path] = None,
    random_seed: int = 0,
    batch_size: int = 10,
    save_path_root: Optional[Path] = None,
    N_samples_quantitative: Optional[int] = None,
    N_samples_qualitative: int = 10,
) -> None:
    """Runs the full evaluation on the provided `model_root_path`.

    If `save_path_root` is provided, it stores the reporting in this folder.
    """
    logging.info(f"Running quantitative evaluation of model number {model_root_path}")
    save_path_quantitative = (
        None
        if not save_path_root
        else save_path_root / model_root_path.stem / "quantitative"
    )
    quantitative_eval(
        model_root_path=model_root_path,
        cache_path=cache_path,
        data_root_path=data_root_path,
        N_sample=N_samples_quantitative,
        batch_size=batch_size,
        save_path=save_path_quantitative,
    )
    logging.info(f"Running qualitative evaluation of model {model_root_path}")
    save_path_qualitative = (
        None
        if not save_path_root
        else save_path_root / model_root_path.stem / "qualitative"
    )
    qualitative_eval(
        model_root_path=model_root_path,
        data_root_path=data_root_path,
        random_seed=random_seed,
        N_samples=N_samples_qualitative,
        save_path=save_path_qualitative,
    )


def full_evaluation_all(
    models_root_path: Path,
    cache_path: Path,
    data_root_path: Optional[Path] = None,
    random_seed: int = 0,
    batch_size: int = 10,
    save_path_root: Optional[Path] = None,
    N_samples_quantitative: Optional[int] = None,
    N_samples_qualitative: int = 10,
):
    """Run a full evaluation on each subdir `model_root_path` in
    `models_root_path`."""
    model_names = os.listdir(models_root_path)
    logging.info(f"Evaluating the following {len(model_names)} models: {model_names}")
    for model_name in tqdm(model_names):
        model_root_path = models_root_path / model_name
        logging.info(f"Model: {model_name}")
        full_evaluation(
            model_root_path=model_root_path,
            cache_path=cache_path,
            data_root_path=data_root_path,
            random_seed=random_seed,
            batch_size=batch_size,
            save_path_root=save_path_root,
            N_samples_quantitative=N_samples_quantitative,
            N_samples_qualitative=N_samples_qualitative,
        )
