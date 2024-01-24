"""YOLOv8 Pytorch TXT format processor.

This script allows the user to process the raw data and turn it into the
YOLOv8 Pytorch TXT format required by YOLOv8 for training.
Here is the link to the format: https://roboflow.com/formats/yolov8-pytorch-txt
"""
import argparse
import functools
import logging
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

LABEL_TO_CLASS_MAPPING = {"soft_coral": 0, "hard_coral": 1}
CLASS_TO_LABEL_MAPPING = {v: k for k, v in LABEL_TO_CLASS_MAPPING.items()}
COLOR_TO_LABEL_MAPPING: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "other",  # Black
    (0, 0, 255): "soft_coral",  # Blue
    (255, 0, 0): "hard_coral",  # Red
}
LABEL_TO_COLOR_MAPPING = {v: k for k, v in COLOR_TO_LABEL_MAPPING.items()}

# For type hints
Contour = np.ndarray
Mask = np.ndarray
Polygon = np.ndarray


def get_all_dataset_names(path: Path) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(path / f)]


def filepath_to_ndarray(filepath: Path) -> np.ndarray:
    return cv2.imread(str(filepath))


def scaffold_output_dir(output_dir: Path) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def list_all_mask_filepaths(
    dataset_name: str,
    input_dir: Path,
) -> list[Path]:
    path = input_dir / dataset_name / "masks"
    return [path / f for f in os.listdir(path) if os.path.isfile(path / f)]


def is_only_black_pixels(mask: Mask) -> bool:
    """Returns True if the mask image is only black pixels."""
    non_black_pixels = np.any(mask != [0, 0, 0], axis=-1)
    black_pixels = ~non_black_pixels
    return black_pixels.all()


def mask_label(mask: Mask) -> str:
    """
    Returns the class of the mask (nd_array) as a string in {other, soft_coral,
    hard_coral}
    Assumption: one mask file will only label one type of coral (either blue or
    red).
    """
    idx = np.any(mask != [0, 0, 0], axis=-1)
    color_tuple = tuple(mask[idx][0])
    return COLOR_TO_LABEL_MAPPING.get(color_tuple, "other")


def normalize_polygon(polygon, W: int, H: int) -> Polygon:
    """`polygon`: numpy array of shape (_,2) containing the polygon x,y
    coordinates.

    `W`: int - width of the image / mask
    `H`: int - height of the image / mask

    returns a numpy array of shape `polygon.shape` with coordinates that are
    normalized between 0-1.

    Will throw an assertion error if all values of the result do not lie in 0-1.
    """
    copy = np.copy(polygon)
    copy = copy.astype(np.float16)
    copy[:, 0] *= 1 / W
    copy[:, 1] *= 1 / H

    assert (
        (copy >= 0) & (copy <= 1)
    ).all(), f"normalized_polygon values are not all in range 0-1, got: {copy}"

    return copy


def normalize_bounding_box(bbox, W: int, H: int):
    """Returns xcyxwh bounding box coordinate to follow the yolov8 format."""
    (x0, y0, x1, y1) = bbox
    xcenter = (x0 + x1) / 2.0 / W
    ycenter = (y0 + y1) / 2.0 / H
    w = abs(x0 - x1) / W
    h = abs(y0 - y1) / H
    return (xcenter, ycenter, w, h)


def contours_to_polygons(contours: list[Contour]) -> list[Polygon]:
    """Turn a list of contours into a list of polygons."""
    polygons = []
    for contour in contours:
        polygon_list = []
        for point in contour:
            x, y = point[0]
            polygon_list.append(np.array([x, y]))
        polygon = np.array(polygon_list)
        polygons.append(polygon)
    return polygons


def is_contour_area_large_enough(contour: Contour, threshold: int = 200) -> bool:
    return cv2.contourArea(contour) > threshold


def mask_to_contours(mask: Mask) -> list[Contour]:
    """Given a mask, it returns its contours."""
    # Loading the mask in grey, only format supported by cv2.findContours
    mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Only keep the areas that are big enough
    valid_contours = [
        contour for contour in contours if is_contour_area_large_enough(contour)
    ]
    return valid_contours


def display_contours(mask: Mask, contours: list[Contour]) -> None:
    """Display the contours information, useful for debugging."""
    N = len(contours)
    label = mask_label(mask)
    plt.subplots(1, N + 1, figsize=(15, 15))

    plt.subplot(1, N + 1, 1)
    plt.imshow(mask)
    plt.title("Mask")

    color = LABEL_TO_COLOR_MAPPING[label]
    thickness = 10
    for i, contour in enumerate(contours):
        image_black = np.zeros(mask.shape, dtype=np.uint8)
        image_contour = cv2.drawContours(image_black, [contour], -1, color, thickness)
        plt.subplot(1, N + 1, i + 2)
        plt.imshow(image_contour)
        plt.title(f"Contour {i}")

    plt.show()


# Yolov8 PyTorch TXT format
def stringify_polygon(polygon: Polygon) -> str:
    """Turns a polygon nd_array into a string in the right YOLOv8 format."""
    return " ".join([f"{x} {y}" for (x, y) in polygon])


def stringify_bounding_box(bbox) -> str:
    """Turns a polygon nd_array into a string in the right YOLOv8 format."""
    xc, yc, w, h = bbox
    return f"{xc} {yc} {w} {h}"


def mask_filepath_to_yolov8_format_string(filepath: Path) -> str:
    """Given a `filepath` for an individual mask, it returns a yolov8 format
    string describing the polygons for the segmentation tasks."""
    mask = filepath_to_ndarray(filepath)
    if is_only_black_pixels(mask):
        return ""
    else:
        label_class = LABEL_TO_CLASS_MAPPING[mask_label(mask)]
        H, W, _ = mask.shape
        contours = mask_to_contours(mask)
        polygons = contours_to_polygons(contours)
        normalized_polygons = [normalize_polygon(p, W, H) for p in polygons]
        return "\n".join(
            [f"{label_class} {stringify_polygon(p)}" for p in normalized_polygons]
        )


def generate_dataset_individual_mask_labels(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Given a `dataset_name` (see function get_all_dataset_names`, it
    generates the instance segmentation labels in `output_dir`."""
    filepaths = list_all_mask_filepaths(dataset_name, input_dir=input_dir)
    output_filepath_root = output_dir / dataset_name / "labels/individual/"

    if not os.path.exists(output_filepath_root):
        os.makedirs(output_filepath_root)

    for filepath in tqdm(filepaths):
        output_filepath = output_filepath_root / f"{filepath.stem}.txt"
        content = mask_filepath_to_yolov8_format_string(filepath)
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        with open(output_filepath, "x") as f:
            f.write(content)


def _individual_mask_labels_generator(
    input_dir: Path,
    output_dir: Path,
):
    def f(dataset_name: str):
        generate_dataset_individual_mask_labels(
            dataset_name=dataset_name, input_dir=input_dir, output_dir=output_dir
        )

    return f


def prefix_to_mask_filepaths(
    dataset_name: str,
    prefix: str,
    output_dir: Path,
) -> list[Path]:
    path: Path = output_dir / dataset_name / "labels/individual"
    return [Path(path / f) for f in os.listdir(path) if f.startswith(prefix)]


def filepath_to_content(filepath: Path) -> str:
    """Given a path, it returns its content as a string."""
    with open(filepath, "r") as f:
        return f.read()


def stitch_mask_label_yolov8_format_string(
    dataset_name: str,
    mask_stitched_filepath: Path,
    output_dir: Path,
) -> str:
    """Returns the YOLOv8 PyTorch TXT format as a string for the given
    `mask_stitched_filepath` and `dataset_name`."""
    prefix = mask_stitched_filepath.stem
    individual_mask_filepaths = prefix_to_mask_filepaths(
        dataset_name=dataset_name,
        prefix=prefix,
        output_dir=output_dir,
    )
    return "\n".join([filepath_to_content(path) for path in individual_mask_filepaths])


def generate_stitched_masks_labels_for_dataset_name(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_filepath_root = output_dir / dataset_name / "labels" / "stitched"
    if not os.path.exists(output_filepath_root):
        os.makedirs(output_filepath_root)

    path = input_dir / dataset_name / "masks_stitched"
    masks_stitched_filepaths = [Path(path / f) for f in os.listdir(path)]

    for mask_stitched_filepath in tqdm(masks_stitched_filepaths):
        content = stitch_mask_label_yolov8_format_string(
            dataset_name=dataset_name,
            mask_stitched_filepath=mask_stitched_filepath,
            output_dir=output_dir,
        )
        output_filepath = output_filepath_root / f"{mask_stitched_filepath.stem}.txt"
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        with open(output_filepath, "x") as f:
            f.write(content)


def generate_all_stitched_masks_labels(
    input_dir: Path,
    output_dir: Path,
) -> None:
    for dataset_name in tqdm(get_all_dataset_names(input_dir)):
        logging.info(f"Generating stitched mask labels for {dataset_name}")
        generate_stitched_masks_labels_for_dataset_name(
            dataset_name=dataset_name, input_dir=input_dir, output_dir=output_dir
        )


# TODO: fix the dataset generation for other datasets
def generate_image_labels_for_dataset(
    dataset_name: str,
    output_dir: Path,
) -> None:
    output_path_stitched_labels_root = output_dir / dataset_name / "labels" / "stitched"
    output_path_image_labels_root = output_dir / dataset_name / "labels" / "images"

    if not os.path.exists(output_path_image_labels_root):
        os.makedirs(output_path_image_labels_root)

    masks_stitched_filepaths = [
        output_path_stitched_labels_root / f
        for f in os.listdir(output_path_stitched_labels_root)
    ]
    for p in masks_stitched_filepaths:
        image_label_filename = p.name.replace("_mask", "")
        image_label_filepath = output_path_image_labels_root / image_label_filename
        if os.path.exists(image_label_filepath):
            os.remove(image_label_filepath)
        content = filepath_to_content(p)
        with open(image_label_filepath, "x") as f:
            f.write(content)


def generate_image_labels(
    input_dir: Path,
    output_dir: Path,
) -> None:
    for dataset_name in tqdm(get_all_dataset_names(input_dir)):
        logging.info(f"Generating image labels for {dataset_name}")
        generate_image_labels_for_dataset(dataset_name, output_dir=output_dir)


def generate_all_individual_masks_labels(
    input_dir: Path,
    output_dir: Path,
):
    dataset_names = get_all_dataset_names(input_dir)
    func = functools.partial(
        generate_dataset_individual_mask_labels,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    with mp.Pool(mp.cpu_count()) as pool:
        list(
            tqdm(
                pool.imap(
                    func,
                    dataset_names,
                ),
                total=len(dataset_names),
            )
        )


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the processor script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        help="directory containing the raw dataset. Make sure to use data/01_raw.",
        required=True,
    )
    parser.add_argument(
        "--to",
        help="directory to save the processed data. Make sure to use data/04_features.",
        required=True,
    )
    return parser


def is_data_raw_path(dest_path: Path) -> bool:
    """Returns whether `dest_path` contains 01_raw."""
    return "01_raw" in str(dest_path)


def is_data_feature_path(dest_path: Path) -> bool:
    """Returns whether `dest_path` contains 04_feature."""
    return "04_feature" in str(dest_path)


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not os.path.isdir(args["to"]):
        logging.error("invalid --to path -- the directory does not exist")
        return False
    elif not os.path.isdir(args["from"]):
        logging.error("invalid --from path -- the directory does not exist")
        return False
    elif not is_data_raw_path(Path(args["from"])):
        logging.error("invalid --from path - should be the data/01_raw folder")
        return False
    elif not is_data_feature_path(Path(args["to"])):
        logging.error("invalid --to path - should be the data/04_feature folder")
        return False
    else:
        return True


def make(
    input_dir: Path,
    output_dir: Path,
):
    logging.info(f"Reading raw data from {input_dir}")
    logging.info(f"Generate YOLOv8 Pytorch TXT format to {output_dir}")

    logging.info(f"Scaffold output dir: {output_dir}")
    scaffold_output_dir(output_dir=output_dir)
    logging.info("Scaffolding done")

    logging.info("Generate all individual masks labels")
    generate_all_individual_masks_labels(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    logging.info("Generate all stitched masks")
    generate_all_stitched_masks_labels(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    logging.info("Generate all image labels")
    generate_image_labels(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    logging.info("Done")


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    if not validate_parsed_args(args):
        exit(1)
    else:
        suffix = "benthic_datasets/mask_labels/rs_labelled"
        input_dir = Path(args["from"]) / "rs_storage_open" / suffix
        output_dir = Path(args["to"]) / "yolov8" / suffix
        logging.info(f"input_dir: {input_dir}")
        logging.info(f"output_dir: {output_dir}")
        make(input_dir=input_dir, output_dir=output_dir)
        exit(0)
