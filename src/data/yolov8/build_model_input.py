"""Model Input builder.

This script allows the user to generate datasets that are ready for
YOLOv8 training. A train/eval split function is provided. For each
generated dataset, a config.yaml file is created and contains all
parameters that were used to generate it.
"""
import argparse
import itertools
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

LABEL_TO_CLASS_MAPPING = {"soft_coral": 0, "hard_coral": 1}
CLASS_TO_LABEL_MAPPING = {v: k for k, v in LABEL_TO_CLASS_MAPPING.items()}

# For type hints
Quadratid = int
Contour = np.ndarray
Mask = np.ndarray
Polygon = np.ndarray
Entry = dict


def rm_r(path: Path) -> None:
    """Equivalent to the bash command `rm -r $path`.

    Warning: Make sure you know which folder you are clearing before running it.
    The erased files won't go to the Trash folder.
    """

    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def write_config_yaml(
    path: Path,
    X_train,
    X_val,
    dataset_names: list[str],
    seed: int,
    train_size_ratio: float,
) -> None:
    """Writes the `config.yaml` file that describes the generated dataset."""

    def entries_to_dict(entries):
        result = defaultdict(list)
        for entry in entries:
            result[entry["dataset_name"]].append(entry["image_filepath"].name)
        return dict(result)

    data = {
        "dataset_names": dataset_names,
        "seed": seed,
        "train_size_ratio": train_size_ratio,
        "train_dataset_size": len(X_train),
        "val_dataset_size": len(X_val),
        "train_dataset": entries_to_dict(X_train),
        "val_dataset": entries_to_dict(X_val),
    }

    with open(path / "config.yaml", "x") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def slurp(filepath: Path) -> str:
    with open(filepath, "r") as f:
        return f.read()


def write_data_yaml(path: Path) -> None:
    """Writes the `data.yaml` file necessary for YOLOv8 training at `path`
    location."""
    data = {
        "train": "./train/images",
        "val": "./val/images",
        "nc": 2,
        # hard and soft corals where incorrectly read using cv2 when builting
        # yolov8_pytorch_txt_format, we need to account for this
        "names": list(reversed([CLASS_TO_LABEL_MAPPING[i] for i in range(2)])),
    }
    with open(path / "data.yaml", "x") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def write_readme(path: Path) -> None:
    """Writes the README.md file of the dataset that describes how to train a
    YOLOv8 model on it."""
    content = [
        "# README",
        "",
        "## Basic training",
        "",
        "To train a yolo model on this dataset, follow the steps:",
        "1. Install ultralytics in a virtualenv:",
        "> pip install ultralytics",
        (
            "2. open data.yaml and edit `train` and `val` value to indicate an absolute"
            " path (eg. /home/user/fruitpunch/datasets/train/images)"
        ),
        (
            "3. run the following basic command to train yolo for object detection for"
            " 1 epoch on the dataset:"
        ),
        "> yolo train data=./data.yaml model=yolov8n.pt epochs=1",
        (
            "4. run the following basic command to train yolo for instance segmentation"
            " for 1 epoch on the dataset:"
        ),
        "> yolo train data=./data.yaml model=yolov8n-seg.pt epochs=1",
        "",
        "## More advanced training",
        "",
        "One can use different model sizes for yolo (n, s, m, l, x):",
        "Eg. Train for 10 epochs the `m` size yolo model for instance segmentation:",
        "> yolo train data=./data.yaml model=yolov8m-seg.pt epochs=10",
        "Eg. Train for 10 epochs the `x` size yolo model for object detection:",
        "> yolo train data=./data.yaml model=yolov8x.pt epochs=10",
    ]
    with open(path / "README.md", "x") as f:
        f.write("\n".join(content))


def init_yolov8_dataset_folder_structure(output_dir: Path, clear: bool = True) -> None:
    """Creates the right yolov8 dataset empty folder structure."""
    if clear:
        logging.info(f"clearing folder {output_dir}")
        rm_r(output_dir)

    dirs = [
        output_dir / "train/images/",
        output_dir / "train/labels/",
        output_dir / "val/images/",
        output_dir / "val/labels/",
    ]

    for dir in dirs:
        if not os.path.isdir(dir):
            logging.info(f"Making directory: {dir}")
            os.makedirs(dir)

    logging.info("Writing data.yaml file")
    write_data_yaml(output_dir)
    logging.info("Writing README.md file")
    write_readme(output_dir)


def list_image_filepaths(
    dataset_name: str,
    input_dir: Path,
) -> list[Path]:
    """Returns a list of paths that are the list of all image names for a given
    `dataset_name`."""
    path = input_dir / dataset_name / "images"
    return [path / f for f in os.listdir(path) if os.path.isfile(path / f)]


def is_label_mismatch(
    dataset_name: str, invalid_seaview_quadratids: set[Quadratid], filepath: Path
) -> bool:
    """Returns whether the `filepath` has a label mismatch."""
    if not dataset_name.startswith("SEAVIEW"):
        return False
    elif int(filepath.stem) in invalid_seaview_quadratids:
        return True
    else:
        return False


def is_only_black_pixels(mask: Mask) -> bool:
    """Returns True if the mask image is only black pixels."""
    non_black_pixels = np.any(mask != [0, 0, 0], axis=-1)
    black_pixels = ~non_black_pixels
    return black_pixels.all()


def get_invalid_seaview_quadratids(csv_data_path: Path) -> set[Quadratid]:
    """Returns a set of quadratids from the seaview folders that contain label
    mismatches.

    Note:
    ReefSupport suggested to discared the following datapoints:
    - For Seaview, discard images with a mismatch of maximum 10 points
      (20% if 50 annotation points or 10% if 100 annotation points)
    - Seaflower and Tetes labelling results are best
    """
    df = pd.read_csv(csv_data_path)
    df_mismatch_labels = df[
        df["folder"].str.startswith("SEAVIEW") & (df["points_mismatch_count"] >= 10)
    ]
    return set(df_mismatch_labels["quadratid"])


def is_empty_label(label_filepath: Path) -> bool:
    """Returns true if the label file is empty (== black mask)"""
    return (
        (not os.path.isfile(label_filepath))
        or slurp(label_filepath) is None
        or slurp(label_filepath) == ""
    )


def image_filepath_to_label_filepath(
    dataset_name: str,
    image_filepath: Path,
    yolov8_pytorch_txt_format_root_dir: Path,
) -> Path:
    label_filename = f"{image_filepath.stem}.txt"
    label_filepath = (
        yolov8_pytorch_txt_format_root_dir
        / dataset_name
        / "labels"
        / "images"
        / label_filename
    )
    return label_filepath


def get_image_filepaths_with_empty_masks(
    yolov8_pytorch_txt_format_root_dir: Path,
    rs_labelled_root_dir: Path,
    dataset_names: list[str],
) -> set[Path]:
    """Returns a list of image filepaths that have empty masks (= empty
    labels)."""
    image_filepaths_with_empty_label = set()
    for dataset_name in dataset_names:
        logging.info(f"Looking for empty masks in {dataset_name}")
        all_image_filepaths = list_image_filepaths(
            dataset_name=dataset_name, input_dir=rs_labelled_root_dir
        )
        empty_labels = [
            image_filepath
            for image_filepath in all_image_filepaths
            if is_empty_label(
                image_filepath_to_label_filepath(
                    dataset_name,
                    image_filepath,
                    yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
                )
            )
        ]
        logging.info(f"    Found {len(empty_labels)} empty label files")
        if len(empty_labels) > 0:
            image_filepaths_with_empty_label = image_filepaths_with_empty_label.union(
                empty_labels
            )
    return image_filepaths_with_empty_label


def get_X(
    dataset_names: list[str],
    invalid_seaview_quadratids: set[Quadratid],
    yolov8_pytorch_txt_format_root_dir: Path,
    rs_labelled_root_dir: Path,
    invalid_image_filepaths: set[Path] = set(),
) -> list[Entry]:
    """Returns a list of {dataset_name, image_filepath, label_filepath} that
    constitues the X dataset.

    Excludes the datapoints that contain data label mismatch.
    """
    X = []
    for dataset_name in dataset_names:
        all_image_filepaths = list_image_filepaths(
            dataset_name=dataset_name, input_dir=rs_labelled_root_dir
        )
        image_filepaths = [
            p
            for p in all_image_filepaths
            # Remove filepaths that are known to have label mismatches
            if (not is_label_mismatch(dataset_name, invalid_seaview_quadratids, p))
            # Remove filepaths that are invalid (empty masks for instance)
            and (p not in invalid_image_filepaths)
        ]

        if len(all_image_filepaths) > len(image_filepaths):
            logging.info(
                f"Excluding {len(all_image_filepaths) - len(image_filepaths)} files"
                f" from {dataset_name} because of label mismatch or empty masks."
            )

        for image_filepath in image_filepaths:
            label_filename = f"{image_filepath.stem}.txt"
            label_filepath = (
                yolov8_pytorch_txt_format_root_dir
                / dataset_name
                / "labels"
                / "images"
                / label_filename
            )
            entry = {
                "dataset_name": dataset_name,
                "image_filepath": image_filepath,
                "label_filepath": label_filepath,
            }
            X.append(entry)
    return X


def split_train_val(
    X: list[Entry],
    train_size_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Entry], list[Entry]]:
    """Returns a splitted dataset X into X_train and X_val using the
    `train_size_ratio` and the random `seed`."""
    N = len(X)
    random.seed(seed)
    random.shuffle(X)
    split_index = int(N * train_size_ratio)

    X_train, X_val = X[:split_index], X[split_index:]
    return X_train, X_val


def split_train_val2(
    X: list[Entry],
    train_size_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Entry], list[Entry]]:
    """Returns a splitted dataset X into X_train and X_val using the
    `train_size_ratio` and the random `seed`.

    If X is a subset of Y in terms of dataset_names, they will have the
    same split at the dataset_name level.
    """
    X_train, X_val = [], []

    for _, group in itertools.groupby(X, lambda e: e["dataset_name"]):
        xs = list(group)
        K = len(xs)
        shuffled = random.Random(seed).sample(xs, K)
        split_index = int(K * train_size_ratio)
        X_train.extend(shuffled[:split_index])
        X_val.extend(shuffled[split_index:])

    return X_train, X_val


def write_entry(
    entry,
    output_dir: Path,
    mode: str = "train",
) -> None:
    """Given an `entry` and a mode in #{`train`, `val`}, it writes it in a
    YOLOv8 format."""
    source_image_filepath = entry["image_filepath"]
    source_label_filepath = entry["label_filepath"]
    destination_image_filepath = (
        output_dir / mode / "images" / source_image_filepath.name
    )
    destination_label_filepath = (
        output_dir / mode / "labels" / source_label_filepath.name
    )

    assert os.path.exists(
        source_image_filepath
    ), f"should exist {source_image_filepath}"
    assert os.path.exists(
        source_label_filepath
    ), f"should exist {source_label_filepath}"
    assert os.path.exists(
        output_dir / mode / "images"
    ), f"the images folder should exist in {output_dir}"
    assert os.path.exists(
        output_dir / mode / "labels"
    ), f"the labels folder should exist  in {output_dir}"

    shutil.copyfile(source_image_filepath, destination_image_filepath)
    shutil.copyfile(source_label_filepath, destination_label_filepath)


def write_dataset(
    X_train: list[Entry],
    X_val: list[Entry],
    output_dir: Path,
) -> None:
    """Writes the dataset splitted in X_train and X_val into the right folder
    structure for YOLOv8."""
    logging.info(f"Generating train set - {len(X_train)} datapoints")
    for entry in X_train:
        write_entry(entry, output_dir=output_dir, mode="train")

    logging.info(f"Generating val set - {len(X_val)} datapoints")
    for entry in X_val:
        write_entry(entry, output_dir=output_dir, mode="val")


def generate(
    split_train_val: Callable,
    dataset_names: list[str],
    yolov8_pytorch_txt_format_root_dir: Path,
    rs_labelled_root_dir: Path,
    csv_data_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_size_ratio: float = 0.8,
) -> None:
    """Main function to generate the full dataset ready for YOLOv8 to be
    trained on."""
    init_yolov8_dataset_folder_structure(output_dir=output_dir)
    logging.info(
        "Splitting datapoints between train and val sets for the datasets:"
        f" {' '.join(dataset_names)}"
    )
    invalid_seaview_quadratids = get_invalid_seaview_quadratids(
        csv_data_path=csv_data_path
    )
    logging.info(f"Found {len(invalid_seaview_quadratids)} mislabelled quadratid")
    invalid_image_filepaths = get_image_filepaths_with_empty_masks(
        rs_labelled_root_dir=rs_labelled_root_dir,
        yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
        dataset_names=dataset_names,
    )
    logging.info(f"Found {len(invalid_image_filepaths)} empty masks")
    X = get_X(
        dataset_names,
        invalid_seaview_quadratids,
        invalid_image_filepaths=invalid_image_filepaths,
        rs_labelled_root_dir=rs_labelled_root_dir,
        yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
    )
    X_train, X_val = split_train_val(X, train_size_ratio=train_size_ratio, seed=seed)
    logging.info(f"Writing the data in {output_dir}")
    write_dataset(X_train, X_val, output_dir=output_dir)
    logging.info("Writing config.yaml file")
    write_config_yaml(
        path=output_dir,
        X_train=X_train,
        X_val=X_val,
        dataset_names=dataset_names,
        seed=seed,
        train_size_ratio=train_size_ratio,
    )


def make_archive(
    output_dir: Path,
    archive_name: str = "archive",
) -> None:
    """Generates an archive file from the `output_dir`"""
    shutil.make_archive(str(output_dir.parent / archive_name), "zip", output_dir)


def get_name_from_dataset_names(dataset_names: list[str]) -> str:
    return f"{'_and_'.join(dataset_names)}"


Splitter = dict


def build(
    output_dir: Path,
    dataset_names: list[str],
    splitters: list[Splitter],
    rs_labelled_root_dir: Path,
    csv_data_path: Path,
    yolov8_pytorch_txt_format_root_dir: Path,
    train_size_ratio: float = 0.8,
    random_seed: int = 42,
    archive: bool = True,
) -> None:
    suffix = get_name_from_dataset_names(dataset_names)
    # dir_v1 = output_dir / "v1" / suffix
    # dir_v2 = output_dir / "v2" / suffix

    for splitter in splitters:
        version = splitter["version"]
        dir = output_dir / version / suffix
        generate(
            split_train_val=splitter["split_train_val_fn"],
            dataset_names=dataset_names,
            rs_labelled_root_dir=rs_labelled_root_dir,
            csv_data_path=csv_data_path,
            seed=random_seed,
            train_size_ratio=train_size_ratio,
            output_dir=dir,
            yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
        )

        if archive:
            archive_name = f"archive_{suffix}"
            logging.info(f"Making archive {archive_name}")
            make_archive(output_dir=dir, archive_name=archive_name)


def build_all(
    output_dir: Path,
    rs_labelled_root_dir: Path,
    csv_data_path: Path,
    yolov8_pytorch_txt_format_root_dir: Path,
    train_size_ratio: float = 0.8,
    random_seed: int = 42,
    archive: bool = True,
) -> None:
    """Builds a sequence of training sets.

    This is in the format and structure expected by YOLOv8 to train on.
    """
    all_dataset_names = [
        "SEAFLOWER_BOLIVAR",
        "SEAFLOWER_COURTOWN",
        "SEAVIEW_ATL",
        "SEAVIEW_IDN_PHL",
        "SEAVIEW_PAC_AUS",
        "SEAVIEW_PAC_USA",
        "TETES_PROVIDENCIA",
    ]

    # Splitters indicate how we want to train/val split the data and associate a version to it
    splitters = [
        {"version": "v1", "split_train_val_fn": split_train_val},
        {"version": "v2", "split_train_val_fn": split_train_val2},
    ]

    for dataset_name in tqdm(all_dataset_names):
        logging.info(f"Generating dataset for {dataset_name}")
        build(
            dataset_names=[dataset_name],
            rs_labelled_root_dir=rs_labelled_root_dir,
            splitters=splitters,
            csv_data_path=csv_data_path,
            random_seed=random_seed,
            train_size_ratio=train_size_ratio,
            output_dir=output_dir,
            archive=archive,
            yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
        )

    all_dataset_names_but_seaview_pac_usa = [
        "SEAFLOWER_BOLIVAR",
        "SEAFLOWER_COURTOWN",
        "SEAVIEW_ATL",
        "SEAVIEW_IDN_PHL",
        "SEAVIEW_PAC_AUS",
        "TETES_PROVIDENCIA",
    ]
    logging.info("Generating combined dataset")
    build(
        dataset_names=all_dataset_names_but_seaview_pac_usa,
        rs_labelled_root_dir=rs_labelled_root_dir,
        splitters=splitters,
        csv_data_path=csv_data_path,
        random_seed=random_seed,
        train_size_ratio=train_size_ratio,
        output_dir=output_dir,
        archive=archive,
        yolov8_pytorch_txt_format_root_dir=yolov8_pytorch_txt_format_root_dir,
    )
    logging.info("Done")


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        help="directory to save the model input dataset. Make sure to use data/05_model_input.",
        required=True,
    )
    parser.add_argument(
        "--yolov8-pytorch-txt-format-root",
        help="directory where the YOLOv8 Pytorch TXT format files are stored. Should be located in data/04_feature.",
        required=True,
    )
    parser.add_argument(
        "--raw-root-rs-labelled",
        help="directory that contains the masks rs labelled datapoints. Should be located in data/01_raw.",
        required=True,
    )
    parser.add_argument(
        "--csv-label-mismatch-file",
        help="path to the csv file that contains the mismatch label info. Should be located in data/04_feature.",
        required=True,
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="generate an archive file for each generated training set?",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


# TODO
def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(f"build model inputs with args {args}")
        build_all(
            output_dir=Path(args["to"]),
            rs_labelled_root_dir=Path(args["raw_root_rs_labelled"]),
            csv_data_path=Path(args["csv_label_mismatch_file"]),
            yolov8_pytorch_txt_format_root_dir=Path(
                args["yolov8_pytorch_txt_format_root"]
            ),
            archive=args["archive"],
        )
        exit(0)
