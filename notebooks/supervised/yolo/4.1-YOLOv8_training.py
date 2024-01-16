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

# %% [markdown] id="yYlmnjXFQhbf"
# # YOLOv8 Training Notebook
#
# In this Notebook, we setup the training pipeline for the YOLOv8 model.

# %% [markdown] id="bPxf4m1QMTtV"
# ## Setup

# %% [markdown] id="N0gyY-jVMGdn"
# ### Dependencies

# %% colab={"base_uri": "https://localhost:8080/"} id="UsJqM-llTIGV" outputId="8cb2e897-84bb-44a5-89db-0c827c83d306"
# !pip install ultralytics

# %% [markdown] id="Ji0TSdrPMLZf"
# ### Imports

# %% id="kBXf7ainDxT3"
import os
import shutil
from pathlib import Path

import yaml
from google.colab import drive

# %% [markdown] id="Z0K9kshQMN8i"
# ### Utils


# %% id="WJUgC2s6GRXU"
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def yaml_content(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    """Writes yaml `data` (as a dict) to file `path` using the MyDumper
    class."""
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def absolute_train_val_paths(extract_folder_path: Path, data: dict) -> dict:
    """Make sure the paths in data.yaml are absolute and pointing to the right
    images and labels."""
    result = dict(data)
    result["train"] = str(extract_folder_path / "train" / "images")
    result["val"] = str(extract_folder_path / "val" / "images")
    return result


def swap_coral_class_order(extract_folder_path: Path, data: dict) -> dict:
    """
    Note: an error was made using opencv when making the dataset.
    blue and red classes got inverted and this function fixes the class string
    labels.
    """
    result = dict(data)
    result["names"][0], result["names"][1] = result["names"][1], result["names"][0]
    return result


def archive_path_to_extract_folder_path(
    path_archive: Path, output_dir: str = "/content/datasets_ready_for_yolov8_training"
) -> Path:
    extract_folder_name = path_archive.name.split(".")[0].replace("archive_", "")
    return Path(output_dir) / extract_folder_name


def extract_archive(path_archive: Path) -> dict:
    # Extract the path_archive
    extract_folder_path = archive_path_to_extract_folder_path(path_archive)
    os.makedirs(extract_folder_path, exist_ok=True)
    shutil.unpack_archive(path_archive, extract_folder_path)
    print(f"archive {path_archive} extracted in {extract_folder_path}")

    # Update the data_yaml file to point to the right files and labels
    path_data_yaml: Path = extract_folder_path / "data.yaml"
    data_yaml: dict = yaml_content(path_data_yaml)
    # new_data_yaml: dict = absolute_train_val_paths(extract_folder_path, data_yaml)
    new_data_yaml: dict = swap_coral_class_order(
        extract_folder_path, absolute_train_val_paths(extract_folder_path, data_yaml)
    )
    write_yaml(path_data_yaml, new_data_yaml)
    print(f"updating absolute paths in data.yaml content {new_data_yaml}")

    return {
        "extract_folder_path": extract_folder_path,
        "new_data_yaml": new_data_yaml,
    }


# %% [markdown] id="KR3WQ0G7MXkU"
# ## Training YOLOv8

# %% [markdown] id="sepR698GMcfc"
# ### Getting the dataset ready

# %% [markdown] id="A_YOOYwRMkro"
# We first need to mount GDrive and extract the archive file in the temporary directory.
# By default, the archive is extracted in `/content/datasets_ready_for_yolov8_training`.
# One needs to update the `GDRIVE_ARCHIVE_ROOT_DIR` variable that points to the root of all the archive files and the 'ARCHIVE_NAME` that contains the filename of the archive.

# %% id="CVfdrPDHXnzP" colab={"base_uri": "https://localhost:8080/"} outputId="fda8f318-9777-4971-ea73-3558da470540"
drive.mount("/content/drive")

# %% id="yYwqXR4dNChR"
GDRIVE_ARCHIVE_ROOT_DIR = (
    "/content/drive/MyDrive/fruitpunchai/coralreefs/datasets_ready_for_yolov8_training/"
)
# ARCHIVE_NAME = 'archive_SEAFLOWER_BOLIVAR.zip'
ARCHIVE_NAME = "archive_SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA.zip"

# %% colab={"base_uri": "https://localhost:8080/"} id="qrzUGsfVNjSL" outputId="61384c87-8c10-4abb-cd91-bdcb74939cfe"
# List all available archive names
# Update `ARCHIVE_NAME` with the one you want to use
[f for f in os.listdir(GDRIVE_ARCHIVE_ROOT_DIR) if f.endswith(".zip")]

# %% colab={"base_uri": "https://localhost:8080/"} id="RytSYhqEERPP" outputId="d2a71b97-1e09-4874-f95b-7c33cf57d76d"
# Archive extraction
path_archive = Path(GDRIVE_ARCHIVE_ROOT_DIR) / ARCHIVE_NAME
archive_result = extract_archive(path_archive)
archive_result

# %% [markdown] id="wSVzUeBeMf_u"
# ### Training

# %% [markdown] id="Q3I8Kg8bY0o0"
# To establish our baseline models, we picked the following parameters:
#
# ```python
# MODEL_SIZE: str = 'm'
# EPOCHS: int = 20
# CV_TASK: str = 'segmentation'
# ```

# %% id="UakEdFCjPB2T"
# Choose the training parameters
MODEL_SIZE: str = "m"  # Can be n, s, m, l, x
EPOCHS: int = 20  # Positive integer
CV_TASK: str = "segmentation"  # `segmentation` or `object_detection`


# TODO: add others like learning_rate, Optimizer, etc.

# %% id="70XtYqzSPz6V"
# Derived parameters from the above cells
yolo_model = f'yolov8{MODEL_SIZE}{"-seg" if CV_TASK == "segmentation" else ""}.pt'
yolo_data_yaml_path = str(archive_result["extract_folder_path"] / "data.yaml")

# %% colab={"base_uri": "https://localhost:8080/"} id="qTU4_Xp87p1f" outputId="ea6af123-1ba0-4ab0-e9aa-211832919f19"
# !yolo train data=$yolo_data_yaml_path model=$yolo_model epochs=$EPOCHS

# %% id="F-WsCa4ETYOY"
shutil.make_archive(
    # '/content/SEAFLOWER_BOLIVAR_baseline_yolov8_session_runs',
    "/content/ALL_REGIONS_baseline_yolov8_session_runs",
    "zip",
    "/content/runs",
)

# %% id="7xVx46lzpATe"
