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
# # Dataset Preparation for YOLOv8 models
#
# In this notebook, some tools are provided to generate datasets directly usable by YOLOv8.

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
from collections import defaultdict
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
import os
import pandas as pd
import random
import shutil
import yaml

# %% [markdown]
# ### Global variables
#
# Adjust the `INPUT_DIR_YOLOV8_INSTANCE_SEGMENTATION_LABELS`, `OUTPUT_DIR_YOLOV8_SEGMENTATION`, `INPUT_LABEL_MISMATCH_CSV_DATA` and `INPUT_DIR_DATASET_ROOT_RS_LABELLED` path to match your filesystem setup.

# %%
# IMPORTANT: Modify these four paths to point to your own data
INPUT_DIR_DATASET_ROOT_RS_LABELLED = Path('/home/chouffe/playground/datasets/benthic_datasets/mask_labels/rs_labelled')
INPUT_DIR_YOLOV8_INSTANCE_SEGMENTATION_LABEL = Path('/home/chouffe/playground/datasets/yolov8/benthic_datasets/instance_segmentation')
INPUT_LABEL_MISMATCH_CSV_DATA = Path('/home/chouffe/fruitpunch/challenges/coralreefs2/datasets/benthic_datasets/label_mismatch/data.csv')
OUTPUT_DIR_YOLOV8_SEGMENTATION = Path('/home/chouffe/playground/datasets/yolov8_ready/benthic_datasets/instance_segmentation/')

ALL_DATASETS = [
    'SEAFLOWER_BOLIVAR',
    'SEAFLOWER_COURTOWN',
    'SEAVIEW_ATL',
    'SEAVIEW_IDN_PHL',
    'SEAVIEW_PAC_AUS',
    'SEAVIEW_PAC_USA',
    'TETES_PROVIDENCIA',
]

LABEL_TO_CLASS_MAPPING = {
    'soft_coral': 0,
    'hard_coral': 1
    }
CLASS_TO_LABEL_MAPPING = {v: k for k, v in LABEL_TO_CLASS_MAPPING.items()}

# For type hints
Quadratid = int
Contour = np.ndarray
Mask = np.ndarray
Polygon = np.ndarray
Entry = list[dict]

# %% [markdown]
# ### Check folder structures

# %% [markdown]
# Make sure that the following command returns something like this:
#
# ```
# ├── SEAFLOWER_BOLIVAR
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# ├── SEAFLOWER_COURTOWN
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# ├── SEAVIEW_ATL
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# ├── SEAVIEW_IDN_PHL
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# ├── SEAVIEW_PAC_AUS
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# ├── SEAVIEW_PAC_USA
# │   └── labels
# │       ├── images
# │       ├── individual
# │       └── stitched
# └── TETES_PROVIDENCIA
#     └── labels
#         ├── images
#         ├── individual
#         └── stitched
# ```

# %%
# !tree -d -L 3 $INPUT_DIR_YOLOV8_INSTANCE_SEGMENTATION_LABEL

# %% [markdown]
# Make sure that the following command returns something like this:
#
# ```
# ├── SEAFLOWER_BOLIVAR
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# ├── SEAFLOWER_COURTOWN
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# ├── SEAVIEW_ATL
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# ├── SEAVIEW_IDN_PHL
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# ├── SEAVIEW_PAC_AUS
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# ├── SEAVIEW_PAC_USA
# │   ├── images
# │   ├── masks
# │   └── masks_stitched
# └── TETES_PROVIDENCIA
#     ├── images
#     ├── masks
#     └── masks_stitched
# ```

# %%
# !tree -d -L 3 $INPUT_DIR_DATASET_ROOT_RS_LABELLED

# %%
def rm_r(path: Path) -> None:
    """
    Equivalent to the bash command `rm -r $path`.
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


def write_config_yaml(path: Path, X_train, X_val, dataset_names: list[str], seed: int, train_size_ratio: float) -> None:
    """
    Writes the `config.yaml` file that describes the generated dataset.
    """

    def entries_to_dict(entries):
        result = defaultdict(list)
        for entry in entries:
            result[entry['dataset_name']].append(entry['image_filepath'].name)
        return dict(result)
        
    data = {
        'dataset_names': dataset_names,
        'seed': seed,
        'train_size_ratio': train_size_ratio,
        'train_dataset_size': len(X_train),
        'val_dataset_size': len(X_val),
        'train_dataset': entries_to_dict(X_train),
        'val_dataset': entries_to_dict(X_val),
    }
    
    with open(path / 'config.yaml', 'x') as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)
    

def write_data_yaml(path: Path) -> None:
    """
    Writes the `data.yaml` file necessary for YOLOv8 training at `path` location.
    """
    data = {
        'train': './train/images',
        'val':  './val/images',
        'nc': 2,
        'names': [CLASS_TO_LABEL_MAPPING[i] for i in range(2)],
    }
    with open(path / 'data.yaml', 'x') as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def write_readme(path: Path) -> None:
    """
    Writes the README.md file of the dataset that describes how to train a 
    YOLOv8 model on it.
    """
    content = [
        '# README',
        '',
        '## Basic training',
        'To train a yolo model on this dataset, follow the steps:',
        '1. Install ultralytics in a virtualenv:',
        '> pip install ultralytics',
        '2. open data.yaml and edit `train` and `val` value to indicate an absolute path (eg. /home/user/fruitpunch/datasets/train/images)',
        '3. run the following basic command to train yolo for object detection for 1 epoch on the dataset:',
        '> yolo train data=./data.yaml model=yolov8n.pt epochs=1',
        '4. run the following basic command to train yolo for instance segmentation for 1 epoch on the dataset:',
        '> yolo train data=./data.yaml model=yolov8n-seg.pt epochs=1',
        '',
        '## More advanced training',
        'One can use different model sizes for yolo (n, s, m, l, x):'
        'Eg. Train for 10 epochs the `m` size yolo model for instance segmentation:',
        '> yolo train data=./data.yaml model=yolov8m-seg.pt epochs=10',
         'Eg. Train for 10 epochs the `x` size yolo model for object detection:',
        '> yolo train data=./data.yaml model=yolov8x.pt epochs=10'
        
    ]
    with open(path / 'README.md', 'x') as f:
        f.write('\n'.join(content))
    

def init_yolov8_dataset_folder_structure(output_dir: Path = OUTPUT_DIR_YOLOV8_SEGMENTATION, clear: bool = True) -> None:
  """
  Creates the right yolov8 dataset empty folder structure.
  """
  if clear:
    print(f'clearing folder {output_dir}')
    rm_r(output_dir)

  dirs = [
      output_dir / 'train/images/',
      output_dir / 'train/labels/',
      output_dir / 'val/images/',
      output_dir / 'val/labels/',
      ]

  for dir in dirs:
    if not os.path.isdir(dir):
      print(f'Making directory: {dir}')
      os.makedirs(dir)

  print('Writing data.yaml file')
  write_data_yaml(output_dir)
  print('Writing README.md file')
  write_readme(output_dir)


# %%
def list_image_filepaths(dataset_name: str, input_dir: Path = INPUT_DIR_DATASET_ROOT_RS_LABELLED) -> list[Path]:
    """
    Returns a list of paths that are the list of all image names for a given
    `dataset_name`.
    """
    path = input_dir / dataset_name / 'images'
    return [path / f for f in os.listdir(path) if os.path.isfile(path / f)]


def is_label_mismatch(dataset_name: str, invalid_seaview_quadratids: set[Quadratid], filepath: Path) -> bool:
    """
    Returns whether the `filepath` has a label mismatch.
    """
    if not dataset_name.startswith('SEAVIEW'):
        return False
    elif int(filepath.stem) in invalid_seaview_quadratids:
        return True
    else:
        return False


def get_invalid_seaview_quadratids(csv_data_path: Path = INPUT_LABEL_MISMATCH_CSV_DATA) -> set[Quadratid]:
    """
    Returns a set of quadratids from the seaview folders that contain label mismatches.

    Note:
    ReefSupport suggested to discared the following datapoints:
    - For Seaview, discard images with a mismatch of maximum 10 points 
      (20% if 50 annotation points or 10% if 100 annotation points)
    - Seaflower and Tetes labelling results are best
    """
    df = pd.read_csv(csv_data_path)
    df_mismatch_labels = df[df['folder'].str.startswith('SEAVIEW') & (df['points_mismatch_count'] >= 10)]
    return set(df_mismatch_labels['quadratid'])


def get_X(dataset_names: list[str], invalid_seaview_quadratids: set[Quadratid]) -> list[Entry]:
    """
    Returns a list of {dataset_name, image_filepath, label_filepath} that constitues the X dataset.
    Excludes the datapoints that contain data label mismatch.
    """
    X = []
    for dataset_name in dataset_names:
        all_image_filepaths = list_image_filepaths(dataset_name)
        image_filepaths = [p for p in all_image_filepaths if not is_label_mismatch(dataset_name, invalid_seaview_quadratids, p)]

        if len(all_image_filepaths) > len(image_filepaths):
            print(f'Excluding {len(all_image_filepaths) - len(image_filepaths)} files from {dataset_name} because of label mismatch')

        for image_filepath in image_filepaths:
            label_filename =  f'{image_filepath.stem}.txt'
            label_filepath = INPUT_DIR_YOLOV8_INSTANCE_SEGMENTATION_LABEL / dataset_name / 'labels' / 'images' / label_filename
            entry = {
                'dataset_name': dataset_name, 
                'image_filepath': image_filepath,
                'label_filepath': label_filepath,
                }
            X.append(entry)
    return X


def split_train_val(X: list[Entry], train_size_ratio: float = 0.8, seed: int = 42) -> tuple[list[Entry], list[Entry]]:
    """
    Returns a splitted dataset X into X_train and X_val using the
    `train_size_ratio` and the random `seed`.
    """
    N = len(X)
    random.seed(seed)
    random.shuffle(X)
    split_index = int(N * train_size_ratio)
    
    X_train, X_val = X[:split_index], X[split_index:]
    return X_train, X_val


def write_entry(entry, mode: str = 'train', output_dir: Path = OUTPUT_DIR_YOLOV8_SEGMENTATION) -> None:
    """
    Given an `entry` and a mode in #{`train`, `val`}, it writes it in a YOLOv8 format.
    """
    source_image_filepath = entry['image_filepath']
    source_label_filepath = entry['label_filepath']
    destination_image_filepath = output_dir / mode / 'images' / source_image_filepath.name
    destination_label_filepath = output_dir / mode / 'labels' / source_label_filepath.name

    assert os.path.exists(source_image_filepath), f"should exist {source_image_filepath}"
    assert os.path.exists(source_label_filepath), f"should exist {source_label_filepath}"
    assert os.path.exists(output_dir / mode / 'images'), f"the images folder should exist in {output_dir}"
    assert os.path.exists(output_dir / mode / 'labels'), f"the labels folder should exist  in {output_dir}"

    shutil.copyfile(source_image_filepath, destination_image_filepath)
    shutil.copyfile(source_label_filepath, destination_label_filepath)


def write_dataset(X_train: list[Entry], X_val: list[Entry], output_dir: Path = OUTPUT_DIR_YOLOV8_SEGMENTATION) -> None:
    """
    Writes the dataset splitted in X_train and X_val into the right folder structure for 
    YOLOv8.
    """
    print(f"Generating train set - {len(X_train)} datapoints")
    for entry in X_train:
        write_entry(entry, mode='train', output_dir=output_dir)
        
    print(f"Generating val set - {len(X_val)} datapoints")
    for entry in X_val:
        write_entry(entry, mode='val', output_dir=output_dir)


def generate(
    dataset_names: list[str], 
    seed: int = 42, 
    train_size_ratio: float = 0.8, 
    output_dir: Path = OUTPUT_DIR_YOLOV8_SEGMENTATION) -> None:
    """
    Main function to generate the full dataset ready for YOLOv8 to be trained on.
    """
    init_yolov8_dataset_folder_structure(output_dir=output_dir)
    print(f'Splitting datapoints between train and val sets for the datasets: {" ".join(dataset_names)}')
    invalid_seaview_quadratids = get_invalid_seaview_quadratids(csv_data_path=INPUT_LABEL_MISMATCH_CSV_DATA)
    X = get_X(dataset_names, invalid_seaview_quadratids)
    X_train, X_val = split_train_val(X, train_size_ratio=train_size_ratio, seed=seed)
    print(f'Writing the data in {output_dir}')
    write_dataset(X_train, X_val, output_dir=output_dir)
    print('Writing config.yaml file')
    write_config_yaml(path=output_dir, X_train=X_train, X_val=X_val, dataset_names=dataset_names, seed=seed, train_size_ratio=train_size_ratio)


# %% [markdown]
# ## Generation

# %% [markdown]
# ### Script

# %%
# Add the dataset names in that list to inlude them in the generated set
dataset_names = [
    'SEAFLOWER_BOLIVAR',
    # 'SEAFLOWER_COURTOWN',
    # 'SEAVIEW_ATL',
    # 'SEAVIEW_IDN_PHL',
    # 'SEAVIEW_PAC_AUS',
    # 'SEAVIEW_PAC_USA',
    # 'TETES_PROVIDENCIA',
]

# Parameters to generate the dataset
seed = 42
train_size_ratio = .80
# This will generate all the folder structure in `OUTPUT_DIR_YOLOV8_SEGMENTATION` for yolov8 to consume
generate(dataset_names, seed=seed, train_size_ratio=train_size_ratio, output_dir=OUTPUT_DIR_YOLOV8_SEGMENTATION)


# %% [markdown]
# ### Sanity check

# %% [markdown]
# Below is what it should look like on your filesystem.
#
# ```
# ├── config.yaml
# ├── data.yaml
# ├── README.md
# ├── train
# │   ├── images
# │   └── labels
# └── val
#     ├── images
#     └── labels
# ```

# %%
# !tree -L 2 $OUTPUT_DIR_YOLOV8_SEGMENTATION

# %% [markdown]
# ### Export
#
# Export the generated dataset as a zip file to make it available anywhere (Eg. Colab instance to train on GPUs).

# %%
def make_archive(output_dir: Path = OUTPUT_DIR_YOLOV8_SEGMENTATION, archive_name: str = 'archive') -> None:
    """
    Generates an archive file from the `output_dir`
    """
    shutil.make_archive(
        str(output_dir.parent / archive_name), 
        'zip', 
        output_dir
    )

def get_archive_name(dataset_names: list[str]) -> str:
    return f"archive_{'_and_'.join(dataset_names)}"


# %%
make_archive(archive_name=get_archive_name(dataset_names))

# %% [markdown]
# #### Export some datasets combinations

# %% [markdown]
# ##### All individual dataset regions

# %%
# Add the dataset names in that list to inlude them in the generated set
all_dataset_names = {
    'SEAFLOWER_BOLIVAR',
    'SEAFLOWER_COURTOWN',
    'SEAVIEW_ATL',
    'SEAVIEW_IDN_PHL',
    'SEAVIEW_PAC_AUS',
    'SEAVIEW_PAC_USA',
    'TETES_PROVIDENCIA',
}

# Parameters to generate the dataset
seed = 42
train_size_ratio = .80

for dataset_name in tqdm(all_dataset_names):
    print(f'Generating dataset for {dataset_name}')
    generate([dataset_name], seed=seed, train_size_ratio=train_size_ratio, output_dir=OUTPUT_DIR_YOLOV8_SEGMENTATION)
    archive_name = get_archive_name([dataset_name])
    print(f'Making archive {archive_name}.zip')
    make_archive(archive_name=archive_name)

# %% [markdown]
# ##### combinations of all regions but SEAVIEW_PAC_AUS

# %%
all_dataset_names_but_seaview_pac_aus = [
    'SEAFLOWER_BOLIVAR',
    'SEAFLOWER_COURTOWN',
    'SEAVIEW_ATL',
    'SEAVIEW_IDN_PHL',
    'SEAVIEW_PAC_AUS',
    'TETES_PROVIDENCIA',
]

# Parameters to generate the dataset
seed = 42
train_size_ratio = .80

print(f'Generating dataset')
generate(all_dataset_names_but_seaview_pac_aus, seed=seed, train_size_ratio=train_size_ratio, output_dir=OUTPUT_DIR_YOLOV8_SEGMENTATION)
archive_name = get_archive_name(all_dataset_names_but_seaview_pac_aus)
print(f'Making archive {archive_name}.zip')
make_archive(archive_name=archive_name)
