"""Hyperparameters search.

This script allows the user to perform Hyperparameters search in a
reproducible way. The results are logged to weights and biases to make
it easy to evaluate the impact of specific hyperparameters.
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np

import train
import wandb

project = "my-awesome-project"

config = {
    "model": "yolov8n-seg",
    "data": "ALL_REGIONS",
    "epochs": 2,
    "close_mosaic": 10,
    "imgsz": 640,
    "degrees": 90,
    "flipud": 0.5,
    "translate": 0.1,
}

param_grid = {
    "flipud": np.linspace(0.0, 0.7, 8, dtype=float),
    "translate": np.linspace(0.1, 0.6, 6, dtype=float),
    "degrees": np.linspace(0, 180, 5, dtype=int),
    "imgsz": np.linspace(640, 1024, 4, dtype=int),
    "epochs": np.linspace(100, 300, 40, dtype=int),
    "close_mosaic": np.linspace(10, 80, 10, dtype=int),
    "lr0": list(np.logspace(np.log10(0.001), np.log10(0.1), base=10, num=20)),
    "lrf": list(np.logspace(np.log10(0.001), np.log10(0.1), base=10, num=20)),
    "amp": [True, False],
    "weight_decay": list(
        np.logspace(np.log10(0.00001), np.log10(0.01), base=10, num=50)
    ),
    "optimizer": [
        "SGD",
        "Adam",
        "Adamax",
        "AdamW",
        "NAdam",
        "RAdam",
        "RMSProp",
        "auto",
    ],
}

param_grid = {
    "boosting_type": ["gbdt", "goss", "dart"],
    "num_leaves": list(range(20, 150)),
    "learning_rate": list(
        np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)
    ),
    "subsample_for_bin": list(range(20000, 300000, 20000)),
    "min_child_samples": list(range(20, 500, 5)),
    "reg_alpha": list(np.linspace(0, 1)),
    "reg_lambda": list(np.linspace(0, 1)),
    "colsample_bytree": list(np.linspace(0.6, 1, 10)),
    "subsample": list(np.linspace(0.5, 1, 100)),
    "is_unbalance": [True, False],
}

# start a new wandb run to track this script
wandb.init(
    project=project,
    config=config,
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
