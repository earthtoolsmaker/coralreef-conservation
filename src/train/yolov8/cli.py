"""Finetuning YOLOv8.

This script allows the user to fine tune some YOLOv8 models. It requires
the data in the YOLOv8 Pytorch TXT format to be provided.
"""
import argparse
import logging
from pathlib import Path

import train


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        help="Name of the training experiment. A folder with this name is created to store the model artifacts.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--model",
        default="yolov8m-seg.pt",
        help="pretrained model to use for finetuning. eg. yolov8x-seg.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size: number of images per batch.",
        default=16,
    )
    parser.add_argument(
        "--degrees",
        type=int,
        help="data augmentation: random degree rotation in 0-degree.",
        default=0,
    )
    parser.add_argument(
        "--translate",
        type=float,
        help="data augmentation: random translation.",
        default=0.1,
    )
    parser.add_argument(
        "--flipud",
        type=float,
        help="data augmentation: flip upside down probability.",
        default=0.0,
    )
    parser.add_argument(
        "--data",
        help="a data.yaml file containing information for yolov8 to train.",
        default="data/05_model_input/yolov8/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/data.yaml",
    )
    parser.add_argument(
        "--data-list",
        nargs="+",
        help="a list of data.yaml file containing information for yolov8 to train. There will be one training run per data.yaml file. It makes it possible to train with the same parameters on different datasets.",
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
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(f"Train with the following args {args}")
        if args["data_list"]:
            logging.info(f"training {len(args['data_list'])} models")
            for data in args["data_list"]:
                region = Path(data).parent.name
                logging.info(f"Training for region {region}")
                experiment_name = f"{args['experiment_name']}_region_{region.lower()}"
                model = train.load_model(args["model"])
                train_args = {**args, "data": data, "experiment_name": experiment_name}
                train.train(model, train_args)
            exit(0)
        else:
            model = train.load_model(args["model"])
            train.train(model, args)
            exit(0)
