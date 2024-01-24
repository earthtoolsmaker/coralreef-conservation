import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from evaluate import full_evaluation, full_evaluation_all


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        help="directory to save the evaluation report. Make sure to use data/08_reporting.",
        required=True,
    )
    parser.add_argument(
        "--cache-path",
        help="cache directory to save intermediate computations.",
        default="./.cache",
    )
    parser.add_argument(
        "--num-workers",
        help="number of workers to use to generate evaluation.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random Seed for sampling images for the qualitative evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for the model to make predictions",
    )
    parser.add_argument(
        "--n-qualitative-samples",
        type=int,
        default=1,
        help="Number of samples to qualitatively evaluate.",
    )
    parser.add_argument(
        "--model-root-path",
        help="model root path. eg `data/06_models/yolov8/segment/train/`. It can also point to a directory that contains all models to be evaluated.",
        # default="data/06_models/yolov8/segment/train31",
        default="data/06_models/yolov8/segment",
        required=True,
    )
    parser.add_argument(
        "--data-root-path",
        help="data root path. eg `data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR`. The dataset that will be used for the evaluation. By default, it uses the one from the training config.",
        required=False,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def model_root_path_to_evaluation_mode(model_root_path: Path) -> Optional[str]:
    """Returns `"single"` or `"multiple"` as a string whether the evaluation is
    done for a set of models or only one.

    _Note_: it can return `None` if the model_root_path is not properly passed.
    """
    if "args.yaml" in os.listdir(model_root_path):
        return "single"
    elif all(
        [
            model_root_path_to_evaluation_mode(model_root_path / model_name) == "single"
            for model_name in os.listdir(model_root_path)
        ]
    ):
        return "multiple"
    else:
        return None


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    model_root_path = Path(args["model_root_path"])
    if not model_root_path.exists():
        logging.error(f"model root path does not exist {model_root_path}")
        return False
    elif not model_root_path_to_evaluation_mode(model_root_path):
        logging.error(
            f"model root path is not pointing to the right folder for evaluation {model_root_path}"
        )
        return False
    elif args["data_root_path"] and not Path(args["data_root_path"]).exists():
        logging.error(
            f"data root path is not pointing to the right folder for evaluation {args['data_root_path']}"
        )
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        model_root_path = Path(args["model_root_path"])
        mode = model_root_path_to_evaluation_mode(model_root_path)
        data_root_path = (
            None if not args["data_root_path"] else Path(args["data_root_path"])
        )
        if mode == "single":
            full_evaluation(
                model_root_path=Path(args["model_root_path"]),
                cache_path=Path(args["cache_path"]),
                data_root_path=data_root_path,
                random_seed=args["random_seed"],
                N_samples_qualitative=args["n_qualitative_samples"],
                batch_size=args["batch_size"],
                save_path_root=Path(args["to"]),
            )
            exit(0)
        elif mode == "multiple":
            full_evaluation_all(
                models_root_path=Path(args["model_root_path"]),
                cache_path=Path(args["cache_path"]),
                data_root_path=data_root_path,
                random_seed=args["random_seed"],
                N_samples_qualitative=args["n_qualitative_samples"],
                batch_size=args["batch_size"],
                save_path_root=Path(args["to"]),
            )
            exit(0)
        else:
            exit(1)
