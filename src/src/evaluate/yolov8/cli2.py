import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

from evaluate2 import full_eval
from tqdm import tqdm


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        help="directory to save the evaluation report. Make sure to use data/08_reporting.",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the model to make predictions",
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
        help="data root path. eg `data/05_model_input/yolov8/v2/SEAFLOWER_BOLIVAR`. The dataset that will be used for the evaluation. By default, it uses the one from the training config. It can also point to a directory, in that case, one evaluation run is performed per subfolder.",
        required=False,
    )
    parser.add_argument(
        "--raw-images-root-path",
        help="Raw images root path.",
        default="data/01_raw/rs_storage_open/benthic_datasets/mask_labels/rs_labelled",
    )
    parser.add_argument(
        "--data-split",
        help="On which data split to evaluate the model on. Could be `val`, `test` or `train`.",
        default="test",
    )
    parser.add_argument(
        "--save-predictions-path",
        help="path to store the dense masks of the predictions. eg `data/07_model_output/yolov8/evaluation/`.",
        default="data/07_model_output/yolov8/evaluation/",
        required=False,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="number of image filepaths to sampe during qualitative evaluation",
        default=10,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed",
        default=42,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def data_root_path_to_evaluation_mode(data_root_path: Optional[Path]) -> Optional[str]:
    """Returns `"single"` or `"multiple"` as a string whether the evaluation is
    done for a set of data_root_path or only one.

    _Note_: it can return `None` if the data_root_path is not properly threaded.
    """
    if not data_root_path:
        return "single"
    elif "data.yaml" in os.listdir(data_root_path):
        return "single"
    elif all(
        [
            data_root_path_to_evaluation_mode(data_root_path / subfolder) == "single"
            for subfolder in os.listdir(data_root_path)
        ]
    ):
        return "multiple"
    else:
        return None


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
    data_root_path = (
        None if not args["data_root_path"] else Path(args["data_root_path"])
    )
    if not model_root_path.exists():
        logging.error(f"model root path does not exist {model_root_path}")
        return False
    elif not model_root_path_to_evaluation_mode(model_root_path):
        logging.error(
            f"model root path is not pointing to the right folder for evaluation {model_root_path}"
        )
        return False
    elif not data_root_path_to_evaluation_mode(data_root_path):
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
    print(args)
    if not validate_parsed_args(args):
        exit(1)
    else:
        model_root_path = Path(args["model_root_path"])
        save_predictions_path = (
            None
            if not args["save_predictions_path"]
            else Path(args["save_predictions_path"])
        )
        data_root_path = (
            None if not args["data_root_path"] else Path(args["data_root_path"])
        )
        raw_images_root_path = Path(args["raw_images_root_path"])
        model_mode = model_root_path_to_evaluation_mode(model_root_path)
        logging.info(f"data_root_path: {data_root_path}")
        data_mode = data_root_path_to_evaluation_mode(data_root_path)
        if model_mode == "single" and data_mode == "single":
            full_eval(
                model_root_path=model_root_path,
                data_root_path=data_root_path,
                raw_images_root_path=raw_images_root_path,
                data_split=args["data_split"],
                save_path=Path(args["to"]),
                random_seed=args["random_seed"],
                batch_size=args["batch_size"],
                N_samples=args["n_samples"],
                save_predictions_path=save_predictions_path,
            )
            exit(0)
        elif model_mode == "multiple" and data_mode == "single":
            model_names = os.listdir(model_root_path)
            logging.info(
                f"Evaluating the following {len(model_names)} models: {model_names}"
            )
            for model_name in tqdm(model_names):
                mrp = model_root_path / model_name
                logging.info(f"model_root_path: {mrp}")
                logging.info(f"model name: {model_name}")
                full_eval(
                    model_root_path=mrp,
                    data_root_path=data_root_path,
                    raw_images_root_path=raw_images_root_path,
                    data_split=args["data_split"],
                    save_path=Path(args["to"]),
                    random_seed=args["random_seed"],
                    batch_size=args["batch_size"],
                    N_samples=args["n_samples"],
                    save_predictions_path=save_predictions_path,
                )
            exit(0)
        elif model_mode == "single" and data_mode == "multiple":
            assert data_root_path, "data_root_path should be defined"
            data_root_paths = [
                Path(data_root_path) / p for p in os.listdir(data_root_path)
            ]
            logging.info(
                f"Evaluating the model {model_root_path.name} on each data path {data_root_paths}"
            )
            for data_root_path in tqdm(data_root_paths):
                logging.info(f"evaluation on data_root_path: {data_root_path}")
                full_eval(
                    model_root_path=model_root_path,
                    data_root_path=data_root_path,
                    raw_images_root_path=raw_images_root_path,
                    data_split=args["data_split"],
                    save_path=Path(args["to"]),
                    random_seed=args["random_seed"],
                    batch_size=args["batch_size"],
                    N_samples=args["n_samples"],
                    save_predictions_path=save_predictions_path,
                )
            exit(0)
        elif model_mode == "multiple" and data_mode == "multiple":
            data_root_paths = [Path(p) for p in os.listdir(data_root_path)]
            model_names = os.listdir(model_root_path)
            logging.info(
                f"Evaluating the models {model_names} on each data path {data_root_paths}"
            )
            for model_name in tqdm(model_names):
                for data_root_path in tqdm(data_root_paths):
                    # TODO: call full_eval with the right params
                    time.sleep(0.1)
            exit(0)
        else:
            exit(1)
