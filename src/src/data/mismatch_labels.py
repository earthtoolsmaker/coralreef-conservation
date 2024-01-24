"""Retrieves mismatch labels.

TODO: generate this file from the 01_raw files provided by ReefSupport using Ponniah's notebook.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        help="directory to save the raw dataset. Make sure to use data/04_feature.",
        required=True,
    )
    parser.add_argument(
        "--from",
        help="directory where the manifest is stored. Should be in data/09_external. TODO: deprecate it as we want to recompute it from first principles.",
        required=True,
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not os.path.isfile(args["from"]):
        logging.error("invalid --to path -- the file does not exist")
        return False
    elif not is_data_feature_path(Path(args["to"])):
        logging.error("invalid --to path - should be the data/04_feature folder")
        return False
    else:
        return True


def is_data_feature_path(dest_path: Path) -> bool:
    """Returns whether `dest_path` contains 04_feature."""
    return "04_feature" in str(dest_path)


def build(orig: Path, dest: Path):
    """Currently only copies the data file in the right data directory.

    Should be generating it from the data/01_raw files.
    """
    if not os.path.isdir(dest.parent):
        logging.info(f"Making directory: {dest.parent}")
        os.makedirs(dest.parent)
    shutil.copyfile(orig, dest)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(f"Generating mismatch label csv file in {args['to']}")
        build(orig=Path(args["from"]), dest=Path(args["to"]))
        logging.info("Done")
        exit(0)
