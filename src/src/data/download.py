"""Raw Dataset Downloader.

This script allows the user to download the raw dataset provided by ReefSupport
and exposed via a public GCP bucket.
One can change the GCP bucket location by updating the `GCP_BUCKET` constant.
"""
import argparse
import logging
import os
import subprocess
from pathlib import Path

# Name of the GCP bucket where the data is stored
GCP_BUCKET: str = "rs_storage_open"


def is_gsutil_installed() -> bool:
    """Returns whether gsutil is installed."""
    command = "gsutil version -l"
    status, _ = subprocess.getstatusoutput(command)
    return status == 0


def is_data_raw_path(dest_path: Path) -> bool:
    """Returns whether `dest_path` contains 01_raw."""
    return "01_raw" in str(dest_path)


def _download_command(dest_path: Path, gcp_bucket: str = GCP_BUCKET) -> str:
    """Returns the shell command that will download the dataset at
    `dest_path`."""
    return f'gsutil -m cp -r "gs://{gcp_bucket}/" {dest_path}'


def download(dest_path: Path, gcp_bucket: str = GCP_BUCKET) -> int:
    """Downloads the full raw dataset and save it at `dest_path`.

    Assumes that `gsutil` is installed.
    """
    command = _download_command(dest_path, gcp_bucket=gcp_bucket)
    return os.system(command)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        help="directory to save the raw dataset. Make sure to use data/01_raw.",
        required=True,
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not os.path.isdir(args["to"]):
        logging.error("invalid --to path -- the directory does not exist")
        return False
    elif not is_data_raw_path(Path(args["to"])):
        logging.error("invalid --to path - should be downloaded in the 01_raw folder")
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    if not validate_parsed_args(args):
        exit(1)
    elif not is_gsutil_installed():
        logging.error("Make sure to install and configure gsutil")
        exit(1)
    else:
        path = Path(args["to"])
        logging.info(f"Downloading dataset in {path}...")
        download(path, gcp_bucket=GCP_BUCKET)
        logging.info("Done")
        exit(0)
