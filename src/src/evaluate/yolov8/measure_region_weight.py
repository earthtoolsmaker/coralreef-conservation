"""Script to count the number of pixels per dataset and region.

This was used to generate a section in the report.
"""
from pathlib import Path

import cv2
import yaml


def yaml_content(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def number_pixels(path: Path) -> int:
    """Returns the number of pixels present in the image located at `path`."""
    im = cv2.imread(str(path))
    return im.shape[0] * im.shape[1] * im.shape[2]


def data_root_path_to_region_pixel_number(
    data_root_path: Path, split: str = "test"
) -> dict:
    """Returns a dict that summarizes the number of pixels per regions for the
    given `data_root_path` and `split`.

    It returns the raw number of pixels and the normalized numbers.
    """
    config = yaml_content(data_root_path / "config.yaml")
    results = {}
    for region, filenames in config[f"{split}_dataset"].items():
        filepaths = [
            data_root_path / split / "images" / filename for filename in filenames
        ]
        pixels = sum([number_pixels(image_filepath) for image_filepath in filepaths])
        results[region] = pixels
    total = sum(results.values())
    for region, pxl in results.items():
        results[region] = {"number_pixels": pxl, "percent": pxl / total}
    return results


# REPL driven
# data_root_path = Path(
#     "data/05_model_input/yolov8/vtest/SEAFLOWER_BOLIVAR_and_SEAFLOWER_COURTOWN_and_SEAVIEW_ATL_and_SEAVIEW_IDN_PHL_and_SEAVIEW_PAC_AUS_and_TETES_PROVIDENCIA/"
# )
# r = data_root_path_to_region_pixel_number(data_root_path, split="test")
# r
