import logging
import os
from pathlib import Path
from typing import Any, Optional

from ultralytics import YOLO


def load_model(weights_path: Path):
    """Loading model weights from the provided `weights_path`.

    Returns the loaded model or None if the weights_path is not correct.
    """
    if not os.path.isfile(weights_path):
        logging.error(f"Cannot load the model weights located at {weights_path}")
        return None
    else:
        model = YOLO(weights_path)
        model.info()
        return model


def predict(model: YOLO, source: Any, save_path: Optional[Path] = None):
    """Main entrypoint for running model inference.

    When `save_path` is provided, the result of the prediction will be saved there.
    """
    if save_path:
        return model.predict(source, save=True, project=str(save_path))
    else:
        return model.predict(source)
