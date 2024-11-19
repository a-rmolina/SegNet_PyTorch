import json
import numpy as np
from pathlib import Path
from typing import Union


def load_json_class(json_path: Union[str, Path]) -> dict:
    """Load the class colors from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_classes_as_numpy(json_path: Union[str, Path]):
    """Convert RGB values to a numpy array of shape (n_classes, 3)."""
    rgb_values = [tuple(rgb) for rgb in load_json_class(json_path).values()]
    return np.array(rgb_values, dtype=np.uint8)
