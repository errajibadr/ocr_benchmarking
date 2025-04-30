"""Data loader utilities for OCR benchmarking."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image


def load_images(directory: Union[str, Path]) -> List[str]:
    """Load images from a directory.

    Args:
        directory: Directory containing image files

    Returns:
        List of image file paths
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"}

    image_paths = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))

    return sorted(image_paths)


def load_ground_truth(file_path: Union[str, Path]) -> Dict[str, str]:
    """Load ground truth data from a JSON file.

    Args:
        file_path: Path to ground truth JSON file

    Returns:
        Dictionary mapping image filenames to ground truth text
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    return ground_truth


def create_ground_truth_template(
    image_directory: Union[str, Path], output_file: Union[str, Path]
) -> None:
    """Create a template ground truth JSON file for a set of images.

    Args:
        image_directory: Directory containing image files
        output_file: Path to output JSON file
    """
    image_paths = load_images(image_directory)

    ground_truth = {}

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        ground_truth[image_name] = ""

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Ground truth template created at {output_file}")


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image at {image_path}: {e}")
