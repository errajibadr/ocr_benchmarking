"""Image processing utilities for OCR preprocessing."""

import cv2
import numpy as np
from PIL import Image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale.

    Args:
        image: Input image as numpy array

    Returns:
        Grayscale image as numpy array
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply denoising to image.

    Args:
        image: Input image as numpy array

    Returns:
        Denoised image as numpy array
    """
    gray = convert_to_grayscale(image)
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


def binarize_image(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize image using threshold.

    Args:
        image: Input image as numpy array
        threshold: Threshold value (0-255)

    Returns:
        Binarized image as numpy array
    """
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_binarize_image(image: np.ndarray) -> np.ndarray:
    """Binarize image using adaptive thresholding.

    Args:
        image: Input image as numpy array

    Returns:
        Adaptively binarized image as numpy array
    """
    gray = convert_to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew image.

    Args:
        image: Input image as numpy array

    Returns:
        Deskewed image as numpy array
    """
    gray = convert_to_grayscale(image)

    # Calculate skew angle
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def resize_image(image: np.ndarray, width: int = 1000) -> np.ndarray:
    """Resize image to a specified width while maintaining aspect ratio.

    Args:
        image: Input image as numpy array
        width: Target width in pixels

    Returns:
        Resized image as numpy array
    """
    h, w = image.shape[:2]
    ratio = width / float(w)
    dimensions = (width, int(h * ratio))

    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using histogram equalization.

    Args:
        image: Input image as numpy array

    Returns:
        Contrast-enhanced image as numpy array
    """
    gray = convert_to_grayscale(image)
    return cv2.equalizeHist(gray)


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """Remove shadows from image.

    Args:
        image: Input image as numpy array

    Returns:
        Image with reduced shadows as numpy array
    """
    rgb_planes = cv2.split(image)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
        )
        result_planes.append(norm_img)

    result = cv2.merge(result_planes)
    return result


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Sharpen image.

    Args:
        image: Input image as numpy array

    Returns:
        Sharpened image as numpy array
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format.

    Args:
        pil_image: PIL Image object

    Returns:
        Image as numpy array in BGR format
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format.

    Args:
        cv_image: OpenCV image as numpy array

    Returns:
        PIL Image object
    """
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
