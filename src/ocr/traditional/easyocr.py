"""EasyOCR processor implementation."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.ocr.base import (
    OCRCapabilities,
    OCRCategory,
    OCRPrivacyLevel,
    OCRProcessor,
    OCRResult,
    ResourceRequirements,
)
from src.utils.data_loader import load_image
from src.utils.image_processing import (
    convert_to_grayscale,
    denoise_image,
    enhance_contrast,
    pil_to_cv,
    resize_image,
)


class EasyOCRProcessor(OCRProcessor):
    """EasyOCR processor implementation."""

    def __init__(
        self,
        langs: List[str] = ["en"],
        gpu: bool = False,
        model_storage_directory: Optional[str] = None,
        preprocess: bool = True,
        paragraph: bool = False,
        detector: bool = True,
        recognizer: bool = True,
        verbose: bool = False,
    ):
        """Initialize EasyOCR processor.

        Args:
            langs: List of language codes (e.g., ["en", "fr"])
            gpu: Whether to use GPU acceleration if available
            model_storage_directory: Path to store/load EasyOCR models
            preprocess: Whether to apply preprocessing to images
            paragraph: Whether to group text into paragraphs
            detector: Whether to use text detection (disable to use only recognition)
            recognizer: Whether to use text recognition (disable to use only detection)
            verbose: Whether to show detailed processing information
        """
        self._langs = langs
        self._gpu = gpu
        self._model_storage_directory = model_storage_directory
        self._preprocess = preprocess
        self._paragraph = paragraph
        self._detector = detector
        self._recognizer = recognizer
        self._verbose = verbose
        self._reader = None

    def _initialize_reader(self):
        """Initialize the EasyOCR reader if not already initialized."""
        if self._reader is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(
                    self._langs,
                    gpu=self._gpu,
                    model_storage_directory=self._model_storage_directory,
                    detector=self._detector,
                    recognizer=self._recognizer,
                    verbose=self._verbose,
                )
            except ImportError:
                raise ImportError("EasyOCR is not installed. Install it with: pip install easyocr")

    @property
    def name(self) -> str:
        """Return the name of the OCR processor."""
        return "EasyOCR"

    @property
    def category(self) -> OCRCategory:
        """Return the category of OCR processor."""
        return OCRCategory.TRADITIONAL

    @property
    def privacy_level(self) -> OCRPrivacyLevel:
        """Return the privacy level of the OCR processor."""
        return OCRPrivacyLevel.LOCAL

    def get_capabilities(self) -> OCRCapabilities:
        """Return capabilities of this OCR processor."""
        return OCRCapabilities(
            languages=self._langs,
            handles_handwriting=True,
            handles_complex_layouts=True,
            handles_tables=False,
            handles_forms=False,
            supports_rotation_correction=True,
            supports_skew_correction=True,
            extracts_structure=False,
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Return resource requirements for this processor."""
        return ResourceRequirements(
            cpu_intensive=True,
            gpu_required=self._gpu,
            min_ram_gb=4.0,
            min_gpu_ram_gb=2.0 if self._gpu else 0.0,
            supports_cpu_only=True,
            estimated_time_per_page_sec=3.0,
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before OCR.

        Args:
            image: Image as numpy array

        Returns:
            Preprocessed image
        """
        if not self._preprocess:
            return image

        # Apply EasyOCR-specific preprocessing
        try:
            # Resize large images to improve speed while maintaining quality
            h, w = image.shape[:2]
            if max(h, w) > 2000:
                image = resize_image(image, width=2000 if w > h else int(2000 * w / h))

            # Convert to grayscale
            gray = convert_to_grayscale(image)

            # Denoise
            denoised = denoise_image(gray)

            # Enhance contrast
            enhanced = enhance_contrast(denoised)

            return enhanced
        except Exception as e:
            print(f"Warning: Preprocessing failed, using original image: {e}")
            return image

    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        """Process an image and return extracted text with metadata.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult containing extracted text and metadata
        """
        # Initialize the reader if needed
        self._initialize_reader()

        start_time = time.time()

        # Load image
        pil_image = load_image(image_path)
        cv_image = pil_to_cv(pil_image)

        # Preprocess image
        preprocessed = self.preprocess_image(cv_image)

        # Perform OCR with EasyOCR
        results = self._reader.readtext(preprocessed, paragraph=self._paragraph)

        # Extract text and bounding boxes
        full_text = ""
        bounding_boxes = []
        confidences = []

        for result in results:
            bbox, text, confidence = result
            if text.strip():
                # Add space between text blocks for readability
                if full_text:
                    full_text += " "
                full_text += text

                # Convert EasyOCR bbox format to our standard format
                # EasyOCR: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Our format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                box = [(int(point[0]), int(point[1])) for point in bbox]
                bounding_boxes.append(box)
                confidences.append(confidence)

        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        processing_time = time.time() - start_time

        return OCRResult(
            text=full_text,
            confidence=float(avg_confidence),
            bounding_boxes=bounding_boxes,
            processing_time_sec=processing_time,
            metadata={
                "engine": "EasyOCR",
                "languages": self._langs,
                "gpu_used": self._gpu,
                "preprocessing_applied": self._preprocess,
                "paragraph_mode": self._paragraph,
            },
        )
