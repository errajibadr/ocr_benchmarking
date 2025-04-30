"""Tesseract OCR processor implementation."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract
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
    adaptive_binarize_image,
    denoise_image,
    deskew_image,
    pil_to_cv,
)


class TesseractProcessor(OCRProcessor):
    """Tesseract OCR processor implementation."""

    def __init__(
        self,
        lang: str = "eng",
        config: Optional[str] = None,
        preprocess: bool = True,
        tessdata_dir: Optional[str] = None,
    ):
        """Initialize Tesseract OCR processor.

        Args:
            lang: Language(s) to use (comma-separated)
            config: Tesseract configuration string
            preprocess: Whether to apply preprocessing to images
            tessdata_dir: Path to tessdata directory (if custom)
        """
        self._lang = lang
        self._config = config
        self._preprocess = preprocess
        self._tessdata_dir = tessdata_dir

        # Use environment variable for tessdata dir if provided
        if tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = tessdata_dir

    @property
    def name(self) -> str:
        """Return the name of the OCR processor."""
        return "Tesseract"

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
            languages=self._lang.split(","),
            handles_handwriting=False,
            handles_complex_layouts=False,
            handles_tables=True,
            handles_forms=False,
            supports_rotation_correction=True,
            supports_skew_correction=True,
            extracts_structure=False,
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Return resource requirements for this processor."""
        return ResourceRequirements(
            cpu_intensive=True,
            gpu_required=False,
            min_ram_gb=2.0,
            min_gpu_ram_gb=0.0,
            supports_cpu_only=True,
            estimated_time_per_page_sec=2.0,
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

        # Apply a series of preprocessing steps to improve OCR accuracy
        try:
            # Denoise
            denoised = denoise_image(image)

            # Deskew
            deskewed = deskew_image(denoised)

            # Binarize
            binarized = adaptive_binarize_image(deskewed)

            return binarized
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
        start_time = time.time()

        # Load image
        pil_image = load_image(image_path)
        cv_image = pil_to_cv(pil_image)

        # Preprocess image
        preprocessed = self.preprocess_image(cv_image)

        # OCR with Tesseract
        custom_config = self._config or f"-l {self._lang} --oem 1 --psm 3"
        result_data = pytesseract.image_to_data(
            preprocessed, config=custom_config, output_type=pytesseract.Output.DICT
        )

        # Extract text and confidence
        text_parts = []
        bounding_boxes = []
        confidences = []

        for i in range(len(result_data["text"])):
            if int(result_data["conf"][i]) > 0:  # Filter out low confidence results
                text = result_data["text"][i]
                if text.strip():
                    text_parts.append(text)
                    x, y, w, h = (
                        result_data["left"][i],
                        result_data["top"][i],
                        result_data["width"][i],
                        result_data["height"][i],
                    )
                    box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    bounding_boxes.append(box)
                    confidences.append(int(result_data["conf"][i]))

        # Calculate average confidence
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

        # Combine text parts
        full_text = " ".join(text_parts)

        processing_time = time.time() - start_time

        return OCRResult(
            text=full_text,
            confidence=float(avg_confidence),
            bounding_boxes=bounding_boxes,
            processing_time_sec=processing_time,
            metadata={
                "engine": "Tesseract",
                "version": pytesseract.get_tesseract_version(),
                "language": self._lang,
                "preprocessing_applied": self._preprocess,
            },
        )
