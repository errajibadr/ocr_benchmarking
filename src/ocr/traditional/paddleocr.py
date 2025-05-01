"""PaddleOCR processor implementation."""

import os
import platform
import time
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

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
    sharpen_image,
)


class PaddleOCRProcessor(OCRProcessor):
    """PaddleOCR processor implementation."""

    def __init__(
        self,
        langs: List[str] = ["en"],
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
        preprocess: bool = True,
        use_mp: bool = False,
        total_process_num: int = 1,
        show_log: bool = False,
    ):
        """Initialize PaddleOCR processor.

        Args:
            langs: List of language codes (e.g., ["en", "fr"])
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU acceleration if available
            det: Whether to use text detection
            rec: Whether to use text recognition
            cls: Whether to use text orientation classification
            preprocess: Whether to apply preprocessing to images
            use_mp: Whether to use multi-processing
            total_process_num: Total processes for multi-processing
            show_log: Whether to show PaddleOCR logs
        """
        self._langs = langs
        self._use_angle_cls = use_angle_cls
        self._use_gpu = use_gpu
        self._det = det
        self._rec = rec
        self._cls = cls
        self._preprocess = preprocess
        self._use_mp = use_mp
        self._total_process_num = total_process_num
        self._show_log = show_log
        self._ocr = None

        # Check platform compatibility
        self._is_macos = platform.system() == "Darwin"
        self._is_arm64 = platform.machine() == "arm64"
        self._platform_warning = None

        if self._is_macos and self._is_arm64 and self._use_gpu:
            self._use_gpu = False
            self._platform_warning = "GPU acceleration disabled on macOS ARM64 (not supported)"

        # Initialize the PaddleOCR instance
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize the PaddleOCR instance if not already initialized."""
        if self._ocr is None:
            try:
                # Set environment variables to avoid issues on macOS
                if self._is_macos:
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

                from paddleocr import PaddleOCR

                # Map language codes to PaddleOCR's language codes
                lang_map = {
                    "en": "en",
                    "fr": "fr",
                    "zh": "ch",
                    "chinese": "ch",
                    "ch": "ch",
                    "japan": "japan",
                    "japanese": "japan",
                    "jp": "japan",
                    "korea": "korean",
                    "korean": "korean",
                    "kr": "korean",
                    "german": "german",
                    "de": "german",
                    "it": "it",
                    "italian": "it",
                    "es": "es",
                    "spanish": "es",
                    "ru": "ru",
                    "russian": "ru",
                    "pt": "pt",
                    "portuguese": "pt",
                    "ar": "ar",
                    "arabic": "ar",
                    "hi": "hi",
                    "hindi": "hi",
                    "ug": "ug",
                    "uighur": "ug",
                    "fa": "fa",
                    "persian": "fa",
                    "ur": "ur",
                    "urdu": "ur",
                    "rs": "rs",
                    "serbian": "rs",
                    "oc": "oc",
                    "occitan": "oc",
                    "mr": "mr",
                    "marathi": "mr",
                    "ne": "ne",
                    "nepali": "ne",
                    "sr": "rs",
                    "vi": "vi",
                    "vietnamese": "vi",
                }

                # Convert language codes to PaddleOCR's language codes
                paddle_langs = [lang_map.get(lang.lower(), lang) for lang in self._langs]
                # Use the first language as the main language
                lang = paddle_langs[0]

                # Silence warnings from PaddleOCR
                if not self._show_log:
                    import warnings

                    warnings.filterwarnings("ignore")

                self._ocr = PaddleOCR(
                    use_angle_cls=self._use_angle_cls,
                    lang=lang,
                    use_gpu=self._use_gpu,
                    det=self._det,
                    rec=self._rec,
                    cls=self._cls,
                    use_mp=self._use_mp,
                    total_process_num=self._total_process_num,
                    show_log=self._show_log,
                )

                if self._platform_warning:
                    print(f"Warning: {self._platform_warning}")

            except ImportError:
                raise ImportError(
                    "PaddleOCR is not installed. Install it with: pip install paddlepaddle paddleocr"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")

    @property
    def name(self) -> str:
        """Return the name of the OCR processor."""
        return "PaddleOCR"

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
            handles_tables=True,
            handles_forms=True,
            supports_rotation_correction=self._use_angle_cls,
            supports_skew_correction=True,
            extracts_structure=True,
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Return resource requirements for this processor."""
        return ResourceRequirements(
            cpu_intensive=True,
            gpu_required=self._use_gpu,
            min_ram_gb=4.0,
            min_gpu_ram_gb=2.0 if self._use_gpu else 0.0,
            supports_cpu_only=True,
            estimated_time_per_page_sec=1.5,
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

        # Apply PaddleOCR-specific preprocessing
        try:
            # Resize large images to improve speed while maintaining quality
            h, w = image.shape[:2]
            if max(h, w) > 2500:
                image = resize_image(image, width=2500 if w > h else int(2500 * w / h))

            # Convert to grayscale
            gray = convert_to_grayscale(image)

            # Denoise
            denoised = denoise_image(gray)

            # Enhance contrast
            enhanced = enhance_contrast(denoised)

            # Sharpen
            sharpened = sharpen_image(enhanced)

            return sharpened
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
        # Ensure the PaddleOCR instance is initialized
        if self._ocr is None:
            self._initialize_ocr()

        # Double-check that PaddleOCR is properly initialized
        if self._ocr is None:
            raise RuntimeError("Failed to initialize PaddleOCR")

        start_time = time.time()

        # Load image
        pil_image = load_image(image_path)
        cv_image = pil_to_cv(pil_image)

        # Preprocess image
        preprocessed = self.preprocess_image(cv_image)

        # Implement a fallback mechanism for platforms with issues
        try:
            # Perform OCR with PaddleOCR
            # PaddleOCR can work with both file paths and numpy arrays
            results = self._ocr.ocr(preprocessed, cls=self._cls)
        except RuntimeError as e:
            # Fallback for macOS or other platform issues
            if "No allocator found for the place" in str(e) or "Place(undefined:0)" in str(e):
                print("Warning: PaddleOCR runtime error detected. Using image file path directly.")
                # Save preprocessed image to a temporary file and use that instead
                temp_image_path = Path(f"/tmp/paddle_temp_{time.time()}.png")
                cv2.imwrite(str(temp_image_path), preprocessed)
                try:
                    results = self._ocr.ocr(str(temp_image_path), cls=self._cls)
                    # Clean up temp file
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                except Exception as inner_e:
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                    print(f"Warning: PaddleOCR fallback failed: {inner_e}")
                    # Return empty result with error message
                    return OCRResult(
                        text="",
                        confidence=0.0,
                        bounding_boxes=[],
                        processing_time_sec=time.time() - start_time,
                        metadata={
                            "engine": "PaddleOCR",
                            "languages": self._langs,
                            "error": str(inner_e),
                            "status": "failed",
                        },
                    )
            else:
                # Reraise other errors
                raise

        # PaddleOCR returns a list of results for each image
        # Extract text and bounding boxes
        full_text = ""
        bounding_boxes = []
        confidences = []

        # Handle results from PaddleOCR v2.6+ which returns a list of list
        if results is not None:
            if isinstance(results, list) and len(results) > 0:
                # Handle results directly if they're not nested (older versions)
                if not isinstance(results[0], list):
                    for line in results:
                        bbox, (text, confidence) = line
                        if text and text.strip():
                            if full_text:
                                full_text += " "
                            full_text += text
                            box = [(int(point[0]), int(point[1])) for point in bbox]
                            bounding_boxes.append(box)
                            confidences.append(confidence)
                # Handle nested results (newer versions)
                elif isinstance(results[0], list):
                    for line in results[0]:
                        # Each line is a tuple of (bounding box, (text, confidence))
                        if len(line) == 2:  # Expected format
                            bbox, (text, confidence) = line
                            if text and text.strip():
                                if full_text:
                                    full_text += " "
                                full_text += text
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
                "engine": "PaddleOCR",
                "languages": self._langs,
                "gpu_used": self._use_gpu,
                "angle_cls": self._use_angle_cls,
                "preprocessing_applied": self._preprocess,
                "platform_warning": self._platform_warning,
            },
        )
