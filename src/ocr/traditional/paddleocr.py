"""PaddleOCR processor implementation."""

import logging
import os
import platform
import time
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    cv_to_pil,
    denoise_image,
    enhance_contrast,
    pil_to_cv,
    resize_image,
    sharpen_image,
)

# Set up logger
logger = logging.getLogger(__name__)


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
        save_preprocessed_images: bool = False,
        preprocessed_images_dir: str = "results/preprocessed_images",
        verbose: bool = True,
        fallback_to_tesseract: bool = False,
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
            save_preprocessed_images: Whether to save preprocessed images
            preprocessed_images_dir: Directory to save preprocessed images
            verbose: Whether to print detailed logs
            fallback_to_tesseract: Whether to fallback to Tesseract OCR if PaddleOCR fails
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
        self._save_preprocessed_images = save_preprocessed_images
        self._preprocessed_images_dir = Path(preprocessed_images_dir)
        self._verbose = verbose
        self._fallback_to_tesseract = fallback_to_tesseract
        self._ocr = None
        self._tesseract = None

        # Configure logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Create preprocessed images directory if needed
        if self._save_preprocessed_images:
            os.makedirs(self._preprocessed_images_dir, exist_ok=True)
            os.makedirs(self._preprocessed_images_dir / "original", exist_ok=True)
            os.makedirs(self._preprocessed_images_dir / "preprocessed", exist_ok=True)
            os.makedirs(self._preprocessed_images_dir / "error_annotated", exist_ok=True)

        # Check platform compatibility
        self._is_macos = platform.system() == "Darwin"
        self._is_arm64 = platform.machine() == "arm64"
        self._platform_warning = None

        if self._is_macos and self._use_gpu:
            self._use_gpu = False
            self._platform_warning = "GPU acceleration disabled on macOS ARM64 (not supported)"

        # Initialize the PaddleOCR instance
        self._initialize_ocr()

        # Initialize Tesseract as a fallback if requested
        if self._fallback_to_tesseract:
            self._initialize_tesseract()

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

    def _initialize_tesseract(self):
        """Initialize Tesseract OCR as a fallback option."""
        if self._tesseract is None:
            try:
                from src.ocr.traditional.tesseract import TesseractProcessor

                # Convert PaddleOCR language code to Tesseract language code
                lang_map = {
                    "en": "eng",
                    "ch": "chi_sim",
                    "fr": "fra",
                    "german": "deu",
                    "it": "ita",
                    "japanese": "jpn",
                    "korean": "kor",
                    "es": "spa",
                    "pt": "por",
                    "ru": "rus",
                }

                # Map the first language
                tesseract_lang = lang_map.get(self._langs[0].lower(), "eng")

                self._tesseract = TesseractProcessor(
                    lang=tesseract_lang, preprocess=self._preprocess
                )

                logger.info(
                    f"Initialized Tesseract OCR as fallback with language: {tesseract_lang}"
                )

            except ImportError:
                logger.warning("Could not import Tesseract as fallback. Fallback not available.")
                self._fallback_to_tesseract = False
            except Exception as e:
                logger.warning(f"Failed to initialize Tesseract fallback: {e}")
                self._fallback_to_tesseract = False

    def _create_error_annotated_image(
        self, image: np.ndarray, error_message: str, image_filename: str
    ) -> None:
        """Create an annotated image showing the error message.

        Args:
            image: Original image as numpy array
            error_message: Error message to display
            image_filename: Filename to save the annotated image
        """
        try:
            # Convert to PIL for text drawing
            pil_image = cv_to_pil(image)

            # Create a drawing object
            draw = ImageDraw.Draw(pil_image)

            # Prepare error message
            if len(error_message) > 100:
                error_message = error_message[:97] + "..."

            # Split into multiple lines if too long
            error_lines = []
            max_chars_per_line = 50
            for i in range(0, len(error_message), max_chars_per_line):
                error_lines.append(error_message[i : i + max_chars_per_line])

            # Add error box at the top
            w, h = pil_image.size
            box_height = 30 * (len(error_lines) + 1)

            # Draw semi-transparent red rectangle
            draw.rectangle([(0, 0), (w, box_height)], fill=(255, 0, 0, 180))

            # Add text
            try:
                font = ImageFont.truetype("Arial", 24)
            except:
                font = ImageFont.load_default()

            # Draw error title
            draw.text((10, 10), "PADDLEOCR ERROR:", fill=(255, 255, 255), font=font)

            # Draw error message
            for i, line in enumerate(error_lines):
                draw.text((10, 40 + i * 30), line, fill=(255, 255, 255), font=font)

            # Save the annotated image
            save_path = (
                self._preprocessed_images_dir / "error_annotated" / f"error_{image_filename}"
            )
            pil_image.save(str(save_path))

            logger.info(f"Created error-annotated image at {save_path}")
            print(f"Created error-annotated image at {save_path}")

        except Exception as e:
            logger.error(f"Failed to create error-annotated image: {e}")

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

    def _save_image_with_stages(self, original_image: np.ndarray, filename: str):
        """Save the original image and create a preprocessing stages visualization.

        Args:
            original_image: Original image as numpy array
            filename: Base filename to use for saved images
        """
        # Save the original image
        original_path = self._preprocessed_images_dir / "original" / filename
        cv2.imwrite(str(original_path), original_image)

        # Process and save each stage
        stages = []

        # Stage 1: Resize if needed
        h, w = original_image.shape[:2]
        if max(h, w) > 2500:
            resized = resize_image(original_image, width=2500 if w > h else int(2500 * w / h))
            stages.append(("1_resized", resized))
        else:
            resized = original_image

        # Stage 2: Grayscale
        gray = convert_to_grayscale(resized)
        stages.append(("2_grayscale", gray))

        # Stage 3: Denoise
        denoised = denoise_image(gray)
        stages.append(("3_denoised", denoised))

        # Stage 4: Enhance contrast
        enhanced = enhance_contrast(denoised)
        stages.append(("4_enhanced", enhanced))

        # Stage 5: Sharpen
        sharpened = sharpen_image(enhanced)
        stages.append(("5_sharpened", sharpened))

        # Save each stage
        for stage_name, stage_image in stages:
            stage_path = self._preprocessed_images_dir / "preprocessed" / f"{stage_name}_{filename}"
            cv2.imwrite(str(stage_path), stage_image)

        return sharpened

    def preprocess_image(self, image: np.ndarray, image_filename: str = "") -> np.ndarray:
        """Preprocess image before OCR.

        Args:
            image: Image as numpy array
            image_filename: Original image filename for saving preprocessed version

        Returns:
            Preprocessed image
        """
        if not self._preprocess:
            print("Preprocessing disabled, using original image")
            return image

        # Apply PaddleOCR-specific preprocessing
        try:
            logger.info(f"Starting preprocessing for image: {image_filename}")
            print(f"Starting preprocessing for image: {image_filename}")

            # If saving is enabled and filename is provided, save all preprocessing stages
            if self._save_preprocessed_images and image_filename:
                result = self._save_image_with_stages(image, image_filename)
                logger.info(f"Preprocessing completed for {image_filename} with all stages saved")
                return result

            # Otherwise just do the preprocessing without saving
            # Resize large images to improve speed while maintaining quality
            h, w = image.shape[:2]
            if max(h, w) > 2500:
                logger.info(f"Resizing image {image_filename} from {w}x{h}")
                image = resize_image(image, width=2500 if w > h else int(2500 * w / h))

            # Convert to grayscale
            gray = convert_to_grayscale(image)
            logger.info(f"Converted {image_filename} to grayscale")

            # Denoise
            denoised = denoise_image(gray)
            logger.info(f"Denoised {image_filename}")

            # Enhance contrast
            enhanced = enhance_contrast(denoised)
            logger.info(f"Enhanced contrast for {image_filename}")

            # Sharpen
            sharpened = sharpen_image(enhanced)
            logger.info(f"Sharpened {image_filename}")

            logger.info(f"Preprocessing completed successfully for {image_filename}")
            return sharpened
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_filename}: {e}")
            print(f"Warning: Preprocessing failed for {image_filename}, using original image: {e}")
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

        # Get image filename for saving preprocessed version
        image_path = Path(image_path)
        image_filename = image_path.name

        logger.info(f"============ PROCESSING IMAGE: {image_filename} ============")
        print(f"\n============ PROCESSING IMAGE: {image_filename} ============")

        start_time = time.time()

        # Load image
        try:
            logger.info(f"Loading image: {image_filename}")
            pil_image = load_image(image_path)
            cv_image = pil_to_cv(pil_image)
            logger.info(f"Successfully loaded image: {image_filename}")
        except Exception as e:
            logger.error(f"Failed to load image {image_filename}: {e}")
            print(f"ERROR: Failed to load image {image_filename}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                processing_time_sec=time.time() - start_time,
                metadata={
                    "engine": "PaddleOCR",
                    "languages": self._langs,
                    "error": f"Failed to load image: {str(e)}",
                    "status": "failed",
                },
            )

        # Preprocess image
        preprocessed = self.preprocess_image(cv_image, image_filename)

        # Save a copy of the original and preprocessed image for later reference
        original_copy = cv_image.copy()
        preprocessed_copy = preprocessed.copy()

        # Try to use PaddleOCR
        paddle_error = None
        try:
            # Perform OCR with PaddleOCR
            logger.info(f"Starting OCR with PaddleOCR for: {image_filename}")
            print(f"Starting OCR with PaddleOCR for: {image_filename}")
            # PaddleOCR can work with both file paths and numpy arrays
            results = self._ocr.ocr(preprocessed, cls=self._cls)
            logger.info(f"OCR completed for: {image_filename}")
        except Exception as e:
            # Store the error
            paddle_error = str(e)
            logger.error(f"PaddleOCR runtime error for {image_filename}: {e}")
            print(
                f"Warning: PaddleOCR runtime error detected for {image_filename}. Trying alternative method."
            )

            # Create error-annotated image
            if self._save_preprocessed_images:
                self._create_error_annotated_image(original_copy, str(e), image_filename)

            # Try fallback method with file path
            try:
                logger.info(f"Trying with file path instead for: {image_filename}")
                print(f"Attempting fallback method for image: {image_filename}")

                # Save preprocessed image to a temporary file and use that instead
                temp_image_path = Path(f"/tmp/paddle_temp_{time.time()}.png")
                cv2.imwrite(str(temp_image_path), preprocessed)
                results = self._ocr.ocr(str(temp_image_path), cls=self._cls)
                logger.info(f"File path method successful for: {image_filename}")

                # Clean up temp file
                if temp_image_path.exists():
                    temp_image_path.unlink()

                # Reset error since we succeeded
                paddle_error = None

            except Exception as inner_e:
                # Clean up temp file if it exists
                if "temp_image_path" in locals() and temp_image_path.exists():
                    temp_image_path.unlink()

                logger.error(f"PaddleOCR fallback failed for {image_filename}: {inner_e}")
                print(f"ERROR: PaddleOCR fallback failed for {image_filename}: {inner_e}")

                # Try fallback to Tesseract if enabled
                if self._fallback_to_tesseract and self._tesseract is not None:
                    logger.info(f"Attempting fallback to Tesseract OCR for {image_filename}")
                    print(f"Attempting fallback to Tesseract OCR for {image_filename}")
                    try:
                        # Process with Tesseract
                        tesseract_result = self._tesseract.process_image(image_path)

                        # Return the Tesseract result with metadata indicating the fallback
                        tesseract_result.metadata["original_engine"] = "PaddleOCR"
                        tesseract_result.metadata["fallback_engine"] = "Tesseract"
                        tesseract_result.metadata["fallback_reason"] = str(inner_e)
                        tesseract_result.metadata["paddle_error"] = paddle_error

                        logger.info(
                            f"Successfully processed {image_filename} with Tesseract fallback"
                        )
                        print(f"Successfully processed {image_filename} with Tesseract fallback")

                        return tesseract_result

                    except Exception as tesseract_error:
                        logger.error(
                            f"Tesseract fallback also failed for {image_filename}: {tesseract_error}"
                        )
                        print(
                            f"ERROR: Tesseract fallback also failed for {image_filename}: {tesseract_error}"
                        )

                # Return empty result with error message if all methods failed
                return OCRResult(
                    text="",
                    confidence=0.0,
                    bounding_boxes=[],
                    processing_time_sec=time.time() - start_time,
                    metadata={
                        "engine": "PaddleOCR",
                        "languages": self._langs,
                        "error": str(inner_e),
                        "original_error": paddle_error,
                        "status": "failed",
                        "image": image_filename,
                    },
                )

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

        # Log detection results
        result_text = full_text if full_text else "[NO TEXT DETECTED]"
        confidence_msg = (
            f"Average confidence: {avg_confidence:.4f}" if confidences else "No text detected"
        )
        detected_regions = len(bounding_boxes)

        logger.info(f"OCR Results for {image_filename}:")
        logger.info(
            f"  - Text detected: {result_text[:100]}{'...' if len(result_text) > 100 else ''}"
        )
        logger.info(f"  - {confidence_msg}")
        logger.info(f"  - Detected regions: {detected_regions}")
        logger.info(f"  - Processing time: {processing_time:.2f} seconds")

        print(f"OCR Results for {image_filename}:")
        print(f"  - Detected text regions: {detected_regions}")
        print(f"  - {confidence_msg}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Extracted text: {result_text[:50]}{'...' if len(result_text) > 50 else ''}")

        # Add preprocessed image information to metadata if saving is enabled
        preprocessed_image_info = None
        if self._save_preprocessed_images:
            preprocessed_image_info = {
                "original": str(self._preprocessed_images_dir / "original" / image_filename),
                "preprocessed_dir": str(self._preprocessed_images_dir / "preprocessed"),
            }

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
                "preprocessed_images": preprocessed_image_info,
                "detected_regions": detected_regions,
                "image_filename": image_filename,
                "error": paddle_error,
            },
        )
