"""Traditional OCR method implementations."""

from src.ocr.traditional.easyocr import EasyOCRProcessor
from src.ocr.traditional.paddleocr import PaddleOCRProcessor
from src.ocr.traditional.tesseract import TesseractProcessor

__all__ = ["TesseractProcessor", "EasyOCRProcessor", "PaddleOCRProcessor"]
