"""Abstract base class for OCR processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel


class OCRCategory(str, Enum):
    """Categories of OCR processors."""

    TRADITIONAL = "traditional"
    VLM = "vision_language_model"


class OCRPrivacyLevel(str, Enum):
    """Privacy levels for OCR processors."""

    LOCAL = "local"  # Data stays on local machine
    CLOUD = "cloud"  # Data is sent to cloud service
    MIXED = "mixed"  # Some processing local, some in cloud


class ResourceRequirements(BaseModel):
    """Resource requirements for OCR processor."""

    cpu_intensive: bool = False
    gpu_required: bool = False
    min_ram_gb: float = 1.0
    min_gpu_ram_gb: float = 0.0
    supports_cpu_only: bool = True
    estimated_time_per_page_sec: float = 0.0


class OCRCapabilities(BaseModel):
    """Capabilities of OCR processor."""

    languages: List[str]
    handles_handwriting: bool = False
    handles_complex_layouts: bool = False
    handles_tables: bool = False
    handles_forms: bool = False
    supports_rotation_correction: bool = False
    supports_skew_correction: bool = False
    extracts_structure: bool = False


class OCRResult(BaseModel):
    """Result of OCR processing."""

    text: str
    confidence: Optional[float] = None
    bounding_boxes: Optional[List[List[Tuple[int, int]]]] = None
    processing_time_sec: float
    metadata: Dict[str, Any] = {}


class OCRProcessor(ABC):
    """Abstract base class for OCR processors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the OCR processor."""
        pass

    @property
    @abstractmethod
    def category(self) -> OCRCategory:
        """Return the category of OCR processor."""
        pass

    @property
    @abstractmethod
    def privacy_level(self) -> OCRPrivacyLevel:
        """Return the privacy level of the OCR processor."""
        pass

    @abstractmethod
    def get_capabilities(self) -> OCRCapabilities:
        """Return capabilities of this OCR processor."""
        pass

    @abstractmethod
    def get_resource_requirements(self) -> ResourceRequirements:
        """Return resource requirements for this processor."""
        pass

    @abstractmethod
    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        """Process an image and return extracted text with metadata.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult containing extracted text and metadata
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before OCR.

        Args:
            image: Image as numpy array

        Returns:
            Preprocessed image
        """
        pass
