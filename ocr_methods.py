#!/usr/bin/env python3
"""
OCR Methods - Collection of different OCR implementations for comparison
"""

import base64
import os
import time
from typing import List


def ocr_docling(image_path: str) -> str:
    """Extract text from image using Docling OCR

    Installation: !pip install docling

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Extracted text from the image in Markdown format, or an error message
    """

    from docling.document_converter import DocumentConverter

    try:
        converter = DocumentConverter()
        result = converter.convert(image_path)
        return result.document.export_to_markdown()
    except Exception as e:
        return f"ERROR: Docling OCR failed: {str(e)}"


def ocr_tesseract(image_path: str) -> str:
    """Extract text from image using Tesseract OCR

    Installation: !pip install pytesseract opencv-python
    Note: Also requires tesseract to be installed on the system.
          In Colab: !apt-get install tesseract-ocr
    """
    import cv2
    import pytesseract

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(binary)

    return text


# 2. EasyOCR
def ocr_easyocr(image_path: str) -> str:
    """Extract text from image using EasyOCR

    Installation: !pip install easyocr
    """
    import easyocr

    reader = easyocr.Reader(["en"])

    result = reader.readtext(image_path)

    text = "\n".join([item[1] for item in result])

    return text


# 3. PaddleOCR
def ocr_paddleocr(image_path: str) -> str:
    """Extract text from image using PaddleOCR

    Installation: !pip install paddlepaddle paddleocr
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    result = ocr.ocr(image_path, cls=True)

    text_lines = []
    for line in result[0]:
        if len(line) >= 2:  # Ensure we have the text part
            text_lines.append(line[1][0])  # Get text content
    text = "\n".join(text_lines)

    return text


# 6. Amazon Textract OCR
def ocr_amazon_textract(image_path: str) -> str:
    """Extract text from image using Amazon Textract

    Installation: !pip install boto3

    Note: Requires AWS credentials. Set as environment variables:
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "your_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret_key"
    os.environ["AWS_REGION_NAME"] = "your_region"
    """
    import os

    import boto3

    # # Check for credentials, uncomment if credentials not stored in .aws/credentials
    # if not (
    #     os.environ.get("AWS_ACCESS_KEY_ID")
    #     and os.environ.get("AWS_SECRET_ACCESS_KEY")
    #     and os.environ.get("AWS_REGION_NAME")
    # ):
    #     return "ERROR: AWS credentials not set"
    # Initialize client
    client = boto3.client(
        "textract",
        # region_name=os.environ.get("AWS_REGION_NAME"),
        # aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        # aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    response = client.detect_document_text(Document={"Bytes": image_bytes})

    text_lines = []
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text_lines.append(item["Text"])

    return "\n".join(text_lines)


# 7. Keras OCR (for simple text detection)
def ocr_kerasocr(image_path: str) -> str:
    """Extract text from image using Keras OCR

    Installation: !pip install keras-ocr
    """
    import keras_ocr

    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(image_path)]
    predictions = pipeline.recognize(images)
    text_with_positions = []
    for prediction in predictions[0]:
        word, box = prediction
        x, y = box[0][0], box[0][1]
        text_with_positions.append((y, x, word))

    # Sort by vertical position first (top to bottom)
    text_with_positions.sort()

    # Group words that are roughly on the same line
    line_height = 20  # Adjust based on image resolution
    lines = []
    current_line = []
    current_y = None

    for y, x, word in text_with_positions:
        if current_y is None or abs(y - current_y) < line_height:
            current_line.append((x, word))
            current_y = y
        else:
            # Sort words in the current line by horizontal position (left to right)
            current_line.sort()
            lines.append(" ".join(word for _, word in current_line))
            current_line = [(x, word)]
            current_y = y

    if current_line:
        current_line.sort()
        lines.append(" ".join(word for _, word in current_line))

    return "\n".join(lines)


# 8. DocTR (from Hugging Face)
def ocr_doctr(image_path: str) -> str:
    """Extract text from image using DocTR from Hugging Face

    Installation: !pip install python-doctr
    """
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError:
        return "ERROR: DocTR not installed. Run: pip install python-doctr"

    doc = DocumentFile.from_images(image_path)

    model = (
        ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            assume_straight_pages=True,
            symmetric_pad=True,
            pretrained=True,
            preserve_aspect_ratio=True,
        )
        # .cuda().half()  uncomment for GPU
    )
    result = model(doc)
    text = result.render()

    return text


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ocr_llm_base(image_path: str, model_name: str) -> str:
    """Base function for LLM-based OCR methods using OpenRouter.

    Args:
        image_path: Path to the image file
        model_name: Name of the model to use

    Returns:
        Extracted text from the image
    """
    try:
        from openai import OpenAI
        from pydantic import BaseModel, Field
    except ImportError:
        return "ERROR: Required packages not installed. Run: pip install openai pydantic"
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return "ERROR: OpenRouter API key not set in environment variables"

    # Create OpenAI client with OpenRouter compatibility
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    class OCRResult(BaseModel):
        markdown: str = Field(
            description="The extracted text from the image with proper formatting"
        )
        category: str = Field(
            description="The category of the document (e.g., invoice, receipt, form, letter, article)"
        )
        tags: List[str] = Field(description="The tags relevant to the document content")

    image_data = encode_image(image_path)

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate OCRs with Markdowns and correctly formatted layout when possible",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image with proper formatting. Also identify the document category and provide relevant tags.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"},
                            },
                        ],
                    },
                ],
                response_format=OCRResult,  # type: ignore
            )

            result = response.choices[0].message.parsed

            if not result:
                return ""

            return result.markdown

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return f"ERROR: {str(e)}"

    return "ERROR: Maximum retries exceeded"


def ocr_qwen32ocr(image_path: str) -> str:
    """Extract text from image using Qwen 2.5 VL 32B Instruct model via OpenRouter.

    Installation: !pip install openai python-dotenv

    Note: Requires OpenRouter API key. Set as environment variable:
    import os
    os.environ["OPENROUTER_API_KEY"] = "your_key"
    """
    return ocr_llm_base(image_path, "qwen/qwen2.5-vl-32b-instruct")


def ocr_pixtral(image_path: str) -> str:
    """Extract text from image using Claude 3.5 Sonnet model via OpenRouter.

    Installation: !pip install openai python-dotenv

    Note: Requires OpenRouter API key. Set as environment variable:
    import os
    os.environ["OPENROUTER_API_KEY"] = "your_key"
    """
    return ocr_llm_base(image_path, "mistralai/pixtral-12b")


def ocr_mistral(image_path: str) -> str:
    """Extract text from image using Mistral 3.1 model via OpenRouter.

    Installation: !pip install openai python-dotenv

    Note: Requires OpenRouter API key. Set as environment variable:
    import os
    os.environ["OPENROUTER_API_KEY"] = "your_key"
    """
    return ocr_llm_base(image_path, "mistralai/mistral-small-3.1-24b-instruct")


def ocr_gemini(image_path: str) -> str:
    """Extract text from image using Gemini 2.5 Flash model via OpenRouter.

    Installation: !pip install openai python-dotenv

    Note: Requires OpenRouter API key. Set as environment variable:
    import os
    """
    return ocr_llm_base(image_path, "google/gemini-2.5-flash-preview")


OCR_METHODS = {
    "docling": ocr_docling,
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "paddleocr": ocr_paddleocr,
    "kerasocr": ocr_kerasocr,
    "doctr": ocr_doctr,
    "amazon_textract": ocr_amazon_textract,
    "qwen32": ocr_qwen32ocr,
    "pixtral": ocr_pixtral,
    "mistral": ocr_mistral,
    "gemini": ocr_gemini,
}
