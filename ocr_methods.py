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
        # Export to Markdown (as in docling docs)
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

    # Read image using OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get image with only black and white
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Recognize text with Tesseract
    text = pytesseract.image_to_string(binary)

    return text


# 2. EasyOCR
def ocr_easyocr(image_path: str) -> str:
    """Extract text from image using EasyOCR

    Installation: !pip install easyocr
    """
    import easyocr

    # Initialize reader for English
    reader = easyocr.Reader(["en"])

    # Read text
    result = reader.readtext(image_path)

    # Combine text from all detected regions
    text = "\n".join([item[1] for item in result])

    return text


# 3. PaddleOCR
def ocr_paddleocr(image_path: str) -> str:
    """Extract text from image using PaddleOCR

    Installation: !pip install paddlepaddle paddleocr
    """
    from paddleocr import PaddleOCR

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    # Process image
    result = ocr.ocr(image_path, cls=True)

    # Extract text
    text_lines = []
    for line in result[0]:
        if len(line) >= 2:  # Ensure we have the text part
            text_lines.append(line[1][0])  # Get text content

    # Join all lines
    text = "\n".join(text_lines)

    return text


# 5. Microsoft Azure Read API
def ocr_azure(image_path: str) -> str:
    """Extract text from image using Azure Computer Vision OCR

    Installation: !pip install azure-cognitiveservices-vision-computervision

    Note: Requires Azure Computer Vision API key. Set as environment variable:
    import os
    os.environ["AZURE_VISION_KEY"] = "your_key"
    os.environ["AZURE_VISION_ENDPOINT"] = "your_endpoint"
    """
    import os
    import time

    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import (
        OperationStatusCodes,
    )
    from msrest.authentication import CognitiveServicesCredentials

    # Get credentials
    key = os.environ.get("AZURE_VISION_KEY")
    endpoint = os.environ.get("AZURE_VISION_ENDPOINT")

    if not key or not endpoint:
        return "ERROR: Azure Vision API key or endpoint not set"

    # Authenticate client
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    # Read image
    with open(image_path, "rb") as image_file:
        read_response = client.read_in_stream(image_file, raw=True)

    # Get operation ID
    operation_id = read_response.headers["Operation-Location"].split("/")[-1]

    # Wait for operation to complete
    max_retries = 10
    sleep_time = 1
    result = None

    for i in range(max_retries):
        read_result = client.get_read_result(operation_id)
        if read_result.status not in [
            OperationStatusCodes.running,
            OperationStatusCodes.not_started,
        ]:
            result = read_result
            break
        time.sleep(sleep_time)

    # Check for success and extract text
    if result and result.status == OperationStatusCodes.succeeded:
        text_lines = []
        for text_result in result.analyze_result.read_results:
            for line in text_result.lines:
                text_lines.append(line.text)

        return "\n".join(text_lines)

    return "Error: OCR operation failed or timed out"


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

    # # Check for credentials
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

    # Read image file
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Process image
    response = client.detect_document_text(Document={"Bytes": image_bytes})

    # Extract text
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

    # Initialize detector and recognizer
    pipeline = keras_ocr.pipeline.Pipeline()

    # Read image
    images = [keras_ocr.tools.read(image_path)]

    # Make prediction
    predictions = pipeline.recognize(images)

    # Extract text with rough position information
    text_with_positions = []
    for prediction in predictions[0]:
        word, box = prediction
        # Calculate rough position (top-left of box)
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

    # Add the last line
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

    # Load the document
    doc = DocumentFile.from_images(image_path)

    # Load model
    model = ocr_predictor(pretrained=True)

    # Analyze
    result = model(doc)

    # Extract text
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

    # Get API key from environment variable
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return "ERROR: OpenRouter API key not set in environment variables"

    # Create OpenAI client with OpenRouter compatibility
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Define the OCR result schema
    class OCRResult(BaseModel):
        markdown: str = Field(
            description="The extracted text from the image with proper formatting"
        )
        category: str = Field(
            description="The category of the document (e.g., invoice, receipt, form, letter, article)"
        )
        tags: List[str] = Field(description="The tags relevant to the document content")

    # Encode the image to base64
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

            # Parse the JSON response
            result = response.choices[0].message.parsed

            if not result:
                return ""

            # Return the extracted text
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


# Dictionary mapping method names to functions
OCR_METHODS = {
    "docling": ocr_docling,
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "paddleocr": ocr_paddleocr,
    "kerasocr": ocr_kerasocr,
    "doctr": ocr_doctr,
    # Uncomment if you have API keys/credentials:
    # "azure": ocr_azure,
    "amazon_textract": ocr_amazon_textract,
    "qwen32": ocr_qwen32ocr,
    "pixtral": ocr_pixtral,
    "mistral": ocr_mistral,
}
