#!/usr/bin/env python3
"""
OCR Methods - Collection of different OCR implementations for comparison
"""


# 1. Tesseract OCR
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


# 4. Azure OCR (Uncomment and fill API key if available)
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


# 5. Google Cloud Vision OCR
def ocr_google_vision(image_path: str) -> str:
    """Extract text from image using Google Cloud Vision OCR

    Installation: !pip install google-cloud-vision

    Note: Requires Google Cloud credentials. Set as environment variable:
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"
    """
    import os

    from google.cloud import vision

    # Check for credentials
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return "ERROR: Google Cloud credentials not set"

    # Initialize client
    client = vision.ImageAnnotatorClient()

    # Read image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Create image object
    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)

    # Extract text
    if response.error.message:
        return f"ERROR: {response.error.message}"

    # Get full text annotation
    text = response.text_annotations[0].description if response.text_annotations else ""

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

    # Check for credentials
    if not (
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
        and os.environ.get("AWS_REGION_NAME")
    ):
        return "ERROR: AWS credentials not set"

    # Initialize client
    client = boto3.client(
        "textract",
        region_name=os.environ.get("AWS_REGION_NAME"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
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


# 7. OCRSpace API
def ocr_ocrspace(image_path: str) -> str:
    """Extract text from image using OCRSpace API

    Installation: !pip install requests

    Note: Requires OCRSpace API key. Set as environment variable:
    import os
    os.environ["OCRSPACE_API_KEY"] = "your_api_key"
    """
    import os

    import requests

    # Check for API key
    api_key = os.environ.get("OCRSPACE_API_KEY")
    if not api_key:
        return "ERROR: OCRSpace API key not set"

    # Endpoint URL
    url = "https://api.ocr.space/parse/image"

    # Prepare payload
    payload = {
        "apikey": api_key,
        "language": "eng",
        "isOverlayRequired": True,
    }

    # Prepare files
    files = {"image": open(image_path, "rb")}

    # Send request
    response = requests.post(url, files=files, data=payload)

    # Check for success
    if response.status_code != 200:
        return f"ERROR: API request failed with status code {response.status_code}"

    # Parse response
    json_data = response.json()

    if not json_data.get("ParsedResults"):
        return "ERROR: No parsed results in API response"

    # Extract text
    text = json_data["ParsedResults"][0]["ParsedText"]

    return text


# 8. Keras OCR (for simple text detection)
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


# Dictionary mapping method names to functions
OCR_METHODS = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "paddleocr": ocr_paddleocr,
    # "azure": ocr_azure,  # Uncomment if Azure credentials are available
    # "google_vision": ocr_google_vision,  # Uncomment if Google Cloud credentials are available
    # "amazon_textract": ocr_amazon_textract,  # Uncomment if AWS credentials are available
    # "ocrspace": ocr_ocrspace,  # Uncomment if OCRSpace API key is available
    "kerasocr": ocr_kerasocr,
}
