#!/usr/bin/env python3
"""
OCR Methods - Collection of different OCR implementations for comparison
"""

import base64
import os
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)


system_prompt = """
You are a professional OCR assistant. Your task is to extract ALL visible text from images while preserving the original document structure and formatting.

**CAPABILITIES**: Text recognition, table extraction, formula detection, layout preservation

**OUTPUT REQUIREMENTS**:\n
- Extract every piece of readable text including headers, body text, captions, footnotes
- Maintain spatial relationships and reading order
- Use markdown for tables: | Column 1 | Column 2 |
- Preserve lists, numbering, and indentation
- Mark unclear text as [UNCLEAR: best_guess]
- Mark illegible text as [ILLEGIBLE]

Do Preserve content exactly as written. Extract only what is visible in the image.
"""

user_prompt = """
document content : 
```markdown
"""


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


def ocr_llm_base(
    image_path: str,
    model_name: str,
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str | None = None,
) -> str:
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
        api_key=api_key or openrouter_api_key,
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


def ocr_vllm_openai(
    image_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Extract text from image using VLLM OpenAI-compatible API.

    Args:
        image_path: Path to the image file
        model_name: Model name to use (default: Qwen/Qwen2.5-VL-3B-Instruct-AWQ)
        base_url: API base URL (defaults to RUNPOD_BASE_URL env var)
        api_key: API key (defaults to RUNPOD_API_KEY env var)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Custom system prompt (optional)
        user_prompt: Custom user prompt (optional)
        verbose: Print timing and token information

    Returns:
        Extracted text from the image
    """
    from openai import OpenAI

    # Use environment variables as defaults
    base_url = base_url or os.getenv("RUNPOD_BASE_URL")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")

    if not base_url or not api_key:
        return "ERROR: base_url and api_key must be provided or set as environment variables"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a professional OCR assistant. Your task is to extract ALL visible text from images "
            "while preserving the original document structure and formatting.\n\n"
            "**CAPABILITIES**: Text recognition, table extraction, formula detection, layout preservation\n\n"
            "**OUTPUT REQUIREMENTS**:\n"
            "- Extract every piece of readable text including headers, body text, captions, footnotes\n"
            "- Maintain spatial relationships and reading order\n"
            "- Use markdown for tables: | Column 1 | Column 2 |\n"
            "- Preserve lists, numbering, and indentation\n"
            "- Mark unclear text as [UNCLEAR: best_guess]\n"
            "- Mark illegible text as [ILLEGIBLE]\n\n"
            "do Preserve content exactly as written. Extract only what is visible in the image."
        )

    # Default user prompt
    if user_prompt is None:
        user_prompt = "Document content : \n```markdown"

    image_data = encode_image(image_path)
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        duration = time.time() - start_time

        if verbose:
            print(f"Duration: {duration:.2f} seconds")

            # Calculate tokens per second if usage information is available
            if response.usage and response.usage.completion_tokens:
                tokens_per_second = response.usage.completion_tokens / duration
                print(f"Tokens/second: {tokens_per_second:.2f}")
                print(f"Total tokens: {response.usage.total_tokens}")
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Completion tokens: {response.usage.completion_tokens}")
            else:
                print("Token usage information not available")

        return response.choices[0].message.content or ""

    except Exception as e:
        return f"ERROR: VLLM OCR failed: {str(e)}"


# Create specific model functions using regular function definitions
def qwen_3b_awq(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 3B AWQ model."""
    return ocr_vllm_openai(image_path, model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", **kwargs)


def qwen_3b(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 3B model."""
    return ocr_vllm_openai(image_path, model_name="Qwen/Qwen2.5-VL-3B-Instruct", **kwargs)


def qwen_7b(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 7B model."""
    return ocr_vllm_openai(image_path, model_name="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)


def qwen_7b_w8a(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 7B W8A model."""
    return ocr_vllm_openai(
        image_path, model_name="RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w8a8", **kwargs
    )


def qwen_7b_awq(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 7B AWQ model."""
    return ocr_vllm_openai(image_path, model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", **kwargs)


def qwen_32b_awq(image_path: str, **kwargs) -> str:
    """OCR using Qwen 2.5 VL 32B AWQ model."""
    return ocr_vllm_openai(image_path, model_name="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", **kwargs)


def rolmocr_8b(image_path: str, **kwargs) -> str:
    """OCR using reducto/RolmOCR 8B model."""
    return ocr_vllm_openai(image_path, model_name="reducto/RolmOCR", **kwargs)


def rolmocr_q4_k_m(image_path: str, **kwargs) -> str:
    """OCR using reducto/RolmOCR Q4_K_M model."""
    return ocr_vllm_openai(image_path, model_name="mradermacher/RolmOCR-GGUF:Q4_K_M", **kwargs)


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


def ocr_ollama(
    image_path: str,
    model_name: str = "qwen2.5vl:3b",
    custom_system_prompt: str = "none",
    custom_user_prompt: str = "none",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> str:
    """Extract text from image using Ollama model.

    Installation: !pip install ollama
    """
    from openai import OpenAI
    from pydantic import BaseModel, Field

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    image_data = encode_image(image_path)

    class OCRResult(BaseModel):
        markdown: str = Field(
            description="The extracted text from the image with proper formatting"
        )
        category: str = Field(
            description="The category of the document (e.g., invoice, receipt, form, letter, article)"
        )
        tags: List[str] = Field(description="The tags relevant to the document content")

    u_system_prompt = custom_system_prompt or system_prompt
    u_user_prompt = custom_user_prompt or user_prompt

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": u_system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": u_user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    markdown = response.choices[0].message.content or ""
    print(markdown)
    return markdown


def ocr_qwen_vl_2_5(image_path: str, model_name: str = "qwen2.5vl:3b") -> str:
    """Extract text from image using Ollama model.

    Installation: !pip install ollama
    """
    return ocr_ollama(image_path, model_name)


def ocr_smolvlm256(
    image_path: str, model_name: str = "hf.co/ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0"
) -> str:
    """Extract text from image using SmolVLM model.

    Installation: !pip install ollama
    """
    return ocr_ollama(image_path, model_name)


def ocr_smolvlm500(
    image_path: str, model_name: str = "hf.co/ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0"
) -> str:
    """Extract text from image using SmolVLM model.



    Installation: !pip install ollama
    """
    return ocr_ollama(image_path, model_name)


def ocr_smoldocling_256(
    image_path: str, model_name: str = "hf.co/mradermacher/SmolDocling-256M-preview-GGUF:Q4_K_M"
) -> str:
    """Extract text from image using SmolDocLing model.

    should pull model from hf.co/mradermacher/SmolDocling-256M-preview-GGUF:Q4_K_M before running it

    Installation: !pip install ollama
    """
    # Load and run the model:
    return ocr_ollama(
        image_path,
        model_name,
        custom_system_prompt="convert this page to docling",
        custom_user_prompt="-",
        max_tokens=2000,
        temperature=0.1,
    )


def ocr_mlx_smolvlm2_2b(image_path, model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
    import mlx.core as mx
    from mlx_lm.generate import generate
    from mlx_lm.utils import load
    from PIL import Image

    # Load SmolVLM with MLX
    model, tokenizer = load(model_name)
    # This is a simplified example - you might need to adapt based on model format
    response = generate(model, tokenizer, prompt="extract text from this image", max_tokens=200)
    return response


# Legacy function for backward compatibility
def ocr_vllm_openai_legacy(image_path: str) -> str:
    """Legacy function - use qwen_7b_awq instead."""
    return qwen_7b_awq(image_path, verbose=True)


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
    "qwen2.5vl-3b": ocr_qwen_vl_2_5,
    "smolvlm256": ocr_smolvlm256,
    "smolvlm500": ocr_smolvlm500,
    "mlx_smolvlm2_2b": ocr_mlx_smolvlm2_2b,
    # VLLM methods - Basic models
    "qwen_3b": qwen_3b,
    "qwen_3b_awq": qwen_3b_awq,
    "qwen_7b": qwen_7b,
    "qwen_7b_w8a": qwen_7b_w8a,
    "qwen_7b_awq": qwen_7b_awq,
    "qwen_32b_awq": qwen_32b_awq,
    "RolmOCR_8b": rolmocr_8b,
    "RolmOCR_Q4_K_M": rolmocr_q4_k_m,
    # Legacy compatibility
    "vllm_openai": ocr_vllm_openai_legacy,
}

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    # print(
    #     ocr_vllm_openai(
    #         "dataset/sample/images/82200067_0069.png",
    #         model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    #         verbose=True,
    #     )
    # )

    # print(qwen_3b_awq("dataset/sample/images/82200067_0069.png", verbose=True))
    # Test the new modular functions

    test_image = "dataset/sample/images/82200067_0069.png"

    print(ocr_smoldocling_256(test_image))
    # # Test different model variants
    # print("Testing Qwen 3B (fast):")
    # result = qwen_3b_fast(test_image, verbose=True)
    # print(f"Result length: {len(result)} characters")
    # print()

    # print("Testing Qwen 7B (detailed):")
    # result = qwen_7b_detailed(test_image, verbose=True)
    # print(f"Result length: {len(result)} characters")
    # print()

    # # Test custom configuration
    # print("Testing custom configuration:")
    # custom_ocr = partial(
    #     ocr_vllm_openai,
    #     model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    #     max_tokens=500,
    #     temperature=0.0,
    #     system_prompt="Extract only the main text, ignore headers and footers.",
    # )
    # result = custom_ocr(test_image, verbose=True)
    # print(f"Custom result length: {len(result)} characters")
