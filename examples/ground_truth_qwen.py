# To run this code you need to install the following dependencies:
# pip install openai python-dotenv tqdm

import base64
import json
import os
import pathlib
import shutil
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sympy import im
from tqdm import tqdm

load_dotenv()


def sample_dataset(dataset_dir: str = "dataset/testing_data/images", seed: int | None = 1):
    import pathlib
    import random

    if seed is not None:
        random.seed(seed)

    dataset = list(pathlib.Path(dataset_dir).glob("*.png"))
    sampled_dataset = random.sample(dataset, k=10)
    return sampled_dataset


def create_sample_dataset(dataset_dir: str = "dataset/testing_data/images"):
    import pathlib

    sampled_dataset = sample_dataset(dataset_dir)
    sample_dir = pathlib.Path("dataset/sample/images")
    sample_dir.mkdir(exist_ok=True, parents=True)

    for image in sampled_dataset:
        shutil.copy(image, sample_dir / image.name)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OCRResult(BaseModel):
    """Schema for OCR result including the extracted text, document category and tags."""

    markdown: str = Field(description="The extracted text from the image with proper formatting")
    category: str = Field(
        description="The category of the document (e.g., invoice, receipt, form, letter, article)"
    )
    tags: list[str] = Field(description="The tags relevant to the document content")


def generate_ocr_for_image(
    client: OpenAI, model: str, image_path: str, retries: int = 1, delay: int = 1
) -> str:
    """Generate OCR for a single image using Gemini through OpenAI API.

    Args:
        client: OpenAI client configured to call Gemini
        model: Model name to use
        image_path: Path to the image file
        retries: Number of retry attempts for API failures
        delay: Delay between retries in seconds

    Returns:
        Extracted text from the image
    """
    # Encode the image to base64
    image_data = encode_image(image_path)

    for attempt in range(retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
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

            # # Sometimes the response might come back with markdown backticks, clean it up
            # if response_content.startswith("```json"):
            #     response_content = response_content.replace("```json", "", 1)
            #     response_content = response_content.replace("```", "", 1)
            # elif response_content.startswith("```"):
            #     response_content = response_content.replace("```", "", 2)

            # response_content = response_content.strip()

            # result = json.loads(response_content)

            # Print the category and tags if available

            if not result:
                return ""
            print(" Category: ", result.category)
            print(" Tags: ", result.tags)

            # Return the extracted text
            return result.markdown

        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

    # This should never be reached due to the raise in the loop
    return "maximum retries exceeded"


def generate_ground_truth(
    sample_dir: str = "dataset/sample/images",
    output_file: str = "dataset/sample/ground_truth_qwen32.json",
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
):
    """Generate ground truth for all images in the sample directory and save to a JSON file.

    Args:
        sample_dir: Directory containing image files
        output_file: Path to save the JSON output
        model: Gemini model to use

    Returns:
        Dictionary of image filenames to OCR results
    """
    # Get API key from environment variable
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # Create OpenAI client with Gemini compatibility
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    results = {}
    errors = []

    sample_path = pathlib.Path(sample_dir)
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    images = list(sample_path.glob("*.png"))
    total_images = len(images)

    print(f"Using model: {model} via OpenAI compatibility API")
    print(f"Found {total_images} images in {sample_dir}")
    print(f"Results will be saved to: {output_file}")

    # Process images with progress bar
    for image_path in tqdm(images, desc="Processing images"):
        try:
            ocr_text = generate_ocr_for_image(client, model=model, image_path=str(image_path))
            results[image_path.name] = ocr_text
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"\n❌ {error_msg}")
            results[image_path.name] = f"ERROR: {str(e)}"
            errors.append({"file": image_path.name, "error": str(e)})

    # Save results to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ground truth saved to {output_path}")

    # Print summary
    successful = total_images - len(errors)
    print(f"\nSummary: {successful}/{total_images} images processed successfully")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for err in errors:
            print(f"- {err['file']}: {err['error']}")

    return results


if __name__ == "__main__":
    # Uncomment to create sample dataset if needed
    # create_sample_dataset()

    # Generate ground truth for all images in sample directory
    generate_ground_truth(
        model="qwen/qwen2.5-vl-32b-instruct"  # "google/gemini-2.5-pro-preview-03-25"
    )  # qwen/qwen2.5-vl-72b-instruct")
