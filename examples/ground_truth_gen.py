# To run this code you need to install the following dependencies:
# pip install google-genai python-dotenv tqdm

import base64
import json
import os
import pathlib
import shutil
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()


class OCRResult(BaseModel):
    markdown: str
    category: str = Field(description="The category of the document")
    tags: list[str] = Field(description="The tags of the document")


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


def generate_ocr_for_image(
    client, model: str, image_path: str, retries: int = 3, delay: int = 2
) -> str:
    """Generate OCR for a single image using Gemini.

    Args:
        client: Gemini API client
        model: Model name to use
        image_path: Path to the image file
        retries: Number of retry attempts for API failures
        delay: Delay between retries in seconds

    Returns:
        Extracted text from the image
    """
    image_bytes = pathlib.Path(image_path).read_bytes()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Extract all text from this image with proper formatting."
                ),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text="""Generate OCRs with Markdowns and correctly formatted layout when possible"""
            ),
        ],
        response_schema=OCRResult,
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            ocr_result: OCRResult = response.parsed
            print(response.text)
            print(f"document category: {ocr_result.category}")
            print(f"document tags: {ocr_result.tags}")
            print(f"document markdown: {ocr_result.markdown}")
            print("-" * 100)
            return ocr_result.markdown
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

    # This should never be reached due to the raise in the loop
    return "ERROR: Maximum retries exceeded"


def generate_ground_truth(
    sample_dir: str = "dataset/sample/images",
    output_file: str = "dataset/sample/ground_truth.json",
    model: str = "gemini-2.5-flash-preview-04-17",
):
    """Generate ground truth for all images in the sample directory and save to a JSON file.

    Args:
        sample_dir: Directory containing image files
        output_file: Path to save the JSON output
        model: Gemini model to use

    Returns:
        Dictionary of image filenames to OCR results
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    results = {}
    errors = []

    sample_path = pathlib.Path(sample_dir)
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    images = list(sample_path.glob("*.png"))
    total_images = len(images)

    print(f"Using model: {model}")
    print(f"Found {total_images} images in {sample_dir}")
    print(f"Results will be saved to: {output_file}")

    # Process images with progress bar
    for image_path in tqdm(images, desc="Processing images"):
        try:
            ocr_text = generate_ocr_for_image(client, model, str(image_path))
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
    generate_ground_truth(model="gemini-2.5-flash-preview-04-17")
