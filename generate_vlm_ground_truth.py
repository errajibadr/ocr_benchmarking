#!/usr/bin/env python3
"""
VLM Ground Truth Generator - Creates ground truth from Vision Language Models via OpenRouter
"""

import argparse
import base64
import json
import os
import pathlib
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


class OCRResult(BaseModel):
    """Schema for OCR result from VLM models"""

    markdown: str = Field(description="The extracted text from the image with proper formatting")
    category: str = Field(
        description="The category of the document (e.g., invoice, receipt, form, letter, article)"
    )
    tags: List[str] = Field(description="The tags relevant to the document content")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate ground truth using VLM models via OpenRouter"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="dataset/sample_test/images",
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="dataset/sample_test/ground_truth_vlm.json",
        help="Output file for the ground truth data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-preview-04-17",
        help="VLM model to use via OpenRouter",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retry attempts for API failures"
    )
    parser.add_argument("--delay", type=int, default=2, help="Delay between retries in seconds")
    return parser.parse_args()


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_ocr_for_image(
    client: OpenAI, model: str, image_path: str, retries: int = 3, delay: int = 2
) -> str:
    """Generate OCR for a single image using VLM through OpenRouter API.

    Args:
        client: OpenAI client configured for OpenRouter
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

            # Get the extracted text
            result = response.choices[0].message.parsed

            # Print the category and tags if available
            if hasattr(result, "category"):
                print(f"  Category: {result.category}")
            if hasattr(result, "tags"):
                print(f"  Tags: {', '.join(result.tags)}")

            # Return the extracted text
            return result.markdown if hasattr(result, "markdown") else ""

        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"  All {retries} attempts failed: {str(e)}")
                return f"ERROR: {str(e)}"

    # This should never be reached due to the return in the loop
    return "ERROR: Maximum retries exceeded"


def generate_vlm_ground_truth(
    image_dir: str,
    output_file: str,
    model: str,
    retries: int = 3,
    delay: int = 2,
) -> Dict[str, str]:
    """Generate ground truth for all images in the directory using VLM models via OpenRouter.

    Args:
        image_dir: Directory containing image files
        output_file: Path to save the JSON output
        model: VLM model to use via OpenRouter
        retries: Number of retry attempts for API failures
        delay: Delay between retries in seconds

    Returns:
        Dictionary of image filenames to OCR results
    """
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not found. "
            "Please set it in your .env file or directly in your environment."
        )

    # Create OpenAI client with OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    results = {}
    errors = []

    image_path = pathlib.Path(image_dir)
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    images = list(image_path.glob("*.png"))
    total_images = len(images)

    print(f"Using model: {model} via OpenRouter")
    print(f"Found {total_images} images in {image_dir}")
    print(f"Results will be saved to: {output_file}")

    # Process images with progress bar
    for image_path in tqdm(images, desc="Processing images"):
        print(f"\nProcessing: {image_path.name}")
        try:
            ocr_text = generate_ocr_for_image(
                client, model, str(image_path), retries=retries, delay=delay
            )

            if ocr_text.startswith("ERROR:"):
                print(f"❌ Failed to process {image_path.name}")
                errors.append({"file": image_path.name, "error": ocr_text})
            else:
                print(f"✓ Successfully processed {image_path.name}")

            results[image_path.name] = ocr_text

        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"❌ {error_msg}")
            results[image_path.name] = f"ERROR: {str(e)}"
            errors.append({"file": image_path.name, "error": str(e)})

    # Save results to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ VLM ground truth saved to {output_path}")

    # Print summary
    successful = total_images - len(errors)
    print(f"\nSummary: {successful}/{total_images} images processed successfully")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for err in errors:
            print(f"- {err['file']}: {err['error']}")

    return results


def main():
    """Main entry point"""
    args = parse_args()
    generate_vlm_ground_truth(
        image_dir=args.image_dir,
        output_file=args.output_file,
        model=args.model,
        retries=args.retries,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
