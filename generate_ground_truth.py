#!/usr/bin/env python3
"""
Ground Truth Generator - Converts FUNSD annotations to OCR ground truth format
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ground truth from FUNSD annotations")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="dataset/sample/annotations",
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="dataset/sample/ground_truth.json",
        help="Output file for the ground truth data",
    )
    parser.add_argument(
        "--vertical-tolerance",
        type=int,
        default=15,
        help="Vertical distance tolerance for grouping text on the same line",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    return parser.parse_args()


def extract_text_from_annotation(annotation_file, vertical_tolerance=15, debug=False):
    """Extract text from FUNSD annotation file, preserving horizontal layout.

    Args:
        annotation_file: Path to the annotation JSON file
        vertical_tolerance: Max vertical distance to consider text on the same line
        debug: Enable debug output

    Returns:
        Extracted text with proper line breaks
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)

    # Extract text elements with their bounding boxes
    text_elements = []

    for item in data.get("form", []):
        # Skip empty items
        if not item.get("text", "").strip():
            continue

        # Get bounding box coordinates
        box = item.get("box", [0, 0, 0, 0])

        # Use item text if available
        text_elements.append(
            {"text": item["text"], "top": box[1], "left": box[0], "bottom": box[3], "right": box[2]}
        )

        # Also process individual words if they exist and don't match the parent text
        # This helps with cases where individual word positions are more accurate
        parent_text = item["text"].lower()
        for word in item.get("words", []):
            word_text = word.get("text", "").strip()
            if word_text and word_text.lower() not in parent_text:
                word_box = word.get("box", [0, 0, 0, 0])
                text_elements.append(
                    {
                        "text": word_text,
                        "top": word_box[1],
                        "left": word_box[0],
                        "bottom": word_box[3],
                        "right": word_box[2],
                    }
                )

    # Group text elements by vertical position
    # Use the vertical center of each box for comparison
    rows = defaultdict(list)

    for element in text_elements:
        center_y = (element["top"] + element["bottom"]) // 2

        # Find the closest existing row within tolerance
        assigned = False
        for row_y in list(rows.keys()):
            if abs(center_y - row_y) <= vertical_tolerance:
                rows[row_y].append(element)
                assigned = True
                break

        # Create a new row if no close match
        if not assigned:
            rows[center_y].append(element)

    # Sort rows by vertical position (top to bottom)
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])

    # Sort elements within each row by horizontal position (left to right)
    result_lines = []

    for row_y, elements in sorted_rows:
        # Sort elements by left coordinate
        sorted_elements = sorted(elements, key=lambda x: x["left"])

        # Join elements in this row with spaces
        row_text = " ".join(element["text"] for element in sorted_elements)
        result_lines.append(row_text)

        if debug:
            print(f"Row {row_y}: {row_text}")

    # Join all rows with newlines
    return "\n".join(result_lines)


def generate_ground_truth(annotations_dir, output_file, vertical_tolerance=15, debug=False):
    """Generate ground truth JSON from annotation files."""
    ground_truth = {}

    for file in os.listdir(annotations_dir):
        if file.endswith(".json"):
            base_name = os.path.splitext(file)[0]
            img_name = f"{base_name}.png"  # Assuming .png extension

            annotation_path = os.path.join(annotations_dir, file)

            if debug:
                print(f"Processing {file}...")

            extracted_text = extract_text_from_annotation(
                annotation_path, vertical_tolerance=vertical_tolerance, debug=debug
            )

            ground_truth[img_name] = extracted_text

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save ground truth to JSON file
    with open(output_file, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated ground truth for {len(ground_truth)} images")
    print(f"Saved to {output_file}")


def main():
    args = parse_args()
    generate_ground_truth(
        args.annotations_dir,
        args.output_file,
        vertical_tolerance=args.vertical_tolerance,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
