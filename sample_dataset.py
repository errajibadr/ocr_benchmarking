#!/usr/bin/env python3
"""
Dataset Sampling Script - Creates reproducible subsets of the FUNSD dataset
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a sample subset of the dataset")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="dataset/testing_data",
        help="Source directory containing images and annotations",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="dataset/sample",
        help="Destination directory for the sample subset",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10, help="Number of images to include in the sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def create_sample(source_dir, dest_dir, sample_size, seed):
    # Set random seed for reproducibility
    random.seed(seed)

    # Create directories if they don't exist
    img_src_dir = os.path.join(source_dir, "images")
    ann_src_dir = os.path.join(source_dir, "annotations")

    img_dest_dir = os.path.join(dest_dir, "images")
    ann_dest_dir = os.path.join(dest_dir, "annotations")

    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(ann_dest_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(img_src_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Sample the files
    if sample_size > len(image_files):
        print(
            f"Warning: Requested sample size {sample_size} is larger than available images {len(image_files)}"
        )
        sample_size = len(image_files)

    sampled_files = random.sample(image_files, sample_size)

    # Copy images and annotation files
    for img_file in sampled_files:
        # Image filename without extension
        base_name = os.path.splitext(img_file)[0]

        # Copy image
        shutil.copy2(os.path.join(img_src_dir, img_file), os.path.join(img_dest_dir, img_file))

        # Copy annotation if it exists
        annotation_file = f"{base_name}.json"
        ann_src_path = os.path.join(ann_src_dir, annotation_file)
        if os.path.exists(ann_src_path):
            shutil.copy2(ann_src_path, os.path.join(ann_dest_dir, annotation_file))
        else:
            print(f"Warning: Annotation file {annotation_file} not found")

    print(f"Created sample dataset with {len(sampled_files)} files")
    print(f"Files: {sampled_files}")

    # Save sample info for reproducibility
    sample_info = {"seed": seed, "sample_size": sample_size, "files": sampled_files}

    with open(os.path.join(dest_dir, "sample_info.json"), "w") as f:
        json.dump(sample_info, f, indent=2)


def main():
    args = parse_args()
    create_sample(args.source_dir, args.dest_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
