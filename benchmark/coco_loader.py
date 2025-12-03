#!/usr/bin/env python3
"""
COCO Dataset Loader and Subset Preparation

Utilities for downloading and preparing COCO validation images for benchmarking.
Supports downloading specific images by ID and creating annotation subsets.

Usage:
    python3 coco_loader.py --num-images 300 --output-dir data
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

from pycocotools.coco import COCO


class COCOLoader:
    """Handles COCO dataset downloading and subset preparation."""

    COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"
    ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    def __init__(self, annotation_file: str, image_dir: Optional[str] = None):
        """
        Initialize with COCO annotations.

        Args:
            annotation_file: Path to instances_val2017.json
            image_dir: Directory to store downloaded images
        """
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir) if image_dir else self.annotation_file.parent / "images"
        self.coco: Optional[COCO] = None

        if self.annotation_file.exists():
            print(f"Loading COCO annotations from {self.annotation_file}...")
            self.coco = COCO(str(self.annotation_file))

    def get_image_ids(self, num_images: Optional[int] = None) -> List[int]:
        """Get list of image IDs from COCO dataset."""
        if self.coco is None:
            raise ValueError("COCO annotations not loaded")

        image_ids = self.coco.getImgIds()
        if num_images:
            return image_ids[:num_images]
        return image_ids

    def download_image(self, image_id: int, force: bool = False) -> Path:
        """
        Download a single COCO image by ID.

        Args:
            image_id: COCO image ID
            force: Re-download even if file exists

        Returns:
            Path to downloaded image
        """
        filename = f"{image_id:012d}.jpg"
        output_path = self.image_dir / filename

        if output_path.exists() and not force:
            return output_path

        self.image_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.COCO_IMAGE_BASE_URL}/{filename}"

        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

        return output_path

    def download_images(
        self,
        num_images: int,
        force: bool = False,
        progress_interval: int = 50
    ) -> List[Path]:
        """
        Download multiple COCO images.

        Args:
            num_images: Number of images to download
            force: Re-download even if files exist
            progress_interval: Print progress every N images

        Returns:
            List of paths to downloaded images
        """
        image_ids = self.get_image_ids(num_images)
        downloaded = []

        print(f"Downloading {len(image_ids)} COCO images...")

        for i, img_id in enumerate(image_ids):
            try:
                path = self.download_image(img_id, force=force)
                downloaded.append(path)

                if (i + 1) % progress_interval == 0:
                    print(f"  Downloaded {i + 1}/{len(image_ids)} images...")

            except Exception as e:
                print(f"  Warning: Failed to download image {img_id}: {e}")

        print(f"Downloaded {len(downloaded)}/{len(image_ids)} images")
        return downloaded

    def create_annotation_subset(
        self,
        image_ids: List[int],
        output_file: str
    ) -> Path:
        """
        Create a COCO annotation file containing only specified images.

        Args:
            image_ids: List of image IDs to include
            output_file: Path for output JSON file

        Returns:
            Path to created annotation file
        """
        if self.coco is None:
            raise ValueError("COCO annotations not loaded")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get image info and annotations
        images = self.coco.loadImgs(image_ids)
        ann_ids = self.coco.getAnnIds(imgIds=image_ids)
        annotations = self.coco.loadAnns(ann_ids)

        # Get category info
        cat_ids = list(set(ann['category_id'] for ann in annotations))
        categories = self.coco.loadCats(cat_ids)

        # Create subset annotation file
        subset = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

        with open(output_path, 'w') as f:
            json.dump(subset, f)

        print(f"Created annotation subset: {output_path}")
        print(f"  Images: {len(images)}, Annotations: {len(annotations)}")

        return output_path

    def get_image_stats(self, image_ids: Optional[List[int]] = None) -> dict:
        """
        Get statistics about images and annotations.

        Args:
            image_ids: Optional subset of image IDs (uses all if None)

        Returns:
            Dictionary with image and annotation statistics
        """
        if self.coco is None:
            raise ValueError("COCO annotations not loaded")

        if image_ids is None:
            image_ids = self.coco.getImgIds()

        ann_ids = self.coco.getAnnIds(imgIds=image_ids)
        annotations = self.coco.loadAnns(ann_ids)

        # Count annotations per category
        category_counts = {}
        for ann in annotations:
            cat_id = ann['category_id']
            cat_info = self.coco.loadCats([cat_id])[0]
            cat_name = cat_info['name']
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        return {
            'num_images': len(image_ids),
            'num_annotations': len(annotations),
            'category_counts': category_counts,
        }


def main():
    parser = argparse.ArgumentParser(description='Download COCO validation images')
    parser.add_argument('--num-images', type=int, default=300,
                        help='Number of images to download (default: 300)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for images (default: data)')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to COCO annotations JSON')
    parser.add_argument('--force', action='store_true',
                        help='Re-download existing images')
    parser.add_argument('--stats-only', action='store_true',
                        help='Print statistics without downloading')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir

    # Find or specify annotation file
    ann_file = args.annotations or str(output_dir / 'instances_val2017.json')

    if not Path(ann_file).exists():
        print(f"Error: Annotation file not found: {ann_file}")
        print("Download annotations from: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        sys.exit(1)

    loader = COCOLoader(ann_file, str(output_dir / 'images'))

    if args.stats_only:
        image_ids = loader.get_image_ids(args.num_images)
        stats = loader.get_image_stats(image_ids)
        print(f"\nCOCO Subset Statistics ({args.num_images} images):")
        print(f"  Total annotations: {stats['num_annotations']}")
        print(f"  Categories ({len(stats['category_counts'])}):")
        for cat, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1])[:10]:
            print(f"    {cat}: {count}")
    else:
        paths = loader.download_images(args.num_images, force=args.force)
        print(f"\nDownloaded {len(paths)} images to {output_dir / 'images'}")


if __name__ == '__main__':
    main()
