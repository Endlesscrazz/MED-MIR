#!/usr/bin/env python3
"""
process_images.py - Image Optimization Pipeline

Processes raw chest X-ray images for web delivery:
- Resizes to max 256px (height/width)
- Converts to WebP format for optimal compression
- Preserves aspect ratio

Usage:
    python process_images.py --input_dir data/raw/images --output_dir output/images

Output:
    output/images/
    └── *.webp  # Optimized images ready for web
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm
from PIL import Image


# Default configuration
DEFAULT_INPUT_DIR = "data/raw/images"
DEFAULT_OUTPUT_DIR = "output/images"
MAX_DIMENSION = 256
WEBP_QUALITY = 85
MAX_WORKERS = 4


def process_single_image(
    input_path: Path,
    output_dir: Path,
    max_dim: int = MAX_DIMENSION,
    quality: int = WEBP_QUALITY
) -> Optional[dict]:
    """
    Process a single image: resize and convert to WebP.
    
    Args:
        input_path: Path to input image
        output_dir: Directory for output
        max_dim: Maximum dimension (height or width)
        quality: WebP quality (0-100)
        
    Returns:
        Dictionary with processing results, or None on failure
    """
    try:
        # Open image
        with Image.open(input_path) as img:
            original_size = img.size
            
            # Convert to RGB if necessary (WebP doesn't support all modes)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            elif img.mode == "L":
                # Grayscale - convert to RGB for consistency
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if width > height:
                if width > max_dim:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_width, new_height = width, height
            else:
                if height > max_dim:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                else:
                    new_width, new_height = width, height
            
            # Resize using high-quality resampling
            if (new_width, new_height) != (width, height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save as WebP
            output_filename = f"{input_path.stem}.webp"
            output_path = output_dir / output_filename
            
            img.save(output_path, "WEBP", quality=quality, method=6)
            
            # Get file sizes
            original_size_bytes = input_path.stat().st_size
            new_size_bytes = output_path.stat().st_size
            
            return {
                "input": str(input_path.name),
                "output": str(output_filename),
                "original_dimensions": original_size,
                "new_dimensions": (new_width, new_height),
                "original_size_kb": original_size_bytes / 1024,
                "new_size_kb": new_size_bytes / 1024,
                "compression_ratio": original_size_bytes / new_size_bytes if new_size_bytes > 0 else 0
            }
            
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return None


def process_image_wrapper(args: tuple) -> Optional[dict]:
    """Wrapper for multiprocessing."""
    return process_single_image(*args)


def main():
    """Main entry point for image processing."""
    parser = argparse.ArgumentParser(
        description="Process images for web delivery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing raw images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--max_dim", "-m",
        type=int,
        default=MAX_DIMENSION,
        help="Maximum dimension (height or width)"
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=WEBP_QUALITY,
        help="WebP quality (0-100)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel workers"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Run download_data.py first to fetch the dataset.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print("=" * 60)
    print("Image Processing Pipeline")
    print("=" * 60)
    print(f"Input: {input_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Images found: {len(image_files)}")
    print(f"Target size: {args.max_dim}px max dimension")
    print(f"WebP quality: {args.quality}")
    print("=" * 60)
    
    # Process images in parallel
    results = []
    process_args = [
        (img_path, output_dir, args.max_dim, args.quality)
        for img_path in image_files
    ]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image_wrapper, arg) for arg in process_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
    
    # Calculate statistics
    if results:
        total_original = sum(r["original_size_kb"] for r in results)
        total_new = sum(r["new_size_kb"] for r in results)
        avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"Images processed: {len(results)}/{len(image_files)}")
        print(f"Total original size: {total_original/1024:.2f} MB")
        print(f"Total new size: {total_new/1024:.2f} MB")
        print(f"Average compression ratio: {avg_compression:.2f}x")
        print(f"Space saved: {(total_original - total_new)/1024:.2f} MB ({100*(1 - total_new/total_original):.1f}%)")
    else:
        print("No images were successfully processed.")


if __name__ == "__main__":
    main()
