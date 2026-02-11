#!/usr/bin/env python3
"""
download_data.py - Open-I Chest X-ray Dataset Downloader

Downloads a subset of the Indiana University Chest X-ray dataset from Open-I.
This script fetches both images and their associated radiology reports.

Usage:
    python download_data.py --num_images 500 --output_dir data/raw

Output Structure:
    data/raw/
    ├── images/          # PNG chest X-ray images
    └── reports/         # XML report files with findings
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET


# Open-I API Configuration
OPENI_API_BASE = "https://openi.nlm.nih.gov/api/search"
OPENI_IMAGE_BASE = "https://openi.nlm.nih.gov/imgs"

# Default parameters
DEFAULT_NUM_IMAGES = 500
DEFAULT_OUTPUT_DIR = "data/raw"
MAX_WORKERS = 8


def fetch_search_results(query: str, num_results: int) -> list[dict]:
    """
    Fetch search results from Open-I API.
    
    Args:
        query: Search query (e.g., "chest x-ray")
        num_results: Number of results to fetch
        
    Returns:
        List of result dictionaries containing image metadata
    """
    results = []
    page_size = 100  # API max per request
    
    print(f"Fetching {num_results} results from Open-I API...")
    
    for offset in tqdm(range(0, num_results, page_size), desc="Fetching pages"):
        params = {
            "query": query,
            "m": min(page_size, num_results - offset),
            "n": offset + 1,  # 1-indexed
            "coll": "iu",  # Indiana University collection
        }
        
        try:
            response = requests.get(OPENI_API_BASE, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "list" in data:
                results.extend(data["list"])
                
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch page at offset {offset}: {e}")
            continue
    
    return results[:num_results]


def download_image(item: dict, output_dir: Path) -> Optional[dict]:
    """
    Download a single image and extract its metadata.
    
    Args:
        item: Dictionary containing image metadata from Open-I API
        output_dir: Directory to save the image
        
    Returns:
        Metadata dictionary if successful, None otherwise
    """
    try:
        # Extract image URL - Open-I provides relative paths
        img_url = item.get("imgLarge") or item.get("imgThumb")
        if not img_url:
            return None
            
        # Construct full URL
        if not img_url.startswith("http"):
            img_url = f"{OPENI_IMAGE_BASE}/{img_url}"
        
        # Generate unique ID from URL
        img_id = Path(img_url).stem
        img_filename = f"{img_id}.png"
        img_path = output_dir / "images" / img_filename
        
        # Skip if already downloaded
        if img_path.exists():
            pass  # Still return metadata
        else:
            # Download image
            response = requests.get(img_url, timeout=30)
            response.raise_for_status()
            
            img_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(response.content)
        
        # Extract metadata
        metadata = {
            "id": img_id,
            "filename": img_filename,
            "url": str(img_path.relative_to(output_dir.parent.parent) if output_dir.parent.parent.exists() else img_path),
            "report_snippet": extract_findings(item),
            "mesh_terms": item.get("MeSH", []),
            "problems": item.get("Problems", ""),
            "modality": item.get("modality", "X-ray"),
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Failed to download image: {e}")
        return None


def extract_findings(item: dict) -> str:
    """
    Extract clinical findings from Open-I item.
    
    Args:
        item: Dictionary containing image metadata
        
    Returns:
        Concatenated findings string
    """
    findings = []
    
    # Try different fields that might contain findings
    if item.get("abstract"):
        findings.append(item["abstract"])
    if item.get("Problems"):
        findings.append(str(item["Problems"]))
    if item.get("findings"):
        findings.append(item["findings"])
    if item.get("impression"):
        findings.append(item["impression"])
        
    # Join and truncate
    text = " ".join(findings).strip()
    
    # Truncate to reasonable length for display
    if len(text) > 500:
        text = text[:497] + "..."
        
    return text if text else "No findings available"


def derive_ground_truth_label(metadata: dict) -> str:
    """
    Derive a ground truth label from MeSH terms or problems.
    
    Args:
        metadata: Dictionary containing image metadata
        
    Returns:
        Primary diagnosis label
    """
    # Priority order for labels
    priority_labels = [
        "pneumonia", "effusion", "cardiomegaly", "atelectasis",
        "pneumothorax", "edema", "consolidation", "mass", "nodule",
        "emphysema", "fibrosis", "fracture", "normal"
    ]
    
    # Combine all text for searching
    search_text = " ".join([
        str(metadata.get("mesh_terms", [])),
        metadata.get("problems", ""),
        metadata.get("report_snippet", "")
    ]).lower()
    
    for label in priority_labels:
        if label in search_text:
            return label.capitalize()
    
    return "Other"


def main():
    """Main entry point for dataset download."""
    parser = argparse.ArgumentParser(
        description="Download Open-I Chest X-ray Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_images", "-n",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of images to download"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="chest xray",
        help="Search query for Open-I"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel download workers"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Open-I Chest X-ray Dataset Downloader")
    print("=" * 60)
    print(f"Target: {args.num_images} images")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)
    
    # Fetch search results
    results = fetch_search_results(args.query, args.num_images)
    print(f"Found {len(results)} results from API")
    
    if not results:
        print("Error: No results found. Check your internet connection.")
        sys.exit(1)
    
    # Download images in parallel
    metadata_list = []
    print(f"\nDownloading images with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, item, output_dir): item 
            for item in results
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            metadata = future.result()
            if metadata:
                # Add ground truth label
                metadata["ground_truth_label"] = derive_ground_truth_label(metadata)
                metadata_list.append(metadata)
    
    # Save metadata
    metadata_path = output_dir / "raw_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Images downloaded: {len(metadata_list)}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print label distribution
    label_counts: dict[str, int] = {}
    for m in metadata_list:
        label = m.get("ground_truth_label", "Unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
