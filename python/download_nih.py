#!/usr/bin/env python3
"""
download_nih.py — NIH ChestX-ray14 Dataset Downloader & Subset Selector

Two-step workflow designed for limited MacBook storage:

  Step 1 (external HDD):
    python download_nih.py --kaggle --output_dir /Volumes/MyDrive/nih-data

  Step 2 (MacBook):
    python download_nih.py --select-subset \\
      --source_dir /Volumes/MyDrive/nih-data \\
      --output_dir data/nih \\
      --per_label 150

Step 1 downloads the full NIH ChestX-ray14 dataset (~45 GB) via the
Kaggle CLI to an external drive.  Step 2 reads the CSV label file,
selects a class-balanced subset, and copies only those images to the
local project directory.

The 15 labels in NIH ChestX-ray14:
  Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
  Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding,
  Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

Output (after --select-subset):
    data/nih/
    ├── images/          # PNG chest X-ray images (balanced subset)
    └── labels.csv       # filename, labels (pipe-separated), primary_label
"""

import os
import csv
import sys
import glob
import shutil
import random
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional

from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────
KAGGLE_DATASET = "nih-chest-xrays/data"
NIH_CSV_NAME = "Data_Entry_2017_v2020.csv"
# Older versions of the dataset use this name
NIH_CSV_NAME_ALT = "Data_Entry_2017.csv"

DEFAULT_OUTPUT = "data/nih"
DEFAULT_PER_LABEL = 150

# The 15 official NIH ChestX-ray14 labels (alphabetical)
NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Display-friendly names (used by generate_index.py and the web app)
NIH_DISPLAY_NAMES: dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "No Finding": "Normal",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — Kaggle Download
# ══════════════════════════════════════════════════════════════════

def kaggle_download(output_dir: Path) -> None:
    """
    Download the full NIH ChestX-ray14 dataset via the Kaggle CLI.

    Requires:
      - ``kaggle`` Python package (``pip install kaggle``)
      - Kaggle API token at ``~/.kaggle/kaggle.json``

    The download is ~45 GB and will extract to ``output_dir``.

    Args:
        output_dir: Directory to download and extract the dataset into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify kaggle CLI is available
    try:
        subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, check=True, text=True,
        )
    except FileNotFoundError:
        print("Error: 'kaggle' CLI not found.")
        print("Install it with:  pip install kaggle")
        print("Then place your API token at:  ~/.kaggle/kaggle.json")
        print("  → https://www.kaggle.com/docs/api#authentication")
        sys.exit(1)

    print("=" * 60)
    print("NIH ChestX-ray14 — Kaggle Download")
    print("=" * 60)
    print(f"Dataset:  {KAGGLE_DATASET}")
    print(f"Target:   {output_dir.absolute()}")
    print(f"Size:     ~45 GB (will extract automatically)")
    print("=" * 60)
    print()

    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(output_dir),
        "--unzip",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("This will take a while — go grab a coffee ☕\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\nError: Kaggle download failed (exit code {result.returncode}).")
        print("Check your Kaggle credentials and internet connection.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

    # Verify the CSV exists
    csv_path = _find_csv(output_dir)
    if csv_path:
        print(f"  ✓ Found label file: {csv_path}")
        num_rows = sum(1 for _ in open(csv_path)) - 1
        print(f"  ✓ Total entries: {num_rows:,}")
    else:
        print("  ⚠ Could not find Data_Entry CSV — check the extracted files.")

    # Count images
    image_dirs = list(output_dir.glob("images_*/images"))
    if not image_dirs:
        image_dirs = [output_dir]
    total_images = sum(
        len(list(d.glob("*.png")))
        for d in image_dirs
    )
    print(f"  ✓ Total images found: {total_images:,}")
    print()
    print("Next step:")
    print(f"  python download_nih.py --select-subset \\")
    print(f"    --source_dir {output_dir} \\")
    print(f"    --output_dir data/nih \\")
    print(f"    --per_label 150")


def _find_csv(base_dir: Path) -> Optional[Path]:
    """
    Locate the NIH label CSV file in the extracted dataset.

    The Kaggle download may place the CSV at the root or inside
    a subdirectory.  This function searches common locations.

    Args:
        base_dir: Root of the extracted Kaggle dataset.

    Returns:
        Path to the CSV, or None if not found.
    """
    candidates = [
        base_dir / NIH_CSV_NAME,
        base_dir / NIH_CSV_NAME_ALT,
    ]
    # Also search one level deep
    for child in base_dir.iterdir():
        if child.is_dir():
            candidates.append(child / NIH_CSV_NAME)
            candidates.append(child / NIH_CSV_NAME_ALT)

    # Glob search as fallback
    for pattern in ["**/Data_Entry*.csv"]:
        for match in base_dir.glob(pattern):
            candidates.append(match)

    for c in candidates:
        if c.exists():
            return c
    return None


def _find_image(base_dir: Path, filename: str) -> Optional[Path]:
    """
    Locate a single image file in the extracted dataset.

    NIH images are spread across subdirectories like
    ``images_001/images/``, ``images_002/images/``, etc.

    Args:
        base_dir: Root of the extracted Kaggle dataset.
        filename: Image filename, e.g. ``00000001_000.png``

    Returns:
        Full path to the image, or None if not found.
    """
    # Direct check
    direct = base_dir / filename
    if direct.exists():
        return direct

    # Check in images/ subdirectory
    in_images = base_dir / "images" / filename
    if in_images.exists():
        return in_images

    # Check in images_XXX/images/ subdirectories
    for subdir in sorted(base_dir.glob("images_*/images")):
        candidate = subdir / filename
        if candidate.exists():
            return candidate

    return None


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — Select Balanced Subset
# ══════════════════════════════════════════════════════════════════

def parse_nih_csv(csv_path: Path) -> list[dict]:
    """
    Parse the NIH Data_Entry CSV into a list of records.

    Each record contains:
      - filename:      e.g. ``00000001_000.png``
      - labels:        list of label strings (multi-label)
      - primary_label: the first label
      - patient_id:    for avoiding patient leakage (optional)

    Args:
        csv_path: Path to Data_Entry_2017_v2020.csv

    Returns:
        List of parsed records.
    """
    records: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Image Index", "").strip()
            finding_labels = row.get("Finding Labels", "").strip()

            if not filename or not finding_labels:
                continue

            labels = [l.strip() for l in finding_labels.split("|") if l.strip()]
            if not labels:
                continue

            records.append({
                "filename": filename,
                "labels": labels,
                "primary_label": labels[0],
                "patient_id": row.get("Patient ID", ""),
            })

    return records


def select_balanced_subset(
    records: list[dict],
    per_label: int = DEFAULT_PER_LABEL,
    seed: int = 42,
) -> list[dict]:
    """
    Select a class-balanced subset from the parsed CSV records.

    Strategy:
      1. Group images by each label they contain (multi-label aware).
      2. For each label, randomly sample up to ``per_label`` images.
      3. De-duplicate (an image may be sampled for multiple labels).
      4. Return the final list.

    This ensures every label has representation while keeping the
    total dataset size manageable.

    Args:
        records: Parsed CSV records from ``parse_nih_csv``.
        per_label: Target number of images per label.
        seed: Random seed for reproducibility.

    Returns:
        De-duplicated list of selected records.
    """
    random.seed(seed)

    # Group records by label
    label_to_records: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        for label in rec["labels"]:
            label_to_records[label].append(rec)

    # Sample per label
    selected_filenames: set[str] = set()
    selected_map: dict[str, dict] = {}

    for label in NIH_LABELS:
        pool = label_to_records.get(label, [])
        # Filter out already-selected to spread diversity
        unselected = [r for r in pool if r["filename"] not in selected_filenames]
        # If not enough unselected, allow duplicates from the pool
        if len(unselected) < per_label:
            sample_pool = unselected + [
                r for r in pool if r["filename"] not in selected_filenames
            ]
        else:
            sample_pool = unselected

        n = min(per_label, len(sample_pool))
        sampled = random.sample(sample_pool, n)

        for rec in sampled:
            selected_filenames.add(rec["filename"])
            selected_map[rec["filename"]] = rec

    return list(selected_map.values())


def select_subset(
    source_dir: Path,
    output_dir: Path,
    per_label: int = DEFAULT_PER_LABEL,
) -> None:
    """
    Read the NIH CSV, select a balanced subset, and copy images.

    Args:
        source_dir: Root of the extracted Kaggle dataset (e.g., external HDD).
        output_dir: Local project directory (e.g., ``data/nih``).
        per_label:  Maximum images per label.
    """
    # Find CSV
    csv_path = _find_csv(source_dir)
    if not csv_path:
        print(f"Error: Cannot find {NIH_CSV_NAME} in {source_dir}")
        print("Make sure you ran --kaggle first and the dataset is extracted.")
        sys.exit(1)

    print("=" * 60)
    print("NIH ChestX-ray14 — Balanced Subset Selection")
    print("=" * 60)
    print(f"Source:     {source_dir.absolute()}")
    print(f"Output:     {output_dir.absolute()}")
    print(f"Per label:  {per_label}")
    print(f"CSV:        {csv_path}")
    print("=" * 60)

    # Parse CSV
    print("\nParsing label CSV...")
    records = parse_nih_csv(csv_path)
    print(f"  Total records: {len(records):,}")

    # Show full dataset distribution
    full_dist: dict[str, int] = defaultdict(int)
    for rec in records:
        for lbl in rec["labels"]:
            full_dist[lbl] += 1
    print("\n  Full dataset label distribution:")
    for lbl in sorted(full_dist, key=lambda x: -full_dist[x]):
        print(f"    {lbl:25s} {full_dist[lbl]:>6,}")

    # Select balanced subset
    print(f"\nSelecting balanced subset ({per_label}/label)...")
    selected = select_balanced_subset(records, per_label)
    print(f"  Selected: {len(selected)} unique images")

    # Show subset distribution
    subset_dist: dict[str, int] = defaultdict(int)
    for rec in selected:
        for lbl in rec["labels"]:
            subset_dist[lbl] += 1
    print("\n  Subset label distribution:")
    for lbl in sorted(subset_dist, key=lambda x: -subset_dist[x]):
        count = subset_dist[lbl]
        bar = "█" * min(count // 3, 40)
        status = "✓" if count >= per_label else "○"
        print(f"    {status} {lbl:25s} {count:>5}  {bar}")

    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    print(f"\nCopying {len(selected)} images to {images_dir}...")
    copied = 0
    missing = 0

    for rec in tqdm(selected, desc="Copying images"):
        src = _find_image(source_dir, rec["filename"])
        if src:
            dst = images_dir / rec["filename"]
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            copied += 1
        else:
            missing += 1
            if missing <= 5:
                print(f"  ⚠ Image not found: {rec['filename']}")

    if missing > 5:
        print(f"  ⚠ ... and {missing - 5} more missing images")

    # Save labels CSV
    csv_out = output_dir / "labels.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "labels", "primary_label"]
        )
        writer.writeheader()
        for rec in selected:
            writer.writerow({
                "filename": rec["filename"],
                "labels": "|".join(rec["labels"]),
                "primary_label": rec["primary_label"],
            })

    print("\n" + "=" * 60)
    print("Subset Selection Complete!")
    print("=" * 60)
    print(f"  Images copied:  {copied}")
    print(f"  Images missing: {missing}")
    print(f"  Labels CSV:     {csv_out}")
    print(f"  Images dir:     {images_dir}")
    print()
    print("Next steps:")
    print(f"  python process_images.py -i {output_dir}/images -o output/images")
    print(f"  python generate_index.py --images_dir output/images --nih_labels {csv_out} --output_dir output")
    print(f"  python evaluate.py --embeddings output/embeddings.bin --metadata output/metadata.json --output_dir output")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    """Main entry point with subcommand-style arguments."""
    parser = argparse.ArgumentParser(
        description="NIH ChestX-ray14 Dataset Downloader & Subset Selector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Download full dataset to external drive
  python download_nih.py --kaggle --output_dir /Volumes/MyDrive/nih-data

  # Step 2: Select balanced subset and copy to project
  python download_nih.py --select-subset \\
    --source_dir /Volumes/MyDrive/nih-data \\
    --output_dir data/nih \\
    --per_label 150
""",
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--kaggle",
        action="store_true",
        help="Download full dataset via Kaggle CLI",
    )
    mode.add_argument(
        "--select-subset",
        action="store_true",
        help="Select a balanced subset from an already-downloaded dataset",
    )

    # Shared arguments
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory (download target or subset destination)",
    )

    # --select-subset specific
    parser.add_argument(
        "--source_dir", "-s",
        type=str,
        default=None,
        help="[select-subset] Path to extracted Kaggle dataset (e.g., external HDD)",
    )
    parser.add_argument(
        "--per_label", "-n",
        type=int,
        default=DEFAULT_PER_LABEL,
        help="[select-subset] Maximum images per label",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.kaggle:
        kaggle_download(output_dir)

    elif args.select_subset:
        if not args.source_dir:
            print("Error: --source_dir is required with --select-subset")
            print("Example: --source_dir /Volumes/MyDrive/nih-data")
            sys.exit(1)
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"Error: Source directory does not exist: {source_dir}")
            sys.exit(1)
        select_subset(source_dir, output_dir, args.per_label)


if __name__ == "__main__":
    main()


"""
python download_nih.py --select-subset \
  --source_dir /Volumes/One_Touch/Med-Mir-proj/NIH-dataset \
  --output_dir data/nih \
  --per_label 150
"""