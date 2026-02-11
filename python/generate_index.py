#!/usr/bin/env python3
"""
generate_index.py - Embedding Generation Pipeline

Generates the search index for Med-MIR using BiomedCLIP:
- Loads processed images
- Generates 768-dimensional embeddings using BiomedCLIP visual encoder
- L2-normalizes all vectors (critical for dot product similarity)
- Saves embeddings as binary Float32Array
- Generates metadata.json with image information
- Pre-computes fallback results for common queries
- Computes nearest neighbors for "Find Similar" feature

Usage:
    python generate_index.py --images_dir output/images --output_dir output

Output:
    output/
    ├── embeddings.bin          # Float32Array binary (N x 768)
    ├── metadata.json           # Array of image metadata
    ├── fallback_results.json   # Pre-computed results for common queries
    └── nearest_neighbors.json  # Pre-computed similar images
"""

import os
import csv
import json
import struct
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from open_clip import create_model_and_transforms, get_tokenizer


# Model configuration
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512  # BiomedCLIP uses 512-dim embeddings

# ── NIH ChestX-ray14 label helpers ────────────────────────────────
# Map from NIH's label strings to user-friendly display names
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

# Common clinical queries for fallback pre-computation
COMMON_QUERIES = [
    "chest xray",
    "normal chest xray",
    "abnormal chest xray",
    "xray",
    "lung",
    "heart",
    "pneumonia",
    "pleural effusion",
    "cardiomegaly",
    "atelectasis",
    "pulmonary edema",
    "pneumothorax",
    "consolidation",
    "lung nodule",
    "lung mass",
    "emphysema",
    "pulmonary fibrosis",
    "rib fracture",
    "mediastinal widening",
    "hilar enlargement",
    "infiltrate",
    "opacity",
    "costophrenic angle blunting",
    "interstitial pattern",
    "airspace disease",
    "bronchiectasis",
    "calcification",
    "granuloma",
    "tuberculosis",
    "sarcoidosis",
    "lung cancer",
    "metastasis",
    "heart failure",
    "pericardial effusion",
    "aortic aneurysm",
    "diaphragm elevation",
    "scoliosis",
    "kyphosis",
    "osteoporosis",
    "degenerative changes",
    "surgical clips",
    "pacemaker",
    "central line",
    "endotracheal tube",
    "nasogastric tube",
    "chest tube",
    "pleural thickening",
    "scarring",
    "linear atelectasis",
    "hyperinflation",
    "flattened diaphragm",
    "bilateral infiltrates",
    "unilateral opacity",
    "lobar pneumonia",
    "aspiration",
]

# Default configuration
DEFAULT_IMAGES_DIR = "output/images"
DEFAULT_RAW_METADATA = "data/raw/raw_metadata.json"
DEFAULT_NIH_LABELS = "data/nih/labels.csv"
DEFAULT_VERIFIED_LABELS = "data/verified_labels.csv"
DEFAULT_OUTPUT_DIR = "output"
TOP_K = 10
NEAREST_NEIGHBORS_K = 10


def normalize_text(value: str) -> str:
    """
    Normalize text for label extraction by lowercasing and collapsing whitespace.
    
    Args:
        value: Raw text value
        
    Returns:
        Normalized string suitable for keyword matching
    """
    return " ".join(value.lower().strip().split())


def derive_label_from_text(text: str) -> str:
    """
    Derive a clinical label from unstructured text.

    The function checks for "normal" signals first (because many reports
    that mention other keywords also state "no acute disease"), then
    walks through pathology keywords ordered from most specific to
    least specific so that, e.g., "pleural effusion" is preferred
    over the generic "effusion".

    Args:
        text: Combined report snippet, problems, and MeSH text

    Returns:
        Derived label or "Unknown" if no match is found
    """
    normalized = normalize_text(text)

    # ── Normal signals (highest priority) ─────────────────────────
    # Many Open-I reports contain only "no acute cardiopulmonary
    # process" and nothing else.  Catch these first.
    normal_signals = [
        "no evidence of acute cardiopulmonary process",
        "no acute cardiopulmonary process",
        "no acute cardiopulmonary disease",
        "no acute cardiopulmonary abnormality",
        "no acute disease",
        "no acute findings",
        "no acute abnormality",
        "no significant abnormality",
        "unremarkable",
        "within normal limits",
        "normal chest",
        "normal heart and lungs",
    ]

    # Only count as "Normal" if NONE of the pathology keywords
    # are present.  This avoids misclassifying something like
    # "no acute disease. large effusion noted." as Normal.
    pathology_hints = [
        "effusion", "pneumonia", "cardiomegaly", "atelectasis",
        "edema", "pneumothorax", "consolidation", "nodule", "mass",
        "emphysema", "fibrosis", "fracture", "opacity", "infiltrate",
        "tuberculosis", "granuloma", "scoliosis", "kyphosis",
        "airspace disease", "volume loss", "hyperinflat",
    ]

    has_pathology = any(p in normalized for p in pathology_hints)

    if not has_pathology:
        for phrase in normal_signals:
            if phrase in normalized:
                return "Normal"
        # Also: if the only content is "medical device" / "thoracic
        # vertebrae" + "no acute …", treat as Normal.
        if "no acute" in normalized or "no evidence" in normalized:
            return "Normal"

    # ── Pathology keywords (ordered specific → generic) ───────────
    label_keywords = [
        # Specific multi-word terms first
        ("pleural effusion", "Effusion"),
        ("pericardial effusion", "Effusion"),
        ("pulmonary edema", "Edema"),
        ("lung edema", "Edema"),
        ("pulmonary fibrosis", "Fibrosis"),
        ("lung nodule", "Nodule"),
        ("pulmonary nodule", "Nodule"),
        ("lung mass", "Mass"),
        ("pulmonary mass", "Mass"),
        ("heart failure", "Cardiomegaly"),
        ("cardiac enlargement", "Cardiomegaly"),
        ("airspace disease", "Opacity"),
        ("volume loss", "Atelectasis"),
        ("hyperinflation", "Emphysema"),
        ("hyperinflated", "Emphysema"),
        # Single-word terms
        ("cardiomegaly", "Cardiomegaly"),
        ("pneumonia", "Pneumonia"),
        ("atelectasis", "Atelectasis"),
        ("pneumothorax", "Pneumothorax"),
        ("consolidation", "Consolidation"),
        ("emphysema", "Emphysema"),
        ("fibrosis", "Fibrosis"),
        ("tuberculosis", "Tuberculosis"),
        ("sarcoidosis", "Sarcoidosis"),
        ("fracture", "Fracture"),
        ("scoliosis", "Scoliosis"),
        ("kyphosis", "Kyphosis"),
        ("granuloma", "Granuloma"),
        ("calcification", "Calcification"),
        ("scarring", "Scarring"),
        ("nodule", "Nodule"),
        ("mass", "Mass"),
        ("effusion", "Effusion"),
        ("edema", "Edema"),
        ("infiltrate", "Infiltrate"),
        ("opacity", "Opacity"),
    ]

    for keyword, label in label_keywords:
        if keyword in normalized:
            return label

    return "Unknown"


def choose_ground_truth_label(raw: dict, filename: str) -> str:
    """
    Choose a display label from raw metadata or derive one from text.
    
    Args:
        raw: Raw metadata record
        filename: Image filename for context
        
    Returns:
        Selected label string
    """
    raw_label = raw.get("ground_truth_label", "").strip()
    if raw_label and raw_label.lower() not in {"unknown", "other"}:
        return raw_label
    
    report_text = raw.get("report_snippet", "")
    problems = raw.get("problems", "")
    mesh_major = " ".join(raw.get("mesh_terms", {}).get("major", []))
    mesh_minor = " ".join(raw.get("mesh_terms", {}).get("minor", []))
    
    combined_text = " ".join(
        value
        for value in [report_text, problems, mesh_major, mesh_minor, filename]
        if value
    )
    
    return derive_label_from_text(combined_text)


def load_model(device: str = "cpu"):
    """
    Load BiomedCLIP model and preprocessing transforms.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, preprocess, tokenizer)
    """
    print(f"Loading BiomedCLIP model on {device}...")
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Embedding dimension: {EMBEDDING_DIM}")
    
    return model, preprocess_val, tokenizer


def generate_image_embeddings(
    model,
    preprocess,
    images_dir: Path,
    device: str = "cpu"
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all images in directory.
    
    Args:
        model: BiomedCLIP model
        preprocess: Image preprocessing transform
        images_dir: Directory containing processed images
        device: Device for inference
        
    Returns:
        Tuple of (embeddings array, list of image filenames)
    """
    # Find all WebP images
    image_files = sorted(list(images_dir.glob("*.webp")))
    
    if not image_files:
        raise ValueError(f"No WebP images found in {images_dir}")
    
    print(f"Generating embeddings for {len(image_files)} images...")
    
    embeddings = []
    filenames = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Encoding images"):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Generate embedding
                image_features = model.encode_image(image_tensor)
                
                # L2 normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy().flatten())
                filenames.append(img_path.name)
                
            except Exception as e:
                print(f"Warning: Failed to process {img_path.name}: {e}")
                continue
    
    embeddings_array = np.vstack(embeddings).astype(np.float32)
    
    print(f"Generated embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array, filenames


def generate_text_embeddings(
    model,
    tokenizer,
    queries: list[str],
    device: str = "cpu"
) -> np.ndarray:
    """
    Generate embeddings for text queries.
    
    Args:
        model: BiomedCLIP model
        tokenizer: Text tokenizer
        queries: List of text queries
        device: Device for inference
        
    Returns:
        Array of text embeddings
    """
    print(f"Generating text embeddings for {len(queries)} queries...")
    
    embeddings = []
    
    with torch.no_grad():
        for query in tqdm(queries, desc="Encoding queries"):
            # Tokenize
            tokens = tokenizer([query]).to(device)
            
            # Generate embedding
            text_features = model.encode_text(tokens)
            
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(text_features.cpu().numpy().flatten())
    
    return np.vstack(embeddings).astype(np.float32)


def compute_similarity(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int = TOP_K
) -> list[list[tuple[int, float]]]:
    """
    Compute cosine similarity between queries and images.
    
    Since vectors are L2-normalized, dot product = cosine similarity.
    
    Args:
        query_embeddings: Array of query embeddings (Q x D)
        image_embeddings: Array of image embeddings (N x D)
        top_k: Number of top results to return per query
        
    Returns:
        List of top-k results per query: [(image_idx, score), ...]
    """
    # Compute all similarities at once (Q x N)
    similarities = query_embeddings @ image_embeddings.T
    
    results = []
    for i in range(similarities.shape[0]):
        # Get top-k indices and scores
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        top_scores = similarities[i][top_indices]
        
        results.append([
            (int(idx), float(score))
            for idx, score in zip(top_indices, top_scores)
        ])
    
    return results


def compute_nearest_neighbors(
    embeddings: np.ndarray,
    k: int = NEAREST_NEIGHBORS_K
) -> dict[int, list[tuple[int, float]]]:
    """
    Compute k nearest neighbors for each image.
    
    Args:
        embeddings: Array of image embeddings (N x D)
        k: Number of neighbors to find
        
    Returns:
        Dictionary mapping image index to list of (neighbor_idx, score)
    """
    print(f"Computing {k} nearest neighbors for each image...")
    
    # Compute full similarity matrix (N x N)
    similarities = embeddings @ embeddings.T
    
    neighbors = {}
    
    for i in tqdm(range(len(embeddings)), desc="Finding neighbors"):
        # Get top-k+1 (including self) and exclude self
        top_indices = np.argsort(similarities[i])[::-1][:k+1]
        
        neighbor_list = []
        for idx in top_indices:
            if idx != i:  # Exclude self
                neighbor_list.append((int(idx), float(similarities[i][idx])))
            if len(neighbor_list) >= k:
                break
        
        neighbors[i] = neighbor_list
    
    return neighbors


def save_embeddings_binary(embeddings: np.ndarray, output_path: Path):
    """
    Save embeddings as binary Float32Array.
    
    Format: Raw float32 values, row-major order.
    Can be loaded in JavaScript as: new Float32Array(buffer)
    
    Args:
        embeddings: Array of embeddings (N x D)
        output_path: Path to save binary file
    """
    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    
    # Save as raw binary
    with open(output_path, "wb") as f:
        f.write(embeddings.tobytes())
    
    print(f"Saved embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")


def load_verified_labels(verified_labels_path: Optional[Path]) -> dict[str, str]:
    """
    Load a verified label mapping from a CSV file.
    
    Expected CSV columns:
      - filename: image filename (with extension) or stem
      - label: verified clinical label
    
    Args:
        verified_labels_path: Path to verified labels CSV
        
    Returns:
        Mapping from filename or stem to verified label
    """
    if not verified_labels_path or not verified_labels_path.exists():
        return {}
    
    labels: dict[str, str] = {}
    with open(verified_labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            if not filename or not label:
                continue
            labels[filename] = label
            labels[Path(filename).stem] = label
    
    return labels


def load_nih_labels(nih_labels_path: Optional[Path]) -> dict[str, dict]:
    """
    Load NIH ChestX-ray14 labels from the CSV produced by download_nih.py.
    
    Expected CSV columns:
      - filename:      e.g. "nih_00042.png"
      - labels:        pipe-separated, e.g. "Atelectasis|Effusion"
      - primary_label: first label, e.g. "Atelectasis"
    
    Args:
        nih_labels_path: Path to labels.csv from download_nih.py
        
    Returns:
        Mapping from image stem (e.g. "nih_00042") to dict with:
          - primary_label: display label
          - all_labels:    list of all labels
          - report_snippet: synthetic description from labels
    """
    if not nih_labels_path or not nih_labels_path.exists():
        return {}

    mapping: dict[str, dict] = {}
    with open(nih_labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue

            stem = Path(filename).stem
            raw_labels = (row.get("labels") or "").strip()
            primary = (row.get("primary_label") or "").strip()

            all_labels = [l.strip() for l in raw_labels.split("|") if l.strip()]
            if not all_labels:
                all_labels = [primary] if primary else ["Unknown"]
            if not primary:
                primary = all_labels[0]

            # Map to display names
            display_primary = NIH_DISPLAY_NAMES.get(primary, primary)
            display_all = [NIH_DISPLAY_NAMES.get(l, l) for l in all_labels]

            # Synthetic report snippet from labels
            if display_primary == "Normal":
                snippet = "No pathological findings. Normal chest radiograph."
            else:
                findings = ", ".join(display_all)
                snippet = f"Findings: {findings}."

            mapping[stem] = {
                "primary_label": display_primary,
                "all_labels": display_all,
                "all_labels_raw": all_labels,
                "report_snippet": snippet,
            }

    return mapping


def build_metadata(
    filenames: list[str],
    raw_metadata_path: Optional[Path] = None,
    images_base_url: str = "",
    verified_labels: Optional[dict[str, str]] = None,
    nih_labels: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """
    Build metadata array for all images.

    Label priority:
      1. NIH labels (from labels.csv — authoritative, published dataset)
      2. Verified labels (from verified_labels.csv — manual curation)
      3. Raw metadata labels (from Open-I download)
      4. Derived labels (NLP extraction from report text)
    
    Args:
        filenames: List of image filenames (in embedding order)
        raw_metadata_path: Path to raw metadata JSON from download step
        images_base_url: Base URL for images (for production)
        verified_labels: Optional mapping from filename to verified label
        nih_labels: Optional mapping from stem to NIH label dict
        
    Returns:
        List of metadata dictionaries
    """
    # Load raw metadata if available (for Open-I fallback)
    raw_metadata: dict[str, dict] = {}
    if raw_metadata_path and raw_metadata_path.exists():
        with open(raw_metadata_path) as f:
            raw_data = json.load(f)
            for item in raw_data:
                stem = Path(item.get("filename", "")).stem
                if stem:
                    raw_metadata[stem] = item
    
    nih_labels = nih_labels or {}
    verified_labels = verified_labels or {}
    metadata: list[dict] = []
    
    for idx, filename in enumerate(filenames):
        stem = Path(filename).stem

        # ── Priority 1: NIH dataset labels ────────────────────────
        nih = nih_labels.get(stem)
        if nih:
            label = nih["primary_label"]
            all_labels = nih["all_labels"]
            snippet = nih["report_snippet"]
            label_source = "dataset"
            label_verified = True
        else:
            all_labels = []
            # ── Priority 2: Verified labels ───────────────────────
            verified_label = verified_labels.get(filename) or verified_labels.get(stem)
            if verified_label:
                label = verified_label
                label_source = "verified"
                label_verified = True
            else:
                # ── Priority 3 & 4: Raw / Derived ────────────────
                raw = raw_metadata.get(stem, {})
                raw_label = raw.get("ground_truth_label", "").strip()
                if raw_label and raw_label.lower() not in {"unknown", "other"}:
                    label = raw_label
                    label_source = "raw"
                    label_verified = False
                else:
                    derived_label = choose_ground_truth_label(raw, filename)
                    label = derived_label
                    label_source = "derived" if derived_label != "Unknown" else "unknown"
                    label_verified = False

            snippet = raw_metadata.get(stem, {}).get("report_snippet", "No report available")
        
        entry = {
            "id": idx,
            "filename": filename,
            "url": f"{images_base_url}/{filename}" if images_base_url else f"images/{filename}",
            "report_snippet": snippet,
            "ground_truth_label": label,
            "all_labels": all_labels if all_labels else [label],
            "label_source": label_source,
            "label_verified": label_verified,
        }
        
        metadata.append(entry)
    
    return metadata


def main():
    """Main entry point for index generation."""
    parser = argparse.ArgumentParser(
        description="Generate search index using BiomedCLIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images_dir", "-i",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing processed WebP images"
    )
    parser.add_argument(
        "--raw_metadata", "-r",
        type=str,
        default=DEFAULT_RAW_METADATA,
        help="Path to raw metadata JSON from download step"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for index files"
    )
    parser.add_argument(
        "--images_base_url",
        type=str,
        default="",
        help="Base URL for images in production (e.g., GitHub Pages URL)"
    )
    parser.add_argument(
        "--nih_labels",
        type=str,
        default=DEFAULT_NIH_LABELS,
        help="Path to NIH labels.csv from download_nih.py (filename,labels,primary_label)"
    )
    parser.add_argument(
        "--verified_labels",
        type=str,
        default=DEFAULT_VERIFIED_LABELS,
        help="Optional CSV with verified labels (filename,label)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cpu or cuda)"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=TOP_K,
        help="Number of top results for fallback queries"
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    raw_metadata_path = Path(args.raw_metadata) if args.raw_metadata else None
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir}")
        print("Run process_images.py first.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Med-MIR Index Generation")
    print("=" * 60)
    print(f"Images: {images_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load model
    model, preprocess, tokenizer = load_model(args.device)
    
    # Generate image embeddings
    image_embeddings, filenames = generate_image_embeddings(
        model, preprocess, images_dir, args.device
    )
    
    # Save embeddings as binary
    embeddings_path = output_dir / "embeddings.bin"
    save_embeddings_binary(image_embeddings, embeddings_path)
    
    # Load label sources
    nih_labels_path = Path(args.nih_labels) if args.nih_labels else None
    nih_labels = load_nih_labels(nih_labels_path)
    if nih_labels:
        print(f"Loaded {len(nih_labels)} NIH labels from {nih_labels_path}")

    verified_labels_path = Path(args.verified_labels) if args.verified_labels else None
    verified_labels = load_verified_labels(verified_labels_path)
    if verified_labels:
        print(f"Loaded {len(verified_labels)} verified labels")

    # Build and save metadata
    metadata = build_metadata(
        filenames,
        raw_metadata_path,
        args.images_base_url,
        verified_labels,
        nih_labels,
    )
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Generate fallback results for common queries
    print("\nGenerating fallback results for common queries...")
    query_embeddings = generate_text_embeddings(model, tokenizer, COMMON_QUERIES, args.device)
    fallback_results = compute_similarity(query_embeddings, image_embeddings, args.top_k)
    
    fallback_dict = {}
    for query, results in zip(COMMON_QUERIES, fallback_results):
        fallback_dict[query.lower()] = [
            {"id": idx, "score": round(score, 4)}
            for idx, score in results
        ]
    
    fallback_path = output_dir / "fallback_results.json"
    with open(fallback_path, "w") as f:
        json.dump(fallback_dict, f, indent=2)
    print(f"Saved fallback results to {fallback_path}")
    
    # Compute nearest neighbors for "Find Similar" feature
    neighbors = compute_nearest_neighbors(image_embeddings, NEAREST_NEIGHBORS_K)
    
    # Convert to JSON-serializable format (string keys)
    neighbors_json = {
        str(k): [{"id": idx, "score": round(score, 4)} for idx, score in v]
        for k, v in neighbors.items()
    }
    
    neighbors_path = output_dir / "nearest_neighbors.json"
    with open(neighbors_path, "w") as f:
        json.dump(neighbors_json, f, indent=2)
    print(f"Saved nearest neighbors to {neighbors_path}")
    
    # Save index info
    index_info = {
        "num_images": len(filenames),
        "embedding_dim": EMBEDDING_DIM,
        "model": MODEL_NAME,
        "num_fallback_queries": len(COMMON_QUERIES),
        "top_k": args.top_k,
        "nearest_neighbors_k": NEAREST_NEIGHBORS_K,
    }
    
    info_path = output_dir / "index_info.json"
    with open(info_path, "w") as f:
        json.dump(index_info, f, indent=2)
    print(f"Saved index info to {info_path}")
    
    # Print label distribution summary
    label_dist: dict[str, int] = {}
    source_dist: dict[str, int] = {}
    for m in metadata:
        lbl = m.get("ground_truth_label", "Unknown")
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
        src = m.get("label_source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1

    print("\n" + "=" * 60)
    print("Index Generation Complete!")
    print("=" * 60)
    print(f"Total images indexed: {len(filenames)}")
    print(f"Embedding dimensions: {EMBEDDING_DIM}")
    print(f"Fallback queries: {len(COMMON_QUERIES)}")

    print(f"\nLabel Distribution:")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:25s} {cnt:5d}")

    print(f"\nLabel Source Breakdown:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {src:15s} {cnt:5d}")

    print(f"\nGenerated files:")
    print(f"  - {embeddings_path}")
    print(f"  - {metadata_path}")
    print(f"  - {fallback_path}")
    print(f"  - {neighbors_path}")
    print(f"  - {info_path}")


if __name__ == "__main__":
    main()
