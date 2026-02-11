#!/usr/bin/env python3
"""
evaluate.py - Model Evaluation and Metrics Generation

Evaluates the BiomedCLIP retrieval system and generates:
- metrics.json: Recall@K, mAP, and other retrieval metrics
- hard_cases.json: Examples where the model fails or confuses diagnoses

Usage:
    python evaluate.py --embeddings output/embeddings.bin --metadata output/metadata.json

Output:
    output/
    ├── metrics.json       # Performance metrics
    └── hard_cases.json    # Failure analysis
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np
from tqdm import tqdm
import torch
from open_clip import create_model_and_transforms, get_tokenizer


# Model configuration
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512

# Evaluation queries - label-query pairs for retrieval evaluation
# Covers all 15 NIH ChestX-ray14 labels (mapped to display names)
EVALUATION_QUERIES = {
    "Normal": ["normal chest xray", "healthy lungs", "no abnormality"],
    "Pneumonia": ["pneumonia", "lung infection", "pneumonic infiltrate"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "Effusion": ["pleural effusion", "fluid in chest", "pleural fluid"],
    "Atelectasis": ["atelectasis", "lung collapse", "collapsed lung"],
    "Pneumothorax": ["pneumothorax", "collapsed lung air", "air in pleural space"],
    "Consolidation": ["consolidation", "lung consolidation", "airspace consolidation"],
    "Edema": ["pulmonary edema", "lung edema", "fluid in lungs"],
    "Nodule": ["lung nodule", "pulmonary nodule", "nodular opacity"],
    "Mass": ["lung mass", "pulmonary mass", "chest mass"],
    # New labels from NIH ChestX-ray14
    "Infiltration": ["infiltration", "pulmonary infiltrate", "lung infiltrate"],
    "Emphysema": ["emphysema", "hyperinflation", "obstructive lung disease"],
    "Fibrosis": ["fibrosis", "pulmonary fibrosis", "interstitial fibrosis"],
    "Pleural Thickening": ["pleural thickening", "thickened pleura", "pleural scarring"],
    "Hernia": ["hernia", "hiatal hernia", "diaphragmatic hernia"],
}

# Known confusion pairs (for hard case analysis)
CONFUSION_PAIRS = [
    ("Atelectasis", "Effusion"),
    ("Pneumonia", "Consolidation"),
    ("Cardiomegaly", "Effusion"),
    ("Nodule", "Mass"),
    ("Pneumonia", "Edema"),
    ("Consolidation", "Infiltration"),
    ("Edema", "Effusion"),
    ("Pneumonia", "Infiltration"),
    ("Mass", "Nodule"),
    ("Fibrosis", "Pleural Thickening"),
    ("Emphysema", "Atelectasis"),
    ("Infiltration", "Pneumonia"),
]

# Semantically equivalent label groups — retrieving any label
# in the same group as the expected label counts as a "soft" hit.
SEMANTIC_GROUPS: list[set[str]] = [
    {"Effusion", "Edema"},                          # both involve fluid
    {"Nodule", "Mass"},                             # focal opacities
    {"Consolidation", "Infiltration"},              # airspace processes
    {"Pneumonia", "Consolidation", "Infiltration"}, # infectious overlap
    {"Atelectasis", "Effusion"},                    # common co-occurrence
    {"Fibrosis", "Pleural Thickening"},             # chronic scarring
    {"Emphysema", "Atelectasis"},                   # airway obstruction overlap
]


def _semantic_match(expected: str, retrieved: str) -> bool:
    """Return True if expected and retrieved labels belong to the same semantic group."""
    for group in SEMANTIC_GROUPS:
        if expected in group and retrieved in group:
            return True
    return False

DEFAULT_EMBEDDINGS = "output/embeddings.bin"
DEFAULT_METADATA = "output/metadata.json"
DEFAULT_OUTPUT_DIR = "output"
K_VALUES = [1, 5, 10, 20]


def load_embeddings(embeddings_path: Path, num_images: int) -> np.ndarray:
    """
    Load embeddings from binary file.
    
    Args:
        embeddings_path: Path to embeddings.bin
        num_images: Number of images (for reshaping)
        
    Returns:
        Embeddings array (N x D)
    """
    with open(embeddings_path, "rb") as f:
        data = f.read()
    
    embeddings = np.frombuffer(data, dtype=np.float32)
    embeddings = embeddings.reshape(num_images, -1)
    
    return embeddings


def load_model(device: str = "cpu"):
    """
    Load BiomedCLIP model for text encoding.
    
    Args:
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading BiomedCLIP model on {device}...")
    
    model, _, _ = create_model_and_transforms(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def encode_queries(
    model,
    tokenizer,
    queries: list[str],
    device: str = "cpu"
) -> np.ndarray:
    """
    Encode text queries using BiomedCLIP.
    
    Args:
        model: BiomedCLIP model
        tokenizer: Text tokenizer
        queries: List of text queries
        device: Device for inference
        
    Returns:
        L2-normalized query embeddings
    """
    embeddings = []
    
    with torch.no_grad():
        for query in queries:
            tokens = tokenizer([query]).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings.append(text_features.cpu().numpy().flatten())
    
    return np.vstack(embeddings).astype(np.float32)


def compute_recall_at_k(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    query_labels: list[str],
    image_labels: list[str],
    k_values: list[int],
    image_all_labels: Optional[list[list[str]]] = None,
) -> dict[int, float]:
    """
    Compute Recall@K for retrieval evaluation.
    
    Recall@K = fraction of queries where at least one relevant
    result appears in top K.

    When ``image_all_labels`` is provided (multi-label dataset), an image
    is considered relevant if **any** of its labels matches the query label.
    
    Args:
        query_embeddings: Query embedding matrix
        image_embeddings: Image embedding matrix
        query_labels: Ground truth label for each query
        image_labels: Primary ground truth label for each image
        k_values: List of K values to evaluate
        image_all_labels: Optional list of *all* labels for each image
        
    Returns:
        Dictionary mapping K to Recall@K
    """
    similarities = query_embeddings @ image_embeddings.T
    
    recalls = {k: 0.0 for k in k_values}
    num_queries = len(query_labels)
    
    for i, query_label in enumerate(query_labels):
        top_indices = np.argsort(similarities[i])[::-1]
        
        # Find relevant images (multi-label aware)
        ql = query_label.lower()
        if image_all_labels:
            relevant_set = set(
                j for j, labels in enumerate(image_all_labels)
                if any(l.lower() == ql for l in labels)
            )
        else:
            relevant_set = set(
                j for j, label in enumerate(image_labels)
                if label.lower() == ql
            )
        
        if not relevant_set:
            continue
        
        for k in k_values:
            top_k = set(top_indices[:k])
            if top_k & relevant_set:
                recalls[k] += 1
    
    for k in k_values:
        recalls[k] /= num_queries if num_queries > 0 else 1
    
    return recalls


def compute_mean_average_precision(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    query_labels: list[str],
    image_labels: list[str],
    image_all_labels: Optional[list[list[str]]] = None,
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        query_embeddings: Query embedding matrix
        image_embeddings: Image embedding matrix
        query_labels: Ground truth label for each query
        image_labels: Primary ground truth label for each image
        image_all_labels: Optional list of *all* labels for each image
        
    Returns:
        mAP score
    """
    similarities = query_embeddings @ image_embeddings.T
    
    average_precisions = []
    
    for i, query_label in enumerate(query_labels):
        ranking = np.argsort(similarities[i])[::-1]
        
        ql = query_label.lower()
        if image_all_labels:
            relevant_set = set(
                j for j, labels in enumerate(image_all_labels)
                if any(l.lower() == ql for l in labels)
            )
        else:
            relevant_set = set(
                j for j, label in enumerate(image_labels)
                if label.lower() == ql
            )
        
        if not relevant_set:
            continue
        
        num_relevant = 0
        precision_sum = 0.0
        
        for rank, idx in enumerate(ranking, 1):
            if idx in relevant_set:
                num_relevant += 1
                precision_sum += num_relevant / rank
        
        if num_relevant > 0:
            ap = precision_sum / len(relevant_set)
            average_precisions.append(ap)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_mrr(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    query_labels: list[str],
    image_labels: list[str],
    image_all_labels: Optional[list[list[str]]] = None,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        query_embeddings: Query embedding matrix
        image_embeddings: Image embedding matrix
        query_labels: Ground truth label for each query
        image_labels: Primary ground truth label for each image
        image_all_labels: Optional list of *all* labels for each image
        
    Returns:
        MRR score
    """
    similarities = query_embeddings @ image_embeddings.T
    
    reciprocal_ranks = []
    
    for i, query_label in enumerate(query_labels):
        ranking = np.argsort(similarities[i])[::-1]
        ql = query_label.lower()
        
        for rank, idx in enumerate(ranking, 1):
            if image_all_labels:
                if any(l.lower() == ql for l in image_all_labels[idx]):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                if image_labels[idx].lower() == ql:
                    reciprocal_ranks.append(1.0 / rank)
                    break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def find_hard_cases(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    query_labels: list[str],
    query_texts: list[str],
    image_labels: list[str],
    metadata: list[dict],
    confusion_pairs: list[tuple[str, str]],
    top_k: int = 5
) -> list[dict]:
    """
    Find hard cases where the model makes mistakes.
    
    Args:
        query_embeddings: Query embedding matrix
        image_embeddings: Image embedding matrix
        query_labels: Ground truth label for each query
        query_texts: Text of each query
        image_labels: Ground truth label for each image
        metadata: Image metadata
        confusion_pairs: Known confusion pairs to analyze
        top_k: Number of top results to analyze
        
    Returns:
        List of hard case dictionaries
    """
    similarities = query_embeddings @ image_embeddings.T
    
    hard_cases = []
    
    for i, (query_label, query_text) in enumerate(zip(query_labels, query_texts)):
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        top_labels = [image_labels[idx] for idx in top_indices]
        top_scores = [float(similarities[i][idx]) for idx in top_indices]
        
        # Check if top result is wrong
        if top_labels[0].lower() != query_label.lower():
            # Check if it's a known confusion pair
            confusion_type = "unknown"
            for pair in confusion_pairs:
                if query_label in pair and top_labels[0] in pair:
                    confusion_type = f"{pair[0]} vs {pair[1]}"
                    break
            
            hard_cases.append({
                "query": query_text,
                "expected_label": query_label,
                "retrieved_label": top_labels[0],
                "confusion_type": confusion_type,
                "top_results": [
                    {
                        "id": int(idx),
                        "label": image_labels[idx],
                        "score": round(score, 4),
                        "filename": metadata[idx].get("filename", ""),
                    }
                    for idx, score in zip(top_indices[:top_k], top_scores)
                ],
            })
    
    return hard_cases


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval model and generate metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=str,
        default=DEFAULT_EMBEDDINGS,
        help="Path to embeddings.bin"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=DEFAULT_METADATA,
        help="Path to metadata.json"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_path}")
        print("Run generate_index.py first.")
        return
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Med-MIR Model Evaluation")
    print("=" * 60)
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    num_images = len(metadata)
    # Primary label (for display)
    image_labels = [m.get("ground_truth_label", "Unknown") for m in metadata]
    # All labels per image (for multi-label matching in evaluation)
    image_all_labels: list[list[str]] = [
        m.get("all_labels", [m.get("ground_truth_label", "Unknown")]) for m in metadata
    ]
    
    print(f"Images: {num_images}")
    print(f"Embeddings: {embeddings_path}")
    
    # Load embeddings
    image_embeddings = load_embeddings(embeddings_path, num_images)
    print(f"Embeddings shape: {image_embeddings.shape}")
    
    # Load model for query encoding
    model, tokenizer = load_model(args.device)
    
    # Build evaluation queries
    all_queries: list[str] = []
    all_query_labels: list[str] = []
    all_query_texts: list[str] = []

    for label, queries in EVALUATION_QUERIES.items():
        for query in queries:
            all_query_labels.append(label)
            all_query_texts.append(query)
            all_queries.append(query)

    print(f"Evaluation queries: {len(all_queries)}")

    # Encode queries
    query_embeddings = encode_queries(model, tokenizer, all_queries, args.device)

    # ── Label distribution (multi-label aware) ───────────────────
    label_counts: dict[str, int] = defaultdict(int)
    for labels_list in image_all_labels:
        for label in labels_list:
            label_counts[label] += 1

    # Identify which evaluation labels actually have images
    labels_with_images = {
        label for label in EVALUATION_QUERIES
        if label_counts.get(label, 0) > 0
    }
    labels_without_images = set(EVALUATION_QUERIES) - labels_with_images

    if labels_without_images:
        print(f"\n⚠ Labels with ZERO images (excluded from adjusted metrics):")
        for lbl in sorted(labels_without_images):
            print(f"    {lbl}")

    # Split queries into "viable" (label exists) and "impossible"
    viable_indices = [
        i for i, lbl in enumerate(all_query_labels)
        if lbl in labels_with_images
    ]
    impossible_indices = [
        i for i, lbl in enumerate(all_query_labels)
        if lbl not in labels_with_images
    ]

    # ── Strict metrics (all queries, original method) ────────────
    print("\nComputing strict metrics (all queries)...")
    strict_recalls = compute_recall_at_k(
        query_embeddings, image_embeddings,
        all_query_labels, image_labels, K_VALUES,
        image_all_labels=image_all_labels,
    )
    strict_mAP = compute_mean_average_precision(
        query_embeddings, image_embeddings,
        all_query_labels, image_labels,
        image_all_labels=image_all_labels,
    )
    strict_mrr = compute_mrr(
        query_embeddings, image_embeddings,
        all_query_labels, image_labels,
        image_all_labels=image_all_labels,
    )

    # ── Adjusted metrics (only viable queries) ───────────────────
    print("Computing adjusted metrics (viable queries only)...")
    if viable_indices:
        viable_embeddings = query_embeddings[viable_indices]
        viable_labels = [all_query_labels[i] for i in viable_indices]

        adj_recalls = compute_recall_at_k(
            viable_embeddings, image_embeddings,
            viable_labels, image_labels, K_VALUES,
            image_all_labels=image_all_labels,
        )
        adj_mAP = compute_mean_average_precision(
            viable_embeddings, image_embeddings,
            viable_labels, image_labels,
            image_all_labels=image_all_labels,
        )
        adj_mrr = compute_mrr(
            viable_embeddings, image_embeddings,
            viable_labels, image_labels,
            image_all_labels=image_all_labels,
        )
    else:
        adj_recalls = {k: 0.0 for k in K_VALUES}
        adj_mAP = 0.0
        adj_mrr = 0.0

    # ── Semantic (soft) metrics ───────────────────────────────────
    # A "soft hit" counts if the retrieved label is in the same
    # semantic group as the expected label.
    print("Computing semantic metrics (soft label matching)...")
    similarities = query_embeddings @ image_embeddings.T
    soft_recalls: dict[int, float] = {k: 0.0 for k in K_VALUES}
    soft_rr_sum = 0.0

    for i, query_label in enumerate(all_query_labels):
        ranking = np.argsort(similarities[i])[::-1]

        # Soft relevant set (multi-label aware)
        soft_relevant: set[int] = set()
        for j in range(len(image_labels)):
            all_lbls = image_all_labels[j] if image_all_labels else [image_labels[j]]
            for lbl in all_lbls:
                if lbl.lower() == query_label.lower() or _semantic_match(query_label, lbl):
                    soft_relevant.add(j)
                    break

        if not soft_relevant:
            continue

        for k in K_VALUES:
            if set(ranking[:k]) & soft_relevant:
                soft_recalls[k] += 1

        for rank, idx in enumerate(ranking, 1):
            if idx in soft_relevant:
                soft_rr_sum += 1.0 / rank
                break

    n_queries = len(all_query_labels)
    for k in K_VALUES:
        soft_recalls[k] /= n_queries if n_queries else 1
    soft_mrr = soft_rr_sum / n_queries if n_queries else 0.0

    # ── Per-label metrics ─────────────────────────────────────────
    per_label_metrics: dict[str, dict] = {}
    for label in EVALUATION_QUERIES:
        label_indices = [i for i, l in enumerate(all_query_labels) if l == label]
        if not label_indices:
            continue

        lqe = query_embeddings[label_indices]
        lql = [all_query_labels[i] for i in label_indices]

        lr = compute_recall_at_k(
            lqe, image_embeddings, lql, image_labels, [10],
            image_all_labels=image_all_labels,
        )

        per_label_metrics[label] = {
            "recall_at_10": round(lr[10], 4),
            "num_images": label_counts.get(label, 0),
            "num_queries": len(label_indices),
            "has_images": label_counts.get(label, 0) > 0,
        }

    # ── Build full metrics dictionary ─────────────────────────────
    metrics = {
        "overall": {
            "recall_at_1": round(strict_recalls[1], 4),
            "recall_at_5": round(strict_recalls[5], 4),
            "recall_at_10": round(strict_recalls[10], 4),
            "recall_at_20": round(strict_recalls[20], 4),
            "mAP": round(strict_mAP, 4),
            "MRR": round(strict_mrr, 4),
        },
        "adjusted": {
            "recall_at_1": round(adj_recalls[1], 4),
            "recall_at_5": round(adj_recalls[5], 4),
            "recall_at_10": round(adj_recalls[10], 4),
            "recall_at_20": round(adj_recalls[20], 4),
            "mAP": round(adj_mAP, 4),
            "MRR": round(adj_mrr, 4),
            "num_viable_queries": len(viable_indices),
            "excluded_labels": sorted(labels_without_images),
        },
        "semantic": {
            "recall_at_1": round(soft_recalls[1], 4),
            "recall_at_5": round(soft_recalls[5], 4),
            "recall_at_10": round(soft_recalls[10], 4),
            "recall_at_20": round(soft_recalls[20], 4),
            "MRR": round(soft_mrr, 4),
            "note": "Counts retrieval as correct if the label is in the same medical semantic group",
        },
        "per_label": per_label_metrics,
        "dataset": {
            "num_images": num_images,
            "num_labels": len(label_counts),
            "label_distribution": dict(label_counts),
        },
        "evaluation": {
            "num_queries": len(all_queries),
            "num_viable_queries": len(viable_indices),
            "num_impossible_queries": len(impossible_indices),
            "k_values": K_VALUES,
        },
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # ── Hard cases ────────────────────────────────────────────────
    print("\nFinding hard cases...")
    hard_cases = find_hard_cases(
        query_embeddings, image_embeddings,
        all_query_labels, all_query_texts,
        image_labels, metadata,
        CONFUSION_PAIRS
    )

    # Annotate each hard case with whether it is semantic-match
    for hc in hard_cases:
        hc["is_semantic_match"] = _semantic_match(
            hc["expected_label"], hc["retrieved_label"]
        )

    # Group
    hard_cases_summary = {
        "total_failures": len(hard_cases),
        "semantic_matches": sum(1 for hc in hard_cases if hc["is_semantic_match"]),
        "true_failures": sum(1 for hc in hard_cases if not hc["is_semantic_match"]),
        "failure_rate": round(len(hard_cases) / len(all_queries), 4) if all_queries else 0,
        "true_failure_rate": round(
            sum(1 for hc in hard_cases if not hc["is_semantic_match"]) / len(all_queries), 4
        ) if all_queries else 0,
        "impossible_queries": len(impossible_indices),
        "by_confusion_type": defaultdict(list),
        "examples": hard_cases[:20],
    }

    for case in hard_cases:
        ctype = case["confusion_type"]
        hard_cases_summary["by_confusion_type"][ctype].append({
            "query": case["query"],
            "expected": case["expected_label"],
            "got": case["retrieved_label"],
            "is_semantic_match": case["is_semantic_match"],
        })

    hard_cases_summary["by_confusion_type"] = dict(hard_cases_summary["by_confusion_type"])

    # Save hard cases
    hard_cases_path = output_dir / "hard_cases.json"
    with open(hard_cases_path, "w") as f:
        json.dump(hard_cases_summary, f, indent=2)
    print(f"Saved hard cases to {hard_cases_path}")

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nStrict Metrics (all {len(all_queries)} queries):")
    print(f"  Recall@1:  {metrics['overall']['recall_at_1']:.4f}")
    print(f"  Recall@5:  {metrics['overall']['recall_at_5']:.4f}")
    print(f"  Recall@10: {metrics['overall']['recall_at_10']:.4f}")
    print(f"  Recall@20: {metrics['overall']['recall_at_20']:.4f}")
    print(f"  mAP:       {metrics['overall']['mAP']:.4f}")
    print(f"  MRR:       {metrics['overall']['MRR']:.4f}")

    print(f"\nAdjusted Metrics ({len(viable_indices)} viable queries):")
    print(f"  Recall@1:  {metrics['adjusted']['recall_at_1']:.4f}")
    print(f"  Recall@5:  {metrics['adjusted']['recall_at_5']:.4f}")
    print(f"  Recall@10: {metrics['adjusted']['recall_at_10']:.4f}")
    print(f"  Recall@20: {metrics['adjusted']['recall_at_20']:.4f}")
    print(f"  mAP:       {metrics['adjusted']['mAP']:.4f}")
    print(f"  MRR:       {metrics['adjusted']['MRR']:.4f}")

    print(f"\nSemantic Metrics (soft label matching):")
    print(f"  Recall@1:  {metrics['semantic']['recall_at_1']:.4f}")
    print(f"  Recall@5:  {metrics['semantic']['recall_at_5']:.4f}")
    print(f"  Recall@10: {metrics['semantic']['recall_at_10']:.4f}")
    print(f"  Recall@20: {metrics['semantic']['recall_at_20']:.4f}")
    print(f"  MRR:       {metrics['semantic']['MRR']:.4f}")

    print(f"\nHard Cases:")
    print(f"  Strict failures:   {hard_cases_summary['total_failures']}/{len(all_queries)}")
    print(f"  Semantic matches:  {hard_cases_summary['semantic_matches']} (medically related)")
    print(f"  True failures:     {hard_cases_summary['true_failures']}")
    print(f"  Impossible queries:{hard_cases_summary['impossible_queries']} (label not in dataset)")

    if hard_cases_summary["by_confusion_type"]:
        print(f"\n  Confusion types:")
        for ctype, cases in hard_cases_summary["by_confusion_type"].items():
            sem = sum(1 for c in cases if c.get("is_semantic_match"))
            print(f"    {ctype}: {len(cases)} cases ({sem} semantic matches)")


if __name__ == "__main__":
    main()
