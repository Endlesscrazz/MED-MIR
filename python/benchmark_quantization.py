#!/usr/bin/env python3
"""
Benchmark quantized vs FP32 ONNX encoders for Med-MIR.

Outputs:
  - JSON report with model sizes, embedding fidelity, and latency
  - Markdown summary for easy sharing
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer


TEXT_SAMPLES = [
    "normal chest xray",
    "pleural effusion",
    "cardiomegaly",
    "pulmonary edema",
    "pneumonia",
    "lung mass",
    "nodule in right upper lobe",
    "bilateral infiltrates",
    "pneumothorax",
    "pleural thickening",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def ms(v: float) -> float:
    return round(v * 1000.0, 3)


@dataclass
class SizeMetrics:
    fp32_mb: float
    int8_mb: float
    reduction_pct: float


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def preprocess_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1,3,224,224]
    return arr.astype(np.float32)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Med-MIR quantization quality and latency")
    parser.add_argument("--model_dir", default="python/output/model")
    parser.add_argument("--images_dir", default="python/output/images")
    parser.add_argument("--output_json", default="python/output/quantization_benchmark.json")
    parser.add_argument("--output_md", default="python/output/quantization_benchmark.md")
    parser.add_argument("--vision_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    images_dir = Path(args.images_dir)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    text_fp_path = model_dir / "text_encoder.onnx"
    text_int8_path = model_dir / "text_encoder_quantized.onnx"
    text_fp_data = model_dir / "text_encoder.onnx.data"
    vision_fp_path = model_dir / "vision_encoder.onnx"
    vision_int8_path = model_dir / "vision_encoder_quantized.onnx"
    vision_fp_data = model_dir / "vision_encoder.onnx.data"

    size_text = SizeMetrics(
        fp32_mb=file_mb(text_fp_data),
        int8_mb=file_mb(text_int8_path),
        reduction_pct=(1.0 - file_mb(text_int8_path) / file_mb(text_fp_data)) * 100.0,
    )
    size_vision = SizeMetrics(
        fp32_mb=file_mb(vision_fp_data),
        int8_mb=file_mb(vision_int8_path),
        reduction_pct=(1.0 - file_mb(vision_int8_path) / file_mb(vision_fp_data)) * 100.0,
    )

    text_fp = load_session(text_fp_path)
    text_int8 = load_session(text_int8_path)
    vision_fp = load_session(vision_fp_path)
    vision_int8 = load_session(vision_int8_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    text_cosines: list[float] = []
    text_lat_fp: list[float] = []
    text_lat_int8: list[float] = []

    # Warm-up
    warm_tokens = tokenizer(TEXT_SAMPLES[0], max_length=77, truncation=True, padding="max_length", return_tensors="np")
    warm_inp = {"input_ids": warm_tokens["input_ids"].astype(np.int64)}
    _ = text_fp.run(None, warm_inp)
    _ = text_int8.run(None, warm_inp)

    for text in TEXT_SAMPLES:
        tokens = tokenizer(text, max_length=77, truncation=True, padding="max_length", return_tensors="np")
        inp = {"input_ids": tokens["input_ids"].astype(np.int64)}

        t0 = time.perf_counter()
        emb_fp = text_fp.run(None, inp)[0].reshape(-1).astype(np.float32)
        t1 = time.perf_counter()
        emb_i8 = text_int8.run(None, inp)[0].reshape(-1).astype(np.float32)
        t2 = time.perf_counter()

        text_lat_fp.append(t1 - t0)
        text_lat_int8.append(t2 - t1)
        text_cosines.append(cosine(emb_fp, emb_i8))

    image_files = sorted(images_dir.glob("*.webp"))
    if len(image_files) < args.vision_samples:
        raise ValueError(f"Not enough images in {images_dir} for {args.vision_samples} samples")
    vision_paths = random.sample(image_files, args.vision_samples)

    vision_cosines: list[float] = []
    vision_lat_fp: list[float] = []
    vision_lat_int8: list[float] = []

    warm_img = preprocess_image(vision_paths[0])
    warm_vinp = {"pixel_values": warm_img}
    _ = vision_fp.run(None, warm_vinp)
    _ = vision_int8.run(None, warm_vinp)

    for img_path in vision_paths:
        vinp = {"pixel_values": preprocess_image(img_path)}

        t0 = time.perf_counter()
        emb_fp = vision_fp.run(None, vinp)[0].reshape(-1).astype(np.float32)
        t1 = time.perf_counter()
        emb_i8 = vision_int8.run(None, vinp)[0].reshape(-1).astype(np.float32)
        t2 = time.perf_counter()

        vision_lat_fp.append(t1 - t0)
        vision_lat_int8.append(t2 - t1)
        vision_cosines.append(cosine(emb_fp, emb_i8))

    combined_fp = size_text.fp32_mb + size_vision.fp32_mb
    combined_int8 = size_text.int8_mb + size_vision.int8_mb

    report = {
        "run_config": {
            "model_dir": str(model_dir),
            "images_dir": str(images_dir),
            "text_samples": len(TEXT_SAMPLES),
            "vision_samples": args.vision_samples,
            "seed": args.seed,
        },
        "size_metrics": {
            "text_encoder": asdict(size_text),
            "vision_encoder": asdict(size_vision),
            "combined": {
                "fp32_mb": combined_fp,
                "int8_mb": combined_int8,
                "reduction_pct": (1.0 - combined_int8 / combined_fp) * 100.0,
            },
        },
        "fidelity_metrics": {
            "text_encoder_cosine_fp32_vs_int8": summarize(text_cosines),
            "vision_encoder_cosine_fp32_vs_int8": summarize(vision_cosines),
        },
        "latency_ms": {
            "text_fp32": summarize([ms(v) for v in text_lat_fp]),
            "text_int8": summarize([ms(v) for v in text_lat_int8]),
            "vision_fp32": summarize([ms(v) for v in vision_lat_fp]),
            "vision_int8": summarize([ms(v) for v in vision_lat_int8]),
        },
    }

    out_json.write_text(json.dumps(report, indent=2))

    md = []
    md.append("# Quantization Benchmark Report")
    md.append("")
    md.append("## Size")
    md.append("")
    md.append(f"- Text FP32: {size_text.fp32_mb:.2f} MB")
    md.append(f"- Text INT8: {size_text.int8_mb:.2f} MB")
    md.append(f"- Text reduction: {size_text.reduction_pct:.2f}%")
    md.append(f"- Vision FP32: {size_vision.fp32_mb:.2f} MB")
    md.append(f"- Vision INT8: {size_vision.int8_mb:.2f} MB")
    md.append(f"- Vision reduction: {size_vision.reduction_pct:.2f}%")
    md.append(f"- Combined FP32: {combined_fp:.2f} MB")
    md.append(f"- Combined INT8: {combined_int8:.2f} MB")
    md.append(f"- Combined reduction: {(1.0 - combined_int8 / combined_fp) * 100.0:.2f}%")
    md.append("")
    md.append("## Fidelity (Cosine Similarity FP32 vs INT8 Embeddings)")
    md.append("")
    md.append(f"- Text mean cosine: {report['fidelity_metrics']['text_encoder_cosine_fp32_vs_int8']['mean']:.6f}")
    md.append(f"- Vision mean cosine: {report['fidelity_metrics']['vision_encoder_cosine_fp32_vs_int8']['mean']:.6f}")
    md.append("")
    md.append("## Latency (ms, CPU)")
    md.append("")
    md.append(f"- Text FP32 mean: {report['latency_ms']['text_fp32']['mean']:.3f}")
    md.append(f"- Text INT8 mean: {report['latency_ms']['text_int8']['mean']:.3f}")
    md.append(f"- Vision FP32 mean: {report['latency_ms']['vision_fp32']['mean']:.3f}")
    md.append(f"- Vision INT8 mean: {report['latency_ms']['vision_int8']['mean']:.3f}")
    md.append("")
    out_md.write_text("\n".join(md))

    print(f"Saved JSON report: {out_json}")
    print(f"Saved Markdown report: {out_md}")


if __name__ == "__main__":
    main()
