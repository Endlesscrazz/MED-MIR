#!/usr/bin/env python3
"""
export_onnx.py - Robust Dual-Encoder Export
Fixes 'ConvInteger' errors by restricting quantization to MatMul/Gemm nodes.
"""

import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import open_clip
    from open_clip import create_model_and_transforms, get_tokenizer
except ImportError:
    print("Error: open_clip not installed.")
    exit(1)

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx libraries missing.")

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OPSET_VERSION = 17 # Use 17 for broader compatibility

# --- Wrappers ---
class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.text
    def forward(self, input_ids):
        features = self.text(input_ids)
        return F.normalize(features, dim=-1)

class VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
    def forward(self, pixel_values):
        features = self.visual(pixel_values)
        return F.normalize(features, dim=-1)

# --- Graph Surgery & Safe Quantization ---
def safe_quantize(input_path, output_path):
    if not ONNX_AVAILABLE: return

    print(f"  ⚡ Optimizing {input_path.name}...")
    
    # 1. Graph Surgery (Strip Shapes)
    model = onnx.load(str(input_path))
    if len(model.graph.value_info) > 0:
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
    
    clean_path = input_path.parent / f"{input_path.stem}_clean.onnx"
    onnx.save(model, str(clean_path))
    
    # 2. Selective Quantization
    # CRITICAL FIX: Only quantize MatMul and Gemm. 
    # Excluding 'Conv' prevents the "ConvInteger not found" error in browser.
    print(f"  ⚡ Quantizing (MatMul/Gemm only)...")
    try:
        quantize_dynamic(
            str(clean_path),
            str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul', 'Gemm'] 
        )
        
        orig = input_path.stat().st_size / (1024*1024)
        new = output_path.stat().st_size / (1024*1024)
        print(f"     ✅ Done: {orig:.1f}MB -> {new:.1f}MB")
        
        clean_path.unlink()
    except Exception as e:
        print(f"     ❌ Quantization failed: {e}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/model")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading BiomedCLIP...")
    model, _, _ = create_model_and_transforms(MODEL_NAME)
    model.eval()
    
    # 1. Text
    print("\nExporting Text Encoder...")
    p1 = out_dir / "text_encoder.onnx"
    p1_q = out_dir / "text_encoder_quantized.onnx"
    dummy_text = torch.randint(0, 1000, (1, 77))
    torch.onnx.export(
        TextEncoderWrapper(model), dummy_text, str(p1),
        input_names=["input_ids"], output_names=["embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "embeds": {0: "batch"}},
        opset_version=OPSET_VERSION
    )
    safe_quantize(p1, p1_q)
    
    # 2. Vision
    print("\nExporting Vision Encoder...")
    p2 = out_dir / "vision_encoder.onnx"
    p2_q = out_dir / "vision_encoder_quantized.onnx"
    dummy_img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        VisionEncoderWrapper(model), dummy_img, str(p2),
        input_names=["pixel_values"], output_names=["embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeds": {0: "batch"}},
        opset_version=OPSET_VERSION
    )
    safe_quantize(p2, p2_q)
    
    # 3. Configs
    print("\nSaving Configs...")
    with open(out_dir / "config.json", "w") as f:
        json.dump({"vocab_size": model.vocab_size}, f)
        
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        tok.save_pretrained(str(out_dir))
    except:
        print("⚠️  Could not download vocab automatically. Please ensure vocab.txt is present.")

if __name__ == "__main__":
    main()

# python python/export_onnx.py --output_dir python/output/model