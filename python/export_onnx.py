#!/usr/bin/env python3
"""
export_onnx.py - BiomedCLIP Text Encoder ONNX Export

Exports the BiomedCLIP text encoder to ONNX format for browser inference.
Uses open_clip to load the model and carefully extracts the text encoding path.

Usage:
    python export_onnx.py --output_dir output/model
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Check for required packages
try:
    import open_clip
    from open_clip import create_model_and_transforms, get_tokenizer
except ImportError:
    print("Error: open_clip not installed. Run: pip install open_clip_torch")
    exit(1)

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not available. Quantization will be skipped.")

# Configuration
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512
CONTEXT_LENGTH = 256
OPSET_VERSION = 18  # Use newer opset to avoid conversion issues


class BiomedCLIPTextEncoder(nn.Module):
    """
    Standalone text encoder extracted from BiomedCLIP.
    
    This wrapper exposes only the text encoding functionality
    in a way that's compatible with ONNX tracing.
    
    For CustomTextCLIP models, the text encoder is accessed via model.text
    and handles tokenization, projection, and normalization internally.
    """
    
    def __init__(self, clip_model):
        super().__init__()
        # For CustomTextCLIP, the text encoder is a complete module
        # that handles everything internally (tokenization, projection, etc.)
        self.text_encoder = clip_model.text
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            
        Returns:
            L2-normalized embeddings [batch_size, embedding_dim]
        """
        # The text encoder module handles everything:
        # - Token embedding
        # - Positional embedding  
        # - Transformer layers
        # - Pooling
        # - Projection
        # We just need to call it with token IDs
        features = self.text_encoder(input_ids)
        
        # L2 normalize (critical for cosine similarity via dot product)
        # Note: encode_text may already normalize, but we ensure it here
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features


def load_biomedclip():
    """Load BiomedCLIP model using open_clip."""
    print(f"Loading BiomedCLIP from: {MODEL_NAME}")
    
    model, _, preprocess = create_model_and_transforms(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    
    model.eval()
    
    # Print model info
    print(f"  Context length: {model.context_length}")
    print(f"  Vocab size: {model.vocab_size}")
    
    return model, tokenizer


def export_to_onnx(text_encoder: nn.Module, output_path: Path, context_length: int, vocab_size: int):
    """
    Export text encoder to ONNX format.
    
    Args:
        text_encoder: The text encoder module
        output_path: Path to save the ONNX model
        context_length: Maximum sequence length
        vocab_size: Vocabulary size for dummy input
    """
    print(f"\nExporting to ONNX (opset {OPSET_VERSION})...")
    
    text_encoder.eval()
    
    # Create dummy input with valid token IDs (not all zeros)
    # Use a reasonable sequence length (not full context_length to avoid issues)
    seq_len = min(77, context_length)  # Standard CLIP uses 77
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    
    print(f"  Dummy input shape: {dummy_input.shape}")
    print(f"  Dummy input range: [{dummy_input.min().item()}, {dummy_input.max().item()}]")
    
    # Test forward pass first
    try:
        with torch.no_grad():
            test_output = text_encoder(dummy_input)
            print(f"  Test forward pass successful, output shape: {test_output.shape}")
    except Exception as e:
        print(f"  ERROR: Forward pass failed: {e}")
        raise
    
    # Export with tracing
    print("  Starting ONNX export...")
    with torch.no_grad():
        # Use dynamic shapes to handle variable sequence lengths
        # The model will accept any sequence length up to context_length
        torch.onnx.export(
            text_encoder,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size"}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Exported: {output_path}")
    print(f"  ✓ Size: {size_mb:.2f} MB")
    
    # Check if model size is reasonable
    # A BERT-based text encoder should be at least 50-100MB
    if size_mb < 10:
        print(f"\n  ⚠️  WARNING: Model size ({size_mb:.2f} MB) is suspiciously small!")
        print("     This might indicate the export didn't capture all weights.")
        print("     However, the model might still work - verify with inference.")
    elif size_mb < 50:
        print(f"\n  ⚠️  NOTE: Model size ({size_mb:.2f} MB) is smaller than expected.")
        print("     This might be normal if the model uses shared weights or compression.")
    
    return size_mb


def quantize_onnx(input_path: Path, output_path: Path):
    """Quantize ONNX model to INT8 using pre-processing to fix shape issues."""
    if not ONNX_AVAILABLE:
        print("Skipping quantization (onnxruntime not available)")
        return None
    
    print("\nQuantizing to INT8...")
    
    try:
        from onnxruntime.quantization import quant_pre_process
        
        # Step 1: Pre-process model to fix shape inference issues
        print("  Step 1: Pre-processing model for quantization...")
        preprocessed_path = input_path.parent / "model_preprocessed.onnx"
        
        try:
            quant_pre_process(
                input_model=str(input_path),
                output_model_path=str(preprocessed_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                auto_merge=True,  # Auto-merge conflicting shapes
                verbose=1
            )
            input_path_for_quant = preprocessed_path
            print("  ✓ Pre-processing successful")
        except Exception as e:
            print(f"  ⚠ Pre-processing failed: {e}")
            print("  Attempting direct quantization...")
            input_path_for_quant = input_path
        
        # Step 2: Quantize
        print("  Step 2: Quantizing model...")
        quantize_dynamic(
            str(input_path_for_quant),
            str(output_path),
            weight_type=QuantType.QInt8
        )
        
        # Clean up temp files
        if input_path_for_quant != input_path and input_path_for_quant.exists():
            input_path_for_quant.unlink()
        
        orig_size = input_path.stat().st_size / (1024 * 1024)
        quant_size = output_path.stat().st_size / (1024 * 1024)
        
        print(f"  ✓ Original: {orig_size:.2f} MB")
        print(f"  ✓ Quantized: {quant_size:.2f} MB")
        print(f"  ✓ Compression: {orig_size / quant_size:.1f}x")
        
        return quant_size
        
    except Exception as e:
        print(f"  ⚠ Quantization failed: {e}")
        print("  Continuing without quantization...")
        return None


def verify_onnx(model_path: Path, tokenizer, original_model):
    """Verify ONNX model produces correct outputs."""
    if not ONNX_AVAILABLE:
        print("Skipping verification (onnxruntime not available)")
        return
    
    print("\nVerifying ONNX model...")
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(str(model_path))
    except Exception as e:
        print(f"  ERROR: Failed to load ONNX model: {e}")
        return
    
    # Test text
    test_texts = ["normal chest xray", "pneumonia", "cardiomegaly"]
    
    for text in test_texts:
        try:
            # Tokenize
            tokens = tokenizer([text])
            # Convert to numpy array of token IDs
            if isinstance(tokens, torch.Tensor):
                input_ids = tokens.cpu().numpy().astype(np.int64)
            else:
                # Handle case where tokenizer returns a dict
                input_ids = tokens['input_ids'].cpu().numpy().astype(np.int64) if isinstance(tokens, dict) else tokens.numpy().astype(np.int64)
            
            # The model was exported with static shape 77, but tokenizer produces 256
            # Truncate or pad to match the exported model's expected input size
            expected_len = 77  # This matches what we used in export
            if input_ids.shape[1] > expected_len:
                input_ids = input_ids[:, :expected_len]
            elif input_ids.shape[1] < expected_len:
                # Pad with zeros (pad token)
                pad_width = expected_len - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            
            # Run ONNX inference
            onnx_output = session.run(None, {"input_ids": input_ids})[0]
            
            # Run PyTorch inference for comparison using the text encoder directly
            text_encoder = BiomedCLIPTextEncoder(original_model)
            text_encoder.eval()
            with torch.no_grad():
                torch_tokens = torch.from_numpy(input_ids)
                torch_output = text_encoder(torch_tokens)
                torch_output = torch_output.numpy()
            
            # Compare
            cosine_sim = np.dot(onnx_output[0], torch_output[0])
            onnx_norm = np.linalg.norm(onnx_output[0])
            torch_norm = np.linalg.norm(torch_output[0])
            
            status = "✓" if cosine_sim > 0.99 and abs(onnx_norm - 1.0) < 0.01 else "⚠"
            print(f"  {status} '{text}': similarity={cosine_sim:.4f}, onnx_norm={onnx_norm:.4f}, torch_norm={torch_norm:.4f}")
        except Exception as e:
            print(f"  ⚠ Error verifying '{text}': {e}")


def save_tokenizer_files(output_dir: Path, tokenizer, model):
    """
    Save tokenizer files for browser-side tokenization.
    
    Exports the underlying HuggingFace tokenizer's vocab.txt and config
    files so our custom JavaScript WordPiece tokenizer can replicate the
    same tokenization in the browser.
    
    Args:
        output_dir: Directory to save tokenizer files
        tokenizer: The open_clip tokenizer (HFTokenizer wrapper)
        model: The CLIP model (for context_length, vocab_size)
    """
    print("\nSaving tokenizer files for browser...")
    
    # Access the underlying HuggingFace AutoTokenizer
    hf_tokenizer = None
    if hasattr(tokenizer, 'tokenizer'):
        hf_tokenizer = tokenizer.tokenizer
    
    if hf_tokenizer is not None:
        # Save all HuggingFace tokenizer files (vocab.txt, tokenizer.json, etc.)
        hf_tokenizer.save_pretrained(str(output_dir))
        
        # List what was saved
        for f in sorted(output_dir.iterdir()):
            if f.suffix in ('.txt', '.json', '.model'):
                size_kb = f.stat().st_size / 1024
                print(f"  ✓ Saved: {f.name} ({size_kb:.1f} KB)")
        
        # Verify vocab.txt exists (critical for browser tokenizer)
        vocab_path = output_dir / "vocab.txt"
        if vocab_path.exists():
            num_tokens = sum(1 for line in open(vocab_path) if line.strip())
            print(f"  ✓ Vocabulary: {num_tokens} tokens")
        else:
            print("  ⚠ WARNING: vocab.txt not found in saved files!")
            print("    The browser tokenizer requires vocab.txt.")
            print("    Attempting manual extraction...")
            _extract_vocab_manually(hf_tokenizer, vocab_path)
    else:
        print("  ⚠ Cannot access HuggingFace tokenizer from open_clip.")
        print("    Attempting to download PubMedBERT tokenizer directly...")
        _download_pubmedbert_tokenizer(output_dir)
    
    # Save model config for the browser (context_length, vocab_size, etc.)
    config = {
        "model_type": "biomedclip",
        "vocab_size": int(model.vocab_size),
        "context_length": int(model.context_length),
        "embedding_dim": EMBEDDING_DIM,
        "tokenizer_type": "bert-wordpiece",
        "special_tokens": {
            "cls": "[CLS]",
            "sep": "[SEP]",
            "pad": "[PAD]",
            "unk": "[UNK]"
        }
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved: config.json")


def _extract_vocab_manually(hf_tokenizer, vocab_path: Path):
    """Extract vocab.txt manually from a HuggingFace tokenizer."""
    try:
        vocab = hf_tokenizer.get_vocab()
        # Sort by ID to get the correct order
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        with open(vocab_path, "w") as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")
        print(f"  ✓ Manually extracted vocab.txt ({len(sorted_vocab)} tokens)")
    except Exception as e:
        print(f"  ✗ Manual extraction failed: {e}")


def _download_pubmedbert_tokenizer(output_dir: Path):
    """Download PubMedBERT tokenizer from HuggingFace as fallback."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
        tok.save_pretrained(str(output_dir))
        print("  ✓ Downloaded and saved PubMedBERT tokenizer")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("    Please manually download vocab.txt from:")
        print("    https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")


def main():
    parser = argparse.ArgumentParser(
        description="Export BiomedCLIP text encoder to ONNX"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output/model",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip INT8 quantization"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip model verification"
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Only save tokenizer files (skip ONNX export)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BiomedCLIP Text Encoder ONNX Export")
    print("=" * 60)
    
    # 1. Load model
    model, tokenizer = load_biomedclip()
    
    # 2. Save tokenizer files (always needed for browser)
    save_tokenizer_files(output_dir, tokenizer, model)
    
    # If tokenizer-only mode, skip ONNX export
    if args.tokenizer_only:
        print("\n✓ Tokenizer-only mode. Skipping ONNX export.")
        print(f"  Files saved to: {output_dir.absolute()}")
        return
    
    # 3. Inspect model structure
    print("\nInspecting model structure...")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Has 'text' attribute: {hasattr(model, 'text')}")
    print(f"  Has 'encode_text' method: {hasattr(model, 'encode_text')}")
    
    # 4. Extract text encoder
    print("\nExtracting text encoder...")
    text_encoder = BiomedCLIPTextEncoder(model)
    text_encoder.eval()
    
    # Test forward pass to ensure it works
    print("Testing forward pass...")
    test_tokens = torch.randint(0, model.vocab_size, (1, min(77, model.context_length)), dtype=torch.long)
    with torch.no_grad():
        test_output = text_encoder(test_tokens)
        print(f"  Test output shape: {test_output.shape}")
        print(f"  Test output norm: {test_output.norm(dim=-1).item():.4f}")
    
    # 5. Export to ONNX
    onnx_path = output_dir / "model.onnx"
    size = export_to_onnx(text_encoder, onnx_path, model.context_length, model.vocab_size)
    
    # 6. Quantize (optional, may fail due to shape inference issues)
    quant_path = output_dir / "model_quantized.onnx"
    if not args.skip_quantize:
        quant_size = quantize_onnx(onnx_path, quant_path)
        if quant_size is None:
            print("\n  Note: Using non-quantized model (quantization failed)")
            quant_path = None
    
    # 7. Verify
    verify_model_path = quant_path if quant_path and quant_path.exists() else onnx_path
    if not args.skip_verify:
        verify_onnx(verify_model_path, tokenizer, model)
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  - {f.name} ({size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  - {f.name} ({size / 1024:.2f} KB)")
    
    print("\nNext steps:")
    print("1. Copy files to the web app:")
    print("   cp output/model/model_flat.onnx web/public/demo-data/model/")
    print("   cp output/model/vocab.txt web/public/demo-data/model/")
    print("   cp output/model/config.json web/public/demo-data/model/")
    print("2. The browser worker loads model_flat.onnx + vocab.txt directly")
    print("3. Tokenization uses BERT WordPiece (PubMedBERT vocabulary)")


if __name__ == "__main__":
    main()
