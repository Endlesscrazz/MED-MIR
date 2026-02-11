import os
import onnx
from onnx.external_data_helper import load_external_data_for_model
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

# Paths
INPUT_DIR = Path("python/output/model")
OUTPUT_DIR = Path("web/public/demo-data/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = INPUT_DIR / "model.onnx"
FLAT_MODEL_PATH = INPUT_DIR / "model_flat.onnx"
QUANTIZED_MODEL_PATH = OUTPUT_DIR / "model_quantized.onnx"

def main():
    print(f"Loading split model from {MODEL_PATH}...")
    
    # 1. Load the split model and force merge external data
    # 'load_external_data=True' is the key fix here
    model = onnx.load(str(MODEL_PATH), load_external_data=True)
    
    # 2. Save as a single "flat" file
    print(f"Saving flattened model to {FLAT_MODEL_PATH}...")
    # save_as_external_data=False forces it into one file
    onnx.save(model, str(FLAT_MODEL_PATH))
    
    # Check size
    size_mb = FLAT_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✓ Flat model size: {size_mb:.2f} MB (Should be ~420MB)")
    
    if size_mb < 100:
        print("  ❌ Error: Flat model is too small. Merging failed.")
        return

    # 3. Quantize the Flat Model
    print("Quantizing to INT8...")
    try:
        quantize_dynamic(
            str(FLAT_MODEL_PATH),
            str(QUANTIZED_MODEL_PATH),
            weight_type=QuantType.QInt8
        )
        
        q_size_mb = QUANTIZED_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"\n✅ SUCCESS! Quantized model saved to: {QUANTIZED_MODEL_PATH}")
        print(f"  Final Size: {q_size_mb:.2f} MB")
        
        # Cleanup
        if FLAT_MODEL_PATH.exists():
            FLAT_MODEL_PATH.unlink()
            print("  Cleaned up temporary flat file.")
            
    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")

if __name__ == "__main__":
    main()