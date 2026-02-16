# Med-MIR Fresh Run Metrics Report

Run date: 2026-02-10  
Dataset source: `/Volumes/One_Touch/Med-Mir-proj/NIH-dataset`  
Target size requested: 500  
Actual selected size: 495

## Important dataset note
The mounted NIH path currently contains only `images_001` and `images_002` (24,966 images available locally), not all 112,120 NIH images listed in CSV.  
To avoid missing files, this run was built from the intersection of CSV entries with files actually present on disk.

---

## 1) Retrieval Quality Metrics (from `web/public/demo-data/metrics.json`)

### Strict
- Recall@1: `0.4000`
- Recall@5: `0.7778`
- Recall@10: `0.9111`
- Recall@20: `1.0000`
- mAP: `0.2199`
- MRR: `0.5657`

### Semantic
- Recall@1: `0.5111`
- Recall@5: `0.8667`
- Recall@10: `1.0000`
- Recall@20: `1.0000`
- MRR: `0.6740`

### Evaluation setup
- Number of images indexed: `495`
- Embedding dimension: `512`
- Evaluation queries: `45`
- Labels with images: `15/15`
- Impossible queries: `0`

---

## 2) Quantization Footprint Metrics (from `web/public/demo-data/model/`)

### Text encoder
- FP32 weights (`text_encoder.onnx.data`): `418.54 MB`
- INT8 model (`text_encoder_quantized.onnx`): `174.58 MB`
- Size reduction: `58.29%`

### Vision encoder
- FP32 weights (`vision_encoder.onnx.data`): `328.81 MB`
- INT8 model (`vision_encoder_quantized.onnx`): `85.72 MB`
- Size reduction: `73.93%`

### Combined
- FP32 total: `747.36 MB`
- INT8 total: `260.30 MB`
- Overall reduction: `65.17%`

---

## 3) Pipeline Outputs Generated

Fresh outputs generated under `python/output/` and copied to `web/public/demo-data/`:
- `embeddings.bin`
- `metadata.json`
- `fallback_results.json`
- `nearest_neighbors.json`
- `index_info.json`
- `metrics.json`
- `hard_cases.json`
- `images/`
- `model/` (text + vision ONNX and quantized versions)

---

## 4) Resume-safe statements you can use now

1. "Built a browser-native medical image retrieval system on NIH ChestX-ray data (495-image balanced subset), achieving Recall@10 of 91.11% (strict) and 100% (semantic)."
2. "Implemented ONNX export + INT8 quantization for dual encoders (text and vision), reducing combined model footprint from 747 MB FP32 to 260 MB INT8 (65.17% smaller)."
3. "Delivered end-to-end ML pipeline from data selection and preprocessing to embedding index generation, retrieval evaluation (Recall@K/mAP/MRR), and frontend integration."

---

## 5) Commands used for this run

```bash
# 1) Build subset from available NIH files (495 selected)
# (custom intersection step was used to avoid missing-file rows)

# 2) Process images
python3 python/process_images.py \
  --input_dir python/data/nih/images \
  --output_dir python/output/images \
  --max_dim 256 --quality 85 --workers 4

# 3) Generate index
med_mir/bin/python python/generate_index.py \
  --images_dir python/output/images \
  --nih_labels python/data/nih/labels.csv \
  --output_dir python/output \
  --device cpu

# 4) Export + quantize text/vision ONNX
med_mir/bin/python python/export_onnx.py --output_dir python/output/model

# 5) Evaluate
med_mir/bin/python python/evaluate.py \
  --embeddings python/output/embeddings.bin \
  --metadata python/output/metadata.json \
  --output_dir python/output \
  --device cpu
```
