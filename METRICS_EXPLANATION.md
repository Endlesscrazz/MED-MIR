# Med-MIR Metrics Explanation

This document explains:
1. How retrieval metrics are calculated
2. How to interpret them
3. How quantization was evaluated
4. Separate text/vision encoder quantization metrics

Data sources used in this document:
- `web/public/demo-data/metrics.json`
- `python/output/quantization_benchmark.json`
- `python/evaluate.py`
- `python/export_onnx.py`

---

## 1) Retrieval Metrics: How They Are Calculated

## Evaluation setup (current fresh run)
- Indexed images: `495`
- Labels: `15`
- Clinical query set: `45` total (`3 queries x 15 labels`)
- Embeddings: BiomedCLIP `512-d`
- Similarity: cosine similarity via dot product on L2-normalized vectors

## A) Recall@K
For each query:
- rank all images by similarity
- check whether at least one relevant image appears in top-K

Formula:
- `Recall@K = (#queries with at least 1 relevant image in top-K) / (#queries)`

In code:
- implemented in `compute_recall_at_k` in `python/evaluate.py`

## B) mAP (Mean Average Precision)
For each query:
- traverse ranked list
- compute precision at each rank where a relevant image appears
- average those precision values to get AP for the query
- then average AP across queries

Formula:
- `mAP = mean(AP_query_i)`

In code:
- implemented in `compute_mean_average_precision` in `python/evaluate.py`

## C) MRR (Mean Reciprocal Rank)
For each query:
- find rank of first relevant image
- reciprocal rank = `1 / rank`
- average across queries

Formula:
- `MRR = mean(1 / rank_first_relevant)`

In code:
- implemented in `compute_mrr` in `python/evaluate.py`

## D) Strict vs Semantic metrics
- **Strict**: relevant means exact label match
- **Semantic**: relevant also includes medically related labels based on `SEMANTIC_GROUPS` in `python/evaluate.py`
  - e.g., overlaps like `Pneumonia/Consolidation/Infiltration`, `Effusion/Edema`, etc.

---

## 2) Current Retrieval Results and Interpretation

From `web/public/demo-data/metrics.json`:

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

## How to explain these in plain language
- `Recall@10 = 0.9111` (strict): for ~91% of clinical queries, at least one exactly relevant image appears in top 10.
- `Recall@10 = 1.0000` (semantic): when clinically related labels are accepted, every query has a relevant/related image in top 10.
- `MRR = 0.5657` (strict): relevant images usually appear early, often in top few ranks.
- `mAP = 0.2199`: long-tail ranking quality is moderate; top-k quality is stronger than full-list precision.

---

## 3) Quantization Method

Quantization flow is implemented in `python/export_onnx.py`:

1. Export text and vision encoders to ONNX.
2. Clean ONNX graph metadata (remove `value_info` intermediates).
3. Run dynamic INT8 quantization on `MatMul` and `Gemm` ops only.
   - This avoids browser/runtime issues seen when quantizing unsupported conv paths.
4. Keep FP32 and INT8 artifacts for benchmarking.

Output models:
- Text:
  - FP32: `text_encoder.onnx` + `text_encoder.onnx.data`
  - INT8: `text_encoder_quantized.onnx`
- Vision:
  - FP32: `vision_encoder.onnx` + `vision_encoder.onnx.data`
  - INT8: `vision_encoder_quantized.onnx`

---

## 4) Quantization Metrics (Text and Vision Separately)

From `python/output/quantization_benchmark.json`:

## A) Text encoder

### Size
- FP32: `418.54 MB`
- INT8: `174.58 MB`
- Reduction: `58.29%`

### Fidelity (FP32 vs INT8 embedding cosine)
- Mean cosine: `0.994167`
- Min cosine: `0.990028`
- Max cosine: `0.997189`

Interpretation:
- Very high cosine indicates INT8 text embeddings closely preserve FP32 directionality.

### Latency (CPU, benchmark script)
- FP32 mean: `63.258 ms`
- INT8 mean: `25.078 ms`

Interpretation:
- ~2.5x faster mean text embedding inference in this local benchmark.

## B) Vision encoder

### Size
- FP32: `328.81 MB`
- INT8: `85.72 MB`
- Reduction: `73.93%`

### Fidelity (FP32 vs INT8 embedding cosine)
- Mean cosine: `0.998051`
- Min cosine: `0.996782`
- Max cosine: `0.999126`

Interpretation:
- Extremely high fidelity; quantized vision embeddings are very close to FP32.

### Latency (CPU, benchmark script)
- FP32 mean: `148.205 ms`
- INT8 mean: `62.688 ms`

Interpretation:
- ~2.36x faster mean vision embedding inference in this local benchmark.

## C) Combined model footprint
- FP32 total: `747.36 MB`
- INT8 total: `260.30 MB`
- Total reduction: `65.17%`

---

## 5) How These Metrics Support Your Project Story

For professor/recruiter explanation:

1. Retrieval quality:
- "The system retrieves clinically relevant images with strong top-k performance (strict Recall@10 = 91.11%)."

2. Edge deployment feasibility:
- "Quantization reduces total model footprint by 65.17%, enabling browser delivery."

3. Quality-speed tradeoff:
- "INT8 preserves embedding fidelity (text cosine 0.994, vision cosine 0.998) while reducing latency substantially."

4. Engineering contribution:
- "This is an end-to-end ML systems project: data pipeline, embedding indexing, ONNX quantization, browser inference, and rigorous evaluation."

---

## 6) Reproducibility Commands

```bash
# Retrieval metrics
med_mir/bin/python python/evaluate.py \
  --embeddings python/output/embeddings.bin \
  --metadata python/output/metadata.json \
  --output_dir python/output \
  --device cpu

# Quantization benchmark (text + vision)
med_mir/bin/python python/benchmark_quantization.py \
  --model_dir python/output/model \
  --images_dir python/output/images \
  --output_json python/output/quantization_benchmark.json \
  --output_md python/output/quantization_benchmark.md \
  --vision_samples 20
```

---

## 7) Notes/Caveats
- Latency numbers are hardware-dependent (CPU, memory, thermal state).
- Semantic metrics depend on configured clinical equivalence groups.
- Current dataset run is based on images present on mounted drive (subset of full NIH archive present locally).
