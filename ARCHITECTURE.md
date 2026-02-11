# Med-MIR: Architecture & Technical Documentation

> A comprehensive technical reference for the Med-MIR (Medical Multimodal Image
> Retrieval) system—covering motivation, architecture, data pipeline,
> evaluation methodology, results, and future work.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Browser Inference Engine](#5-browser-inference-engine)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Results](#7-results)
8. [Technology Stack](#8-technology-stack)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [References](#11-references)

---

## 1. Problem Statement

Hospitals generate thousands of medical images daily—chest X-rays, CTs, MRIs.
When a radiologist encounters an ambiguous finding, they often need to ask:

- *"Show me similar images to this one"*
- *"Show me all chest X-rays with cardiomegaly"*
- *"What does pleural effusion look like in comparison?"*

**Current solutions** require:

| Requirement | Typical Cost |
|-------------|-------------|
| Cloud-based GPU servers | $2,000–10,000/month |
| Sending patient data to external servers | HIPAA/privacy risk |
| Enterprise medical IT software | $50K+ licensing |
| Dedicated ML engineering team | $200K+/year |

**Med-MIR eliminates all of these** by running the entire retrieval system
in the user's browser—no server, no cloud, no cost, no privacy risk.

---

## 2. Solution Overview

Med-MIR is a **Content-Based Medical Image Retrieval (CBMIR)** system with a
unique edge-computing approach:

| Property | Med-MIR | Traditional CBIR |
|----------|---------|-------------------|
| **Server** | None (static files) | GPU server required |
| **Cost** | $0 | $2,000+/month |
| **Privacy** | Data never leaves device | Data sent to cloud |
| **Internet** | Works offline after first load | Always required |
| **Deployment** | Static website (GitHub Pages) | Docker + Kubernetes |
| **Model** | BiomedCLIP (SOTA, 15M pre-training pairs) | Often generic CLIP |

### Core Innovation

The AI model (BiomedCLIP's text encoder) runs **entirely in the browser** via
WebAssembly through ONNX Runtime Web. A custom BERT WordPiece tokenizer,
implemented in JavaScript, processes text input without any server-side
dependency.

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       OFFLINE PIPELINE (Python)                         │
│                    Runs once on developer machine                        │
│                                                                         │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐    │
│  │  Download   │──▶│  Process   │──▶│  Generate   │──▶│  Evaluate  │    │
│  │  NIH Data   │   │  Images    │   │  Index      │   │  Metrics   │    │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘    │
│     labels.csv      *.webp         embeddings.bin    metrics.json      │
│     images/         (256px)        metadata.json     hard_cases.json   │
│                                    fallback.json                       │
│                                    neighbors.json                      │
│                                                                         │
│  ┌────────────┐                                                         │
│  │  Export     │──▶  model_flat.onnx + vocab.txt + tokenizer configs   │
│  │  ONNX      │                                                         │
│  └────────────┘                                                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                           Static files served by
                           GitHub Pages / Vercel
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ONLINE APPLICATION (Browser)                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       Main Thread (React)                        │   │
│  │                                                                  │   │
│  │  ┌────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐  │   │
│  │  │ Search │──▶│ Fallback     │──▶│ Display   │   │ Metrics  │  │   │
│  │  │ Box    │   │ Lookup       │   │ Results   │   │ Dashboard│  │   │
│  │  └────────┘   └──────┬───────┘   └───────────┘   └──────────┘  │   │
│  │                      │ no match                                  │   │
│  │                      ▼                                           │   │
│  │              postMessage()                                       │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │                                               │
│  ┌──────────────────────▼───────────────────────────────────────────┐   │
│  │                    Web Worker (Background Thread)                 │   │
│  │                                                                  │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────────────────┐   │   │
│  │  │  Custom     │──▶│  ONNX      │──▶│  512-dim embedding     │   │   │
│  │  │  BERT       │   │  Runtime   │   │  (L2-normalized)       │   │   │
│  │  │  Tokenizer  │   │  Web       │   └────────────┬───────────┘   │   │
│  │  └────────────┘   └────────────┘                │               │   │
│  └──────────────────────────────────────────────────┼───────────────┘   │
│                                                     │                   │
│                              ┌───────────────────────┘                  │
│                              ▼                                          │
│                    Cosine Similarity (dot product)                       │
│                    vs. all image embeddings                              │
│                    (pre-loaded Float32Array)                             │
│                              │                                          │
│                              ▼                                          │
│                       Top-K Ranked Results                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Search Strategy: Three Paths

| Path | When | Latency | Method |
|------|------|---------|--------|
| **Fallback Lookup** | Query matches 1 of 55 pre-computed queries | < 50ms | JSON key lookup |
| **Fuzzy Fallback** | Query words appear in a pre-computed key | < 50ms | Word-overlap matching |
| **Live AI Inference** | Completely novel query | 5–30s first time, 1–2s cached | ONNX model in Web Worker |

**Why pre-compute at all?** Pre-computed results provide instant UX for the
most common medical queries. The live AI inference path handles the long tail—
a doctor might search *"bilateral pleural effusion with cardiomegaly"*, which
no pre-computed result covers.

### 3.3 Image-to-Image Retrieval

The "Find Similar" feature uses pre-computed nearest neighbors:

1. During index generation, each image's K=10 nearest neighbors are computed
   using cosine similarity between image embeddings
2. Results are stored in `nearest_neighbors.json`
3. When a user clicks "Find Similar" on any result, the pre-computed neighbors
   are returned instantly (< 10ms)

---

## 4. Data Pipeline

### 4.1 Dataset: NIH ChestX-ray14

| Property | Value |
|----------|-------|
| **Full name** | NIH Clinical Center Chest X-ray Dataset |
| **Paper** | Wang et al., *"ChestX-ray8"*, CVPR 2017 |
| **Total images** | 112,120 frontal-view chest X-rays |
| **Patients** | 30,805 unique |
| **Labels** | 14 pathologies + "No Finding" (15 total, multi-label) |
| **Label extraction** | NLP-extracted from radiology reports (>90% accuracy) |
| **License** | CC0 1.0 (public domain) |
| **Citation count** | 10,000+ |

#### The 15 Labels

| Label | Display Name | Full Dataset Count | Our Subset |
|-------|-------------|-------------------|------------|
| No Finding | Normal | ~60,361 | 17 |
| Infiltration | Infiltration | ~19,894 | 12 |
| Effusion | Effusion | ~13,317 | 21 |
| Atelectasis | Atelectasis | ~11,559 | 35 |
| Nodule | Nodule | ~6,331 | 6 |
| Mass | Mass | ~5,782 | 7 |
| Pneumothorax | Pneumothorax | ~5,302 | 4 |
| Consolidation | Consolidation | ~4,667 | 21 |
| Pleural_Thickening | Pleural Thickening | ~3,385 | 12 |
| Cardiomegaly | Cardiomegaly | ~2,776 | 24 |
| Emphysema | Emphysema | ~2,516 | 14 |
| Edema | Edema | ~2,303 | 6 |
| Fibrosis | Fibrosis | ~1,686 | 30 |
| Pneumonia | Pneumonia | ~1,431 | 5 |
| Hernia | Hernia | ~227 | 19 |

#### Multi-Label Structure

Each image can have multiple diagnoses:

```
00000013_005.png → "Atelectasis|Effusion"
00000032_001.png → "Cardiomegaly|Effusion|Edema"
00000076_000.png → "No Finding"
```

- **Display**: Primary (first) label shown in UI
- **Evaluation**: An image is "relevant" if *any* of its labels match the query
- **Report snippet**: Synthesized from label list (e.g., "Findings: Atelectasis, Effusion.")

### 4.2 Data Ingestion (`download_nih.py`)

Two-step process designed for limited local storage:

```bash
# Step 1: Download full dataset to external HDD (~45 GB)
python download_nih.py --kaggle --output_dir /Volumes/MyDrive/nih-data

# Step 2: Select balanced subset, copy to project (~1.2 GB → 233 images)
python download_nih.py --select-subset \
  --source_dir /Volumes/MyDrive/nih-data \
  --output_dir data/nih \
  --per_label 150
```

**Balancing strategy**: Parse `Data_Entry_2017_v2020.csv`, group images by
label, randomly sample up to 150 per label with seed=42 for reproducibility.
Multi-label images count toward all their labels.

**Output**:
```
data/nih/
├── images/      # 233 PNG images (original resolution)
└── labels.csv   # filename, labels (pipe-separated), primary_label
```

### 4.3 Image Processing (`process_images.py`)

Optimizes raw PNGs for web delivery:

| Transform | Details |
|-----------|---------|
| Resize | Max 256px (preserving aspect ratio) |
| Color space | Convert to RGB |
| Format | WebP (quality 85, method 6) |
| Compression | **98.8%** (87 MB → 1.1 MB) |

```bash
python process_images.py -i data/nih/images -o output/images
```

### 4.4 Embedding Generation (`generate_index.py`)

**Model**: BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)

| Component | Architecture | Details |
|-----------|-------------|---------|
| Visual encoder | ViT-B/16 | 224px input, 512-dim output |
| Text encoder | PubMedBERT | 77 token context, 512-dim output |
| Pre-training | 15M pairs | Biomedical image-text from PubMed Central |

**Steps**:

1. Load BiomedCLIP on CPU (or CUDA)
2. Pass each WebP image through the visual encoder → 512-dim vector
3. **L2-normalize** all vectors (enables dot product = cosine similarity)
4. Save as raw `Float32Array` binary (`embeddings.bin`)
5. Build metadata from NIH `labels.csv` → `metadata.json`
6. Encode 55 common clinical queries → compute top-10 results → `fallback_results.json`
7. Compute 10 nearest neighbors per image → `nearest_neighbors.json`

```bash
python generate_index.py \
  --images_dir output/images \
  --nih_labels data/nih/labels.csv \
  --output_dir output
```

**Output files**:

| File | Size | Description |
|------|------|-------------|
| `embeddings.bin` | 466 KB | Float32Array (233 × 512) |
| `metadata.json` | 260 KB | Image metadata with verified labels |
| `fallback_results.json` | 30 KB | Pre-computed top-10 for 55 queries |
| `nearest_neighbors.json` | 360 KB | Top-10 similar images per image |
| `index_info.json` | ~200 B | Summary (num_images, dim, model) |

### 4.5 ONNX Export (`export_onnx.py`)

Converts BiomedCLIP's text encoder to ONNX for browser inference:

1. Load model via `open_clip`
2. Wrap text encoder + projection layer in a torch Module
3. Export to ONNX (opset 18, dynamic batch axis)
4. Export tokenizer files (vocab.txt, configs) for custom JS tokenizer

```bash
python export_onnx.py --output_dir output/model
```

**Output**:

| File | Size | Description |
|------|------|-------------|
| `model_flat.onnx` | ~420 MB | BiomedCLIP text encoder (FP32) |
| `vocab.txt` | 220 KB | PubMedBERT vocabulary (30,522 tokens) |
| `config.json` | ~500 B | Model config |
| `tokenizer_config.json` | ~500 B | Tokenizer settings |
| `special_tokens_map.json` | ~300 B | [CLS], [SEP], [PAD], [UNK] |

### 4.6 Why Not Quantize?

We export the **FP32** model rather than INT8 quantized because:

- BiomedCLIP's open_clip architecture has non-standard graph structures that
  cause quantization errors (shape mismatches during ONNX optimization)
- The 420 MB model is cached by the browser after first load, so download
  cost is one-time
- FP32 preserves full embedding quality—important for medical retrieval
  where subtle differences matter

Future work: INT8 quantization with careful graph surgery, or distillation
to a smaller model.

---

## 5. Browser Inference Engine

### 5.1 Web Worker Architecture

The ONNX model runs in a **dedicated Web Worker** to prevent UI blocking:

```
Main Thread (React)           Web Worker (inference.worker.js)
      │                              │
      │── INIT ──────────────────────▶│ Load ONNX Runtime (CDN)
      │                              │ Load vocab.txt
      │                              │ Load model_flat.onnx
      │◀── INIT_COMPLETE ────────────│
      │                              │
      │── EMBED_TEXT("pneumonia") ──▶│ Tokenize (custom BERT)
      │                              │ Run ONNX inference
      │                              │ L2-normalize output
      │◀── EMBEDDING([0.12, ...]) ───│
      │                              │
```

### 5.2 Custom BERT Tokenizer

Since we bypassed `@xenova/transformers` (which couldn't load the open_clip-based
BiomedCLIP), we implemented a **custom BERT WordPiece tokenizer** in plain
JavaScript:

1. Load `vocab.txt` (30,522 tokens)
2. Build token-to-ID lookup map
3. Lowercase + strip accents
4. Split into words → apply WordPiece decomposition
5. Add [CLS] / [SEP] tokens, pad to context length (77)
6. Output: `input_ids` Int64Array for ONNX model

### 5.3 ONNX Runtime Web

- Loaded from CDN (`cdn.jsdelivr.net/npm/onnxruntime-web`)
- Runs via **WebAssembly (WASM)** backend
- Model cached in browser's Cache Storage after first download
- Subsequent queries: ~1–2 seconds inference time

---

## 6. Evaluation Methodology

### 6.1 Protocol

1. Define **45 evaluation queries** (3 natural-language variations per label × 15 labels)
   - Example for Pneumonia: `"pneumonia"`, `"pneumonia chest xray"`, `"lung infection pneumonia"`
2. Encode each query with BiomedCLIP's text encoder → 512-dim vector
3. Compute cosine similarity against all 233 image embeddings
4. Rank images by similarity score
5. Check if top-K results contain at least one image whose label matches

### 6.2 Three Evaluation Tiers

| Tier | Description | Why |
|------|-------------|-----|
| **Strict** | Exact label match on all 45 queries | Harshest—penalizes for labels with few images |
| **Adjusted** | Excludes queries for labels with 0 images in dataset | Fairest—model can't succeed on absent labels |
| **Semantic** | Counts medically-related labels as correct | Reflects clinical utility |

### 6.3 Semantic Groups

Medical images often share overlapping conditions. These groups define which
label confusions are "medically reasonable":

| Group | Labels | Clinical Rationale |
|-------|--------|--------------------|
| Fluid-related | Effusion, Edema | Both involve fluid accumulation |
| Focal opacities | Nodule, Mass | Both are focal lesions of different size |
| Airspace processes | Consolidation, Infiltration | Both fill alveolar spaces |
| Infectious overlap | Pneumonia, Consolidation, Infiltration | Pneumonia manifests as consolidation/infiltration |
| Common co-occurrence | Atelectasis, Effusion | Frequently appear together |
| Chronic scarring | Fibrosis, Pleural Thickening | Both represent chronic tissue changes |

### 6.4 Hard Case Analysis

For queries where the model fails (top-1 ≠ expected), we categorize:

- **Semantic match**: The "wrong" label is in the same semantic group
  (e.g., Edema → Effusion). Clinically useful despite being technically wrong.
- **True failure**: An unrelated label was retrieved. These represent genuine
  model weaknesses.
- **Confusion pattern**: Recurring label pairs that the model confuses,
  revealing systematic biases.

### 6.5 Key Metrics Explained

| Metric | Definition | Intuition |
|--------|-----------|-----------|
| **Recall@K** | Fraction of queries with ≥1 correct result in top K | "Did we find something relevant?" |
| **mAP** | Mean of per-query average precision | "How high are correct results ranked?" |
| **MRR** | Mean of 1/rank_of_first_correct | "How quickly do we find the first relevant result?" |

---

## 7. Results

### 7.1 Current Evaluation (233 NIH ChestX-ray14 Images)

| Metric | Strict | Adjusted | Semantic |
|--------|--------|----------|----------|
| **Recall@1** | 28.9% | 28.9% | 37.8% |
| **Recall@5** | 71.1% | 71.1% | 86.7% |
| **Recall@10** | **84.4%** | **84.4%** | **95.6%** |
| **Recall@20** | 95.6% | 95.6% | 97.8% |
| **mAP** | 22.3% | 22.3% | — |
| **MRR** | 0.478 | 0.478 | 0.593 |

> Strict = Adjusted because all 15 labels have images (0 excluded labels).

### 7.2 Interpretation

- **Recall@10 = 84.4%**: For 84% of clinical queries, at least one relevant
  image appears in the top 10 results. This is strong for a zero-shot retrieval
  system with no task-specific fine-tuning.

- **Semantic Recall@10 = 95.6%**: When we count medically-related retrievals
  (e.g., returning Effusion for an Edema query), the system is useful 96% of
  the time. This reflects real clinical utility.

- **mAP = 22.3%**: Lower because correct results aren't always ranked first.
  With 15 competing labels and 233 images, many images share high similarity.
  This is expected for a dense multi-label dataset.

- **MRR = 0.478**: On average, the first correct result is around rank 2–3.

### 7.3 Per-Label Performance

| Label | Recall@10 | Images | Notes |
|-------|-----------|--------|-------|
| Normal | 100% | 17 | Easily distinguished |
| Cardiomegaly | 100% | 24 | Enlarged heart is visually distinct |
| Effusion | 100% | 21 | Clear pleural fluid pattern |
| Atelectasis | 100% | 35 | Well-represented |
| Pneumothorax | 100% | 4 | Despite few images, distinct appearance |
| Edema | 100% | 6 | Fluid patterns well-captured |
| Nodule | 100% | 6 | Focal lesion detected |
| Infiltration | 100% | 12 | Good retrieval |
| Fibrosis | 100% | 30 | Chronic changes recognized |
| Pleural Thickening | 100% | 12 | Good performance |
| Hernia | 100% | 19 | Visually distinctive |
| Mass | 66.7% | 7 | Sometimes confused with Nodule |
| Pneumonia | 33.3% | 5 | Often confused with Consolidation |
| Consolidation | 33.3% | 21 | Confused with Infiltration |
| Emphysema | 33.3% | 14 | Subtle findings harder to distinguish |

### 7.4 Hard Cases Summary

| Statistic | Count |
|-----------|-------|
| Total queries | 45 |
| Strict failures (top-1 wrong) | 35 |
| Semantic matches (medically related) | 3 |
| True failures | 32 |
| Top confusion patterns | Pneumonia ↔ Consolidation, Atelectasis ↔ Effusion |

The 35 "strict failures" are misleading—they mean the top-1 result wasn't
an exact label match, but by top-10, 84.4% of queries found a correct result.
The 3 semantic matches (e.g., Pneumonia → Consolidation) are clinically
reasonable since pneumonia typically manifests as consolidation.

---

## 8. Technology Stack

### 8.1 Complete Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **AI Model** | BiomedCLIP | — | Vision-language embeddings |
| **Model Framework** | PyTorch + open_clip | 2.0+ | Model loading & export |
| **ONNX Export** | torch.onnx | opset 18 | Convert to portable format |
| **Browser Runtime** | ONNX Runtime Web | latest (CDN) | WASM-based model inference |
| **Tokenizer** | Custom JS (WordPiece) | — | BERT tokenization in browser |
| **Frontend** | Next.js | 15 | React framework (App Router) |
| **Language** | TypeScript | 5.x | Type-safe frontend |
| **Styling** | Tailwind CSS | 3.x | Utility-first CSS |
| **Components** | shadcn/ui | latest | Accessible component library |
| **Charts** | Recharts | 2.x | Metrics visualization |
| **Data Pipeline** | Python | 3.12 | Offline processing |
| **Dataset** | NIH ChestX-ray14 | 2017 | Benchmark chest X-rays |
| **Deployment** | Static export | — | GitHub Pages / Vercel |

### 8.2 Browser Payload

| Asset | Size | Caching |
|-------|------|---------|
| ONNX model | ~420 MB | Cached after first load |
| Embeddings | 466 KB | Per session |
| Metadata + JSON | ~650 KB | Per session |
| Images (233 WebP) | ~1.1 MB | Browser cache |
| **Total first visit** | **~422 MB** | — |
| **Subsequent visits** | **~2.2 MB** | Model cached |

---

## 9. Key Design Decisions

### 9.1 Why BiomedCLIP over Generic CLIP?

BiomedCLIP is pre-trained on **15 million biomedical image-text pairs** from
PubMed Central, compared to generic CLIP's training on natural images.
For medical queries like "pleural effusion" or "cardiomegaly", BiomedCLIP
produces significantly better embeddings because it understands medical
terminology and visual patterns.

### 9.2 Why ONNX Runtime Web Instead of Transformers.js?

We initially attempted to use Hugging Face's `@xenova/transformers` library,
but it couldn't load BiomedCLIP because:

1. BiomedCLIP is an `open_clip` model, not a standard HuggingFace `transformers` model
2. It lacks the expected `config.json` format for auto-detection
3. The fallback to generic CLIP defeated the purpose of using a medical model

**Solution**: Direct ONNX Runtime Web loaded from CDN, with a custom BERT
tokenizer implemented in JavaScript. This gives us full control and avoids
all webpack bundling conflicts with native Node.js modules.

### 9.3 Why Pre-Computed Fallbacks?

The ONNX model takes 5–30 seconds on first inference (model download + WASM
compilation). Pre-computed results ensure the demo is **always responsive**:

- 55 common clinical queries have instant results
- Fuzzy matching catches variations (e.g., "chest xray" matches "normal chest xray")
- The AI model handles truly novel queries that pre-computation can't cover

### 9.4 Why L2 Normalization?

All embedding vectors are L2-normalized during index generation. This enables
**dot product = cosine similarity** in the browser, which is computationally
cheaper than computing full cosine similarity (avoids division by norms at
query time).

### 9.5 Why Static Export?

Next.js is configured for `output: 'export'`, producing pure static HTML/JS/CSS.
This means:

- No Node.js server needed at runtime
- Deployable to any static hosting (GitHub Pages, Vercel, S3, etc.)
- Zero ongoing server costs
- CDN-cacheable for global performance

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **233 images** (small subset) | Limited label diversity per class | Can scale to full 112K with CHPC GPU |
| **FP32 model** (420 MB) | Large first download | Browser caches model; one-time cost |
| **Text encoder only** in browser | Can't do image upload search in real-time | "Find Similar" uses pre-computed neighbors |
| **Single modality** (chest X-ray) | Only one imaging type | Architecture supports any image type |
| **First inference latency** (~10s) | User waits on novel queries | Pre-computed fallbacks cover common cases |
| **Multi-label evaluation** | Primary label used for display; evaluation simplified | All labels considered for Recall computation |

### 10.2 Future Work

| Enhancement | Approach | Expected Impact |
|-------------|----------|-----------------|
| **Full dataset** (112K images) | Process on CHPC GPU cluster | Better per-label coverage, higher mAP |
| **INT8 quantization** | Custom ONNX graph surgery | ~100 MB model (4× smaller) |
| **Model caching** | IndexedDB persistence | Instant repeat-visit inference |
| **Image upload search** | Export visual encoder to ONNX, run in browser | Real image-to-image retrieval |
| **Multi-modality** | Add CT, MRI, pathology datasets | Broader clinical utility |
| **Fine-tuning** | LoRA adaptation on chest X-ray reports | Higher mAP on domain-specific queries |
| **Progressive loading** | Stream embeddings, load images on demand | Faster initial page load |
| **Approximate NN** | HNSW index in WASM | Sub-millisecond search for >50K images |

### 10.3 Scaling Estimates

| Scenario | Images | Embedding File | Index Time | Browser Search |
|----------|--------|---------------|------------|---------------|
| Current demo | 233 | 466 KB | ~20s (CPU) | < 1ms |
| Medium | 10,000 | ~20 MB | ~15 min (CPU) | ~5ms |
| Full NIH | 112,000 | ~220 MB | ~3 hrs (GPU) | ~50ms |
| Multi-dataset | 500,000 | ~1 GB | ~12 hrs (GPU) | ~200ms* |

*Would require approximate nearest neighbor (ANN) indexing for real-time search.

---

## 11. References

1. **BiomedCLIP**: Zhang, S., et al. "BiomedCLIP: A multimodal biomedical
   foundation model pretrained from fifteen million scientific image-text pairs."
   *arXiv:2303.00915*, 2023.

2. **NIH ChestX-ray14**: Wang, X., et al. "ChestX-ray8: Hospital-scale Chest
   X-ray Database and Benchmarks on Weakly-Supervised Classification and
   Localization of Common Thorax Diseases." *CVPR*, 2017.

3. **CLIP**: Radford, A., et al. "Learning Transferable Visual Models From
   Natural Language Supervision." *ICML*, 2021.

4. **ONNX Runtime**: Microsoft. "ONNX Runtime: cross-platform, high performance
   ML inferencing and training accelerator." https://onnxruntime.ai/

5. **OpenCLIP**: Ilharco, G., et al. "OpenCLIP." 2021. https://github.com/mlfoundations/open_clip

---

<p align="center">
  <i>Med-MIR — Privacy-preserving medical image retrieval at zero cost.</i>
</p>
