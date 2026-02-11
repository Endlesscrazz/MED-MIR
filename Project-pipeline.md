# Med-MIR: End-to-End Data Pipeline

> This document describes the complete data ingestion, preprocessing, labelling,
> embedding generation, evaluation, and deployment pipeline for the Med-MIR
> medical image retrieval system.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE (Python)                     │
│  Runs once on local machine or university CHPC cluster           │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Download  │───▶│ Process  │───▶│ Generate │───▶│ Evaluate │  │
│  │ NIH Data │    │ Images   │    │  Index   │    │ Metrics  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│   labels.csv     *.webp         embeddings.bin    metrics.json  │
│   images/        (256px)        metadata.json     hard_cases.json│
│                                 fallback.json                    │
│                                 neighbors.json                   │
└──────────────────────────────────────────────────────────────────┘
                          │
                   Copy to web/public/demo-data/
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ONLINE APPLICATION (Next.js)                  │
│  Runs in user's browser — zero server, zero cost                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐     │
│  │ Load Index   │───▶│ Search       │───▶│ Display Results │     │
│  │ (binary)     │    │ (dot product │    │ (grid + cards)  │     │
│  │              │    │  or fallback)│    │                 │     │
│  └──────────────┘    └──────────────┘    └────────────────┘     │
│         │                   │                                    │
│  embeddings.bin      BiomedCLIP ONNX                             │
│  metadata.json       (Web Worker)                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Source: NIH ChestX-ray14

| Property              | Value                                          |
|-----------------------|------------------------------------------------|
| **Full name**         | NIH Clinical Center Chest X-ray Dataset        |
| **Paper**             | Wang et al., CVPR 2017                         |
| **Total images**      | 112,120 frontal-view X-rays                    |
| **Patients**          | 30,805 unique                                  |
| **Labels**            | 14 pathologies + "No Finding" (multi-label)    |
| **Label accuracy**    | >90% (NLP-extracted from radiology reports)    |
| **License**           | CC0 1.0 (public domain)                        |
| **Access**            | Free — Kaggle account required                 |
| **Download method**   | Kaggle CLI (`kaggle datasets download`)         |
| **Citation count**    | 10,000+                                        |

### 2.1 The 15 Labels

| Label              | Approx. Count | Display Name       |
|--------------------|---------------|--------------------|
| No Finding         | ~60,361       | Normal             |
| Infiltration       | ~19,894       | Infiltration       |
| Effusion           | ~13,317       | Effusion           |
| Atelectasis        | ~11,559       | Atelectasis        |
| Nodule             | ~6,331        | Nodule             |
| Mass               | ~5,782        | Mass               |
| Pneumothorax       | ~5,302        | Pneumothorax       |
| Consolidation      | ~4,667        | Consolidation      |
| Pleural_Thickening | ~3,385        | Pleural Thickening |
| Cardiomegaly       | ~2,776        | Cardiomegaly       |
| Emphysema          | ~2,516        | Emphysema          |
| Edema              | ~2,303        | Edema              |
| Fibrosis           | ~1,686        | Fibrosis           |
| Pneumonia          | ~1,431        | Pneumonia          |
| Hernia             | ~227          | Hernia             |

### 2.2 Multi-Label Structure

Each image can have **multiple labels** separated by `|`:

```
00000013_005.png  →  "Atelectasis|Effusion"
00000032_001.png  →  "Cardiomegaly|Effusion|Edema"
00000076_000.png  →  "No Finding"
```

For our system:
- **Display label**: The first (primary) label is shown in the UI
- **Evaluation**: An image is considered "relevant" if **any** of its labels matches the query
- **Report snippet**: Synthesized from the label list (e.g., "Findings: Atelectasis, Effusion")

---

## 3. Pipeline Steps

### Step 1: Data Ingestion (`download_nih.py`)

**Purpose**: Download a class-balanced subset of NIH ChestX-ray14.

**Method**: Two-step process designed for limited MacBook storage:

1. **Download** the full dataset (~45 GB) to an **external hard drive** via
   the Kaggle CLI
2. **Select** a balanced subset (~2,500 images) and copy only those to the
   local project directory

**Balancing strategy**: Parse the `Data_Entry_2017_v2020.csv` label file,
group images by label, and randomly sample up to 150 images per label.
Multi-label images count toward all their labels.

```bash
cd python

# Step 1: Download to external drive (one-time, ~45 GB)
python download_nih.py --kaggle --output_dir /Volumes/MyDrive/nih-data

# Step 2: Select balanced subset → copy to project (~1.2 GB of PNGs)
python download_nih.py --select-subset \
  --source_dir /Volumes/MyDrive/nih-data \
  --output_dir data/nih \
  --per_label 150
```

**Output**:
```
data/nih/
├── images/          # ~2,000-2,500 PNG images (1024×1024)
└── labels.csv       # Columns: filename, labels, primary_label
```

**labels.csv format**:
```csv
filename,labels,primary_label
00000013_005.png,Atelectasis|Effusion,Atelectasis
00000076_000.png,No Finding,No Finding
00000032_001.png,Cardiomegaly|Effusion|Edema,Cardiomegaly
```

### Step 2: Image Processing (`process_images.py`)

**Purpose**: Optimise raw PNG images for web delivery.

**Transforms**:
1. Resize to max 256px (height or width), preserving aspect ratio
2. Convert to RGB colour space
3. Save as WebP (quality 85, method 6)

```bash
python process_images.py \
  --input_dir data/nih/images \
  --output_dir output/images \
  --max_dim 256 \
  --quality 85
```

**Result**: ~80% size reduction. 1024×1024 PNG (~400 KB) → 256×256 WebP (~8 KB).

**Output**:
```
output/images/
├── nih_00000.webp
├── nih_00001.webp
└── ...
```

### Step 3: Embedding Generation & Index Building (`generate_index.py`)

**Purpose**: Create the search index used by the browser application.

**Model**: BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
- Visual encoder: ViT-B/16 (224px input)
- Text encoder: PubMedBERT
- Embedding dimension: 512
- Pre-trained on 15M biomedical image–text pairs from PubMed

**Steps**:
1. **Load model** on CPU (or CUDA if available)
2. **Generate image embeddings**: Pass each WebP image through BiomedCLIP's
   visual encoder → 512-dim vector
3. **L2-normalise** all vectors (critical: enables dot product = cosine
   similarity in the browser)
4. **Save embeddings** as raw `Float32Array` binary (`embeddings.bin`)
5. **Build metadata** from NIH labels.csv → `metadata.json`
6. **Pre-compute fallback results** for 50+ common clinical queries →
   `fallback_results.json`
7. **Compute nearest neighbours** for image-to-image similarity →
   `nearest_neighbors.json`

```bash
python generate_index.py \
  --images_dir output/images \
  --nih_labels data/nih/labels.csv \
  --output_dir output
```

**Output**:
```
output/
├── embeddings.bin          # Float32Array (N × 512), ~5 MB for 2,500 images
├── metadata.json           # Image metadata with labels
├── fallback_results.json   # Pre-computed top-10 for 50+ queries
├── nearest_neighbors.json  # Pre-computed 10 nearest neighbours per image
└── index_info.json         # Summary (num_images, dim, model name)
```

**metadata.json entry** (per image):
```json
{
  "id": 0,
  "filename": "nih_00000.webp",
  "url": "images/nih_00000.webp",
  "report_snippet": "Findings: Atelectasis, Effusion.",
  "ground_truth_label": "Atelectasis",
  "all_labels": ["Atelectasis", "Effusion"],
  "label_source": "dataset",
  "label_verified": true
}
```

### Step 4: ONNX Model Export (`export_onnx.py`)

**Purpose**: Convert BiomedCLIP's text encoder to ONNX format for browser
inference via `onnxruntime-web`.

**Steps**:
1. Load BiomedCLIP via `open_clip`
2. Wrap the text encoder in a torch Module
3. Export to ONNX (opset 18, dynamic batch axis)
4. Export tokenizer files (`vocab.txt`, `config.json`, etc.)

```bash
python export_onnx.py --output_dir output/model
```

**Output**:
```
output/model/
├── model_flat.onnx         # Text encoder (~420 MB FP32)
├── vocab.txt               # PubMedBERT vocabulary (30,522 tokens)
├── config.json             # Model config (context_length, vocab_size)
├── tokenizer_config.json   # HuggingFace tokenizer config
└── special_tokens_map.json # [CLS], [SEP], [PAD], [UNK]
```

### Step 5: Evaluation (`evaluate.py`)

**Purpose**: Compute retrieval quality metrics and identify failure cases.

**Methodology**:
- Generate text embeddings for 45 evaluation queries (3 per label × 15 labels)
- Compute cosine similarity against all image embeddings
- Measure how often the correct images appear in top-K results

**Metrics (three tiers)**:

| Tier       | Description                                             |
|------------|---------------------------------------------------------|
| **Strict** | Exact label match. All 45 queries evaluated.            |
| **Adjusted** | Excludes queries whose label has 0 images in dataset. |
| **Semantic** | Counts retrieval as correct if labels are in the same medical semantic group (e.g., Nodule ≈ Mass). |

**Multi-label evaluation**: An image with labels "Atelectasis|Effusion" is
counted as relevant for both "Atelectasis" and "Effusion" queries.

```bash
python evaluate.py \
  --embeddings output/embeddings.bin \
  --metadata output/metadata.json \
  --output_dir output
```

**Output**:
```
output/
├── metrics.json      # Recall@K, mAP, MRR (strict/adjusted/semantic)
└── hard_cases.json   # Failure analysis with confusion patterns
```

---

## 4. Deployment to Browser

### Copy artifacts to web app:
```bash
cp output/embeddings.bin      web/public/demo-data/
cp output/metadata.json       web/public/demo-data/
cp output/fallback_results.json web/public/demo-data/
cp output/nearest_neighbors.json web/public/demo-data/
cp output/metrics.json        web/public/demo-data/
cp output/hard_cases.json     web/public/demo-data/
cp output/index_info.json     web/public/demo-data/

# Model files (for live AI inference)
mkdir -p web/public/demo-data/model
cp output/model/model_flat.onnx         web/public/demo-data/model/
cp output/model/vocab.txt               web/public/demo-data/model/
cp output/model/config.json             web/public/demo-data/model/
cp output/model/tokenizer_config.json   web/public/demo-data/model/
cp output/model/special_tokens_map.json web/public/demo-data/model/

# Processed images
cp -r output/images/ web/public/demo-data/images/
```

### Start the dev server:
```bash
cd web
npm run dev
```

### Browser search flow:
1. Browser loads `embeddings.bin` (Float32Array) + `metadata.json`
2. User types query (e.g., "pneumonia")
3. **Fast path**: Check `fallback_results.json` — if match found, return instant results
4. **AI path**: Send query to Web Worker → ONNX text encoder → 512-dim vector → dot product against embeddings → top-10 results
5. Display results as image cards with labels, scores, and report snippets

---

## 5. Evaluation Metrics Explained

### Recall@K
> "Out of all queries, what fraction had at least one correct image in the top K?"

- **Recall@1** = Did the very first result match? (hardest)
- **Recall@10** = Was there a correct result in the top 10? (practical)
- **Recall@20** = Expanded view

### Mean Average Precision (mAP)
> "On average, how high were correct images ranked?"

Rewards systems that rank correct images higher, not just having them somewhere in top-K.

### Mean Reciprocal Rank (MRR)
> "On average, what was the rank of the first correct result?"

MRR = 1.0 means the correct image was always rank 1. MRR = 0.5 means it was usually rank 2.

### Semantic Matching
Medical images often have overlapping conditions. Retrieving an "Effusion" image for an "Edema" query is clinically reasonable (both involve fluid). The semantic tier captures this.

**Semantic groups**:
- {Effusion, Edema} — fluid-related
- {Nodule, Mass} — focal opacities
- {Consolidation, Infiltration} — airspace processes
- {Pneumonia, Consolidation, Infiltration} — infectious overlap
- {Atelectasis, Effusion} — common co-occurrence
- {Fibrosis, Pleural Thickening} — chronic scarring

---

## 6. File Summary

| File | Size (approx.) | Description |
|------|----------------|-------------|
| `embeddings.bin` | ~5 MB | Binary Float32 matrix (N × 512) |
| `metadata.json` | ~500 KB | Image metadata, labels, snippets |
| `fallback_results.json` | ~50 KB | Pre-computed top-10 for 50+ queries |
| `nearest_neighbors.json` | ~500 KB | Top-10 similar images per image |
| `index_info.json` | ~200 B | Summary metadata |
| `metrics.json` | ~5 KB | Evaluation metrics |
| `hard_cases.json` | ~20 KB | Failure analysis |
| `model_flat.onnx` | ~420 MB | BiomedCLIP text encoder (ONNX) |
| `vocab.txt` | ~220 KB | PubMedBERT vocabulary |
| `images/*.webp` | ~20 MB | Processed images (256px WebP) |

**Total browser payload**: ~450 MB first load (model cached after that), ~6 MB data files, ~20 MB images.

---

## 7. Quick Start (Full Pipeline)

```bash
# 1. Set up Python environment
cd python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download NIH ChestX-ray14 balanced subset (~2,250 images)
python download_nih.py --output_dir data/nih --per_label 150

# 3. Process images (resize + WebP conversion)
python process_images.py -i data/nih/images -o output/images

# 4. Generate search index
python generate_index.py \
  --images_dir output/images \
  --nih_labels data/nih/labels.csv \
  --output_dir output

# 5. Export ONNX model and tokenizer (if not already done)
python export_onnx.py --output_dir output/model

# 6. Run evaluation
python evaluate.py \
  --embeddings output/embeddings.bin \
  --metadata output/metadata.json \
  --output_dir output

# 7. Copy to web app
cp output/embeddings.bin output/metadata.json output/fallback_results.json \
   output/nearest_neighbors.json output/metrics.json output/hard_cases.json \
   output/index_info.json ../web/public/demo-data/
cp -r output/images/ ../web/public/demo-data/images/

# 8. Start web app
cd ../web
npm install
npm run dev
```

---

## 8. Scaling Notes

| Scenario | Images | Embedding Time | Browser Load |
|----------|--------|---------------|--------------|
| Demo (current) | ~2,500 | ~5 min (CPU) | ~6 MB data |
| Medium | ~10,000 | ~20 min (CPU) | ~25 MB data |
| Full NIH | ~112,000 | ~3 hrs (GPU) | ~250 MB data |

For datasets >10K images, use university CHPC GPUs for embedding generation.
The browser can handle up to ~50K images before search latency becomes
noticeable (>100ms per query).
