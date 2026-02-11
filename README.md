# Med-MIR

**Medical Multimodal Image Retrieval**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Web-005CED?style=flat-square&logo=onnx)](https://onnxruntime.ai/)
[![BiomedCLIP](https://img.shields.io/badge/BiomedCLIP-Microsoft-FFD21E?style=flat-square)](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Deployment](https://img.shields.io/badge/Cost-$0-brightgreen?style=flat-square)](https://vercel.com/)

A **serverless, privacy-preserving, local-first** medical image retrieval system powered by [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224). Users can type clinical queries (e.g., *"Pleural Effusion"*, *"Cardiomegaly"*) or explore visually similar cases—**all without sending data to external servers**.

> **Research Context**: This project demonstrates the feasibility of deploying state-of-the-art medical AI at the edge, eliminating infrastructure costs while maintaining strict data privacy—critical for clinical and research applications.

📄 See [ARCHITECTURE.md](ARCHITECTURE.md) for a full technical deep-dive.

---

## How It Works

Med-MIR employs a **Hybrid Architecture** combining offline pre-computation with real-time client-side inference.

```
┌──────────────────────────────────────────────────────────────────┐
│                  OFFLINE PIPELINE (Python, run once)              │
│                                                                  │
│  NIH ChestX-ray14 ──▶ Balanced ──▶ BiomedCLIP ──▶ Static Files  │
│  (112K images,        Subset       Encoder        (embeddings,   │
│   15 labels)          (233 imgs)                   metadata,     │
│                                                    ONNX model)   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ deployed as static files
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ONLINE APPLICATION (Browser)                     │
│                                                                  │
│  User Query ──▶ Fallback ──▶ [match?] ──YES──▶ Instant Results   │
│                  Lookup          │               (< 50ms)        │
│                                 NO                               │
│                                 │                                │
│                                 ▼                                │
│                           Web Worker                             │
│                        ONNX Inference                            │
│                     (custom tokenizer +                          │
│                      BiomedCLIP text                             │
│                       encoder via WASM)                          │
│                                 │                                │
│                                 ▼                                │
│                       Cosine Similarity                          │
│                     vs 233 image embeddings                      │
│                                 │                                │
│                                 ▼                                │
│                        Ranked Results                            │
└──────────────────────────────────────────────────────────────────┘
```

**Two search paths**:
- **Fast Path**: Query matches one of 55 pre-computed clinical queries → instant results (< 50ms)
- **AI Path**: Novel query → ONNX model runs in a Web Worker via WebAssembly → cosine similarity search → ranked results

All computation happens locally. No data leaves the user's device.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-Time Semantic Search** | Natural language queries matched against medical image embeddings using cosine similarity |
| 💰 **Zero-Cost Deployment** | Static export to Vercel / GitHub Pages—no GPU servers required |
| 🔒 **Privacy-First Architecture** | All AI inference runs in the browser via Web Workers; medical data never transmitted |
| ⚡ **Hybrid Retrieval Strategy** | Pre-computed fallbacks for common queries; ONNX inference for novel queries |
| 🔗 **Find Similar** | One-click discovery of visually similar cases using pre-computed nearest neighbors |
| 📊 **Reliability Metrics** | Dashboard with Recall@K, mAP, MRR across strict/adjusted/semantic tiers |
| 🧪 **Hard Case Analysis** | Transparent reporting of model failures with semantic-match annotations |
| 🏷️ **Verified Labels** | NIH ChestX-ray14 dataset with 15 peer-reviewed diagnostic labels |

---

## Evaluation Results

Evaluated on 233 NIH ChestX-ray14 images across 15 diagnostic labels with 45 clinical queries:

| Metric | Strict | Semantic |
|--------|--------|----------|
| **Recall@1** | 28.9% | 37.8% |
| **Recall@5** | 71.1% | 86.7% |
| **Recall@10** | **84.4%** | **95.6%** |
| **Recall@20** | 95.6% | 97.8% |
| **mAP** | 22.3% | — |
| **MRR** | 0.478 | 0.593 |

> **Recall@10 = 84.4%** means: for 84% of queries, at least one relevant image appears in the top 10 results. With semantic matching (counting medically-related retrievals), this rises to **95.6%**.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **AI Model** | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (PubMedBERT + ViT-B/16) |
| **Browser Inference** | ONNX Runtime Web (WASM) + Web Workers + Custom BERT Tokenizer |
| **Frontend** | Next.js 15 (App Router), TypeScript, Tailwind CSS, shadcn/ui |
| **Charts** | Recharts |
| **Data Pipeline** | Python 3.12, PyTorch, open_clip |
| **Dataset** | [NIH ChestX-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) (112K images, 15 labels) |
| **Deployment** | Static export → Vercel / GitHub Pages |

---

## Quick Start

> **Prerequisites**: Python 3.10+, Node.js 18+, [Kaggle account](https://www.kaggle.com/) (for dataset)

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/yourusername/Med-MIR.git
cd Med-MIR

python -m venv med_mir
source med_mir/bin/activate    # Windows: med_mir\Scripts\activate
pip install -r python/requirements.txt
```

### 2. Download the Dataset

```bash
cd python

# Option A: Download full NIH dataset to external drive (~45 GB)
pip install kaggle
python download_nih.py --kaggle --output_dir /Volumes/YourDrive/nih-data

# Then select a balanced subset (~233 images across 15 labels)
python download_nih.py --select-subset \
  --source_dir /Volumes/YourDrive/nih-data \
  --output_dir data/nih \
  --per_label 150
```

### 3. Run the Pipeline

```bash
# Process images (resize + WebP)
python process_images.py -i data/nih/images -o output/images

# Generate search index + embeddings
python generate_index.py \
  --images_dir output/images \
  --nih_labels data/nih/labels.csv \
  --output_dir output

# Export ONNX model for browser inference
python export_onnx.py --output_dir output/model

# Run evaluation
python evaluate.py \
  --embeddings output/embeddings.bin \
  --metadata output/metadata.json \
  --output_dir output
```

### 4. Deploy to Web App

```bash
# Copy all artifacts to the web app's public directory
cp output/embeddings.bin output/metadata.json output/fallback_results.json \
   output/nearest_neighbors.json output/metrics.json output/hard_cases.json \
   output/index_info.json ../web/public/demo-data/
cp -r output/images/ ../web/public/demo-data/images/
mkdir -p ../web/public/demo-data/model
cp output/model/* ../web/public/demo-data/model/
```

### 5. Run the Web Application

```bash
cd ../web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to explore.

---

## Project Structure

```
Med-MIR/
├── python/                        # Offline data pipeline
│   ├── download_nih.py            # NIH ChestX-ray14 download + subset selection
│   ├── process_images.py          # Image optimization (PNG → WebP, 256px)
│   ├── generate_index.py          # BiomedCLIP embedding generation + index
│   ├── export_onnx.py             # ONNX model export (text encoder + tokenizer)
│   ├── evaluate.py                # Recall@K, mAP, MRR + hard case analysis
│   └── requirements.txt           # Python dependencies
│
├── web/                           # Next.js application
│   ├── src/
│   │   ├── app/                   # Pages: search (/), metrics, hard-cases
│   │   ├── components/            # UI: SearchBox, ResultCard, Header, etc.
│   │   └── lib/                   # Core logic: search, similarity, binary-loader
│   │       └── hooks/             # useSearch hook (Web Worker management)
│   ├── public/
│   │   ├── demo-data/             # Embeddings, metadata, model, images (gitignored)
│   │   └── workers/               # inference.worker.js (ONNX + custom tokenizer)
│   └── package.json
│
├── ARCHITECTURE.md                # Full technical deep-dive
├── Project-pipeline.md            # Pipeline documentation
├── DECISIONS.md                   # Architectural decision records
└── README.md                      # This file
```

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Main search interface with text-to-image and "Find Similar" |
| `/metrics` | Reliability dashboard: Recall@K charts, per-label performance, methodology |
| `/hard-cases` | Failure analysis: confusion patterns, semantic matches, detailed examples |

---

## Acknowledgments

- **Dataset**: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — NIH Clinical Center (Wang et al., CVPR 2017)
- **Model**: [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Microsoft Research (Zhang et al., 2023)
- **Inference**: [ONNX Runtime Web](https://onnxruntime.ai/) — Microsoft
- **UI**: [shadcn/ui](https://ui.shadcn.com/) + [Tailwind CSS](https://tailwindcss.com/)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with a focus on privacy, accessibility, and zero-cost deployment for medical AI research.</i>
</p>
