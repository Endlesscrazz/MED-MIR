# Med-MIR

**Medical Multimodal Image Retrieval**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx)](https://onnxruntime.ai/)
[![Transformers.js](https://img.shields.io/badge/Transformers.js-Client--Side_AI-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers.js)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Deployment](https://img.shields.io/badge/Cost-$0-brightgreen?style=flat-square)](https://vercel.com/)

A **serverless, privacy-preserving, local-first** medical image retrieval system. Users can type clinical queries (e.g., *"Pleural Effusion"*, *"Cardiomegaly"*) or explore similar cases to find semantically relevant medical images—all without sending data to external servers.

> **Research Context**: This project demonstrates the feasibility of deploying medical AI systems at the edge, eliminating infrastructure costs while maintaining strict data privacy—critical considerations for clinical and research applications.

---

## How It Works

Med-MIR employs a **Hybrid Architecture** that combines offline pre-computation with real-time client-side inference, achieving both speed and flexibility.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID ARCHITECTURE                               │
├─────────────────────────────────┬───────────────────────────────────────────┤
│     OFFLINE PIPELINE            │         ONLINE APPLICATION                │
│     (Run Once)                  │         (User's Browser)                  │
├─────────────────────────────────┼───────────────────────────────────────────┤
│                                 │                                           │
│  ┌─────────────┐                │      User Query: "Pneumonia"              │
│  │  Open-I     │                │              │                            │
│  │  Dataset    │                │              ▼                            │
│  └──────┬──────┘                │    ┌─────────────────────┐                │
│         │                       │    │ Fallback Lookup?    │                │
│         ▼                       │    └──────────┬──────────┘                │
│  ┌─────────────┐                │         Yes   │   No                      │
│  │ BiomedCLIP  │                │           │   │   │                       │
│  │  Encoder    │                │           │   │   ▼                       │
│  └──────┬──────┘                │           │   │  ┌──────────────┐         │
│         │                       │           │   │  │ Web Worker   │         │
│         ▼                       │           │   │  │ ONNX Encode  │         │
│  ┌─────────────┐                │           │   │  └──────┬───────┘         │
│  │ embeddings  │───────────────────────────────────────▶  │                 │
│  │   .bin      │                │           │      Dot    │                 │
│  └─────────────┘                │           │    Product  │                 │
│  ┌─────────────┐                │           │             │                 │
│  │ fallback_   │───────────────────────▶ ◄──┴─────────────┘                 │
│  │ results.json│                │        │                                  │
│  └─────────────┘                │        ▼                                  │
│                                 │   Top-K Results                           │
│                                 │                                           │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### The Two-Stage Approach

**Stage 1: Offline Pre-computation** (Python)
- Process the Open-I chest X-ray dataset through [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- Generate L2-normalized 768-dimensional embeddings for all images
- Pre-compute results for 50 common clinical queries as a "safety net"
- Export the text encoder to ONNX format for browser inference

**Stage 2: Client-Side Inference** (Browser)
- **Fast Path**: Query matches pre-computed results → instant response
- **Full Path**: Novel query → ONNX inference in Web Worker → vector similarity search
- All computation happens locally; no data leaves the user's device

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Real-Time Semantic Search** | Natural language queries matched against medical image embeddings using cosine similarity |
| **Zero-Cost Deployment** | Static export to Vercel + GitHub Pages for asset hosting—no GPU servers required |
| **Privacy-First Architecture** | All AI inference runs in the browser via Web Workers; medical data never transmitted |
| **Hybrid Retrieval Strategy** | Pre-computed fallbacks for common queries ensure reliability; ONNX inference for novel queries |
| **Find Similar** | One-click discovery of visually similar cases using pre-computed nearest neighbors |
| **Reliability Metrics** | Built-in dashboard showing Recall@K, mAP, and model performance analysis |
| **Hard Case Analysis** | Transparent reporting of failure modes (e.g., conditions the model confuses) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 15 (App Router), TypeScript, Tailwind CSS, shadcn/ui |
| **AI Inference** | Transformers.js, ONNX Runtime Web, Web Workers |
| **Data Pipeline** | Python 3.10+, PyTorch, HuggingFace Transformers, BiomedCLIP |
| **Deployment** | Vercel (static hosting), GitHub Pages (assets) |

---

## Installation

> **Prerequisites**: Python 3.10+, Node.js 18+, pnpm (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Med-MIR.git
cd Med-MIR
```

### 2. Set Up the Python Pipeline

```bash
cd python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate the Index (Requires Dataset)

```bash
# Download dataset (N=500 images)
python download_data.py

# Process images and generate embeddings
python process_images.py
python generate_index.py

# Export ONNX model
python export_onnx.py
```

### 4. Run the Web Application

```bash
cd ../web
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to explore the application.

---

## Project Structure

```
Med-MIR/
├── python/                    # Offline data processing pipeline
│   ├── download_data.py       # Dataset acquisition
│   ├── process_images.py      # Image optimization (WebP, 256px)
│   ├── generate_index.py      # Embedding generation
│   └── export_onnx.py         # ONNX model conversion
│
├── web/                       # Next.js application
│   ├── src/
│   │   ├── app/               # Pages (search, metrics, hard-cases)
│   │   ├── components/        # UI components
│   │   ├── workers/           # Web Worker for ONNX inference
│   │   └── lib/               # Utilities (binary loader, search logic)
│   └── public/                # Static assets
│
├── PROJECT_SPEC.md            # Technical specification
├── TASKS.md                   # Development task tracking
└── CHANGELOG.md               # Version history
```

---

## Acknowledgments

- **Dataset**: [Open-I](https://openi.nlm.nih.gov/) - Indiana University Chest X-ray Collection
- **Model**: [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) by Microsoft Research
- **Inference**: [Transformers.js](https://huggingface.co/docs/transformers.js) by Hugging Face

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with a focus on privacy, accessibility, and zero-cost deployment for medical AI research.</i>
</p>
