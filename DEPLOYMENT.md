# Med-MIR Deployment Guide

This guide explains how to deploy Med-MIR to production using Vercel (frontend) and GitHub Pages (assets).

## Prerequisites

1. GitHub account
2. Vercel account (free tier)
3. Completed Python pipeline with generated data files

## Step 1: Generate Data (Python Pipeline)

```bash
cd python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset
python download_data.py --num_images 500

# Process images
python process_images.py

# Generate index
python generate_index.py

# Export ONNX model (optional - for custom model)
python export_onnx.py

# Generate metrics
python evaluate.py
```

## Step 2: Create Asset Repository on GitHub Pages

1. Create a new GitHub repository (e.g., `med-mir-assets`)
2. Enable GitHub Pages in repository settings (Settings > Pages > Source: main branch)
3. Upload the following files from `output/`:
   - `embeddings.bin`
   - `metadata.json`
   - `fallback_results.json`
   - `nearest_neighbors.json`
   - `index_info.json`
   - `metrics.json`
   - `hard_cases.json`
   - `images/` directory (all WebP files)

```bash
# Example upload script
cd output
git init
git add .
git commit -m "Initial data upload"
git remote add origin https://github.com/YOUR_USERNAME/med-mir-assets.git
git push -u origin main
```

Your assets will be available at: `https://YOUR_USERNAME.github.io/med-mir-assets/`

## Step 3: Update Metadata URLs

Before deploying, update `metadata.json` to use the GitHub Pages URLs:

```python
# In Python
import json

with open('output/metadata.json') as f:
    metadata = json.load(f)

BASE_URL = 'https://YOUR_USERNAME.github.io/med-mir-assets'

for item in metadata:
    item['url'] = f"{BASE_URL}/images/{item['filename']}"

with open('output/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

Re-upload the updated `metadata.json` to your asset repository.

## Step 4: Configure Next.js

Create `.env.local` in the `web/` directory:

```env
NEXT_PUBLIC_DATA_URL=https://YOUR_USERNAME.github.io/med-mir-assets
```

## Step 5: Deploy to Vercel

### Option A: Via Vercel CLI

```bash
cd web
npm install -g vercel
vercel login
vercel --prod
```

### Option B: Via GitHub Integration

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your GitHub repository
5. Set the root directory to `web`
6. Add environment variable:
   - `NEXT_PUBLIC_DATA_URL` = `https://YOUR_USERNAME.github.io/med-mir-assets`
7. Deploy

## Step 6: Verify Deployment

1. Visit your Vercel deployment URL
2. Check browser console for any errors
3. Test a search query
4. Verify images load from GitHub Pages
5. Check the Metrics and Hard Cases pages

## Troubleshooting

### Images Not Loading
- Verify GitHub Pages is enabled
- Check that images are in `images/` subdirectory
- Ensure metadata.json has correct URLs

### CORS Errors
- GitHub Pages should handle CORS automatically
- If issues persist, check browser network tab

### Model Loading Slow
- First load downloads ~100MB model
- Model is cached in IndexedDB for subsequent visits
- Consider showing progress indicator

### Static Export Issues
- Ensure `next.config.js` has `output: 'export'`
- Check for any dynamic API routes (not supported)

## Estimated Costs

| Service | Tier | Cost |
|---------|------|------|
| Vercel | Hobby | $0 |
| GitHub Pages | Free | $0 |
| **Total** | | **$0** |

## Performance Optimization

1. **Image Compression**: WebP format at 256px keeps images small
2. **Model Caching**: transformers.js caches model in browser
3. **Fallback Strategy**: Pre-computed results for common queries
4. **CDN**: Both Vercel and GitHub Pages use global CDNs

## Security Notes

- No backend servers = no server vulnerabilities
- No data leaves user's browser
- Medical images never transmitted to external servers
- HIPAA-friendly architecture (no PHI processing server-side)
