#!/bin/bash
# Helper script to regenerate index with proper label extraction

set -e

echo "🔄 Regenerating Med-MIR index with label extraction..."
echo ""

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Virtual environment not active!"
    echo "   Please run: source med_mir/bin/activate"
    echo ""
    exit 1
fi

# Check if raw metadata exists
if [[ ! -f "python/data/raw/raw_metadata.json" ]]; then
    echo "❌ raw_metadata.json not found!"
    echo "   Expected: python/data/raw/raw_metadata.json"
    echo ""
    exit 1
fi

# Check if images exist
if [[ ! -d "python/output/images" ]]; then
    echo "❌ Processed images directory not found!"
    echo "   Expected: python/output/images/"
    echo "   Run: python python/process_images.py"
    echo ""
    exit 1
fi

echo "📁 Found raw metadata and images"
echo ""

# Regenerate index with proper paths
echo "🔧 Generating embeddings and metadata..."
python python/generate_index.py \
  --images_dir python/output/images \
  --raw_metadata python/data/raw/raw_metadata.json \
  --output_dir python/output

echo ""
echo "✅ Index regeneration complete!"
echo ""

# Copy to web app
echo "📋 Copying files to web app..."
cp python/output/metadata.json web/public/demo-data/metadata.json
cp python/output/embeddings.bin web/public/demo-data/embeddings.bin
cp python/output/fallback_results.json web/public/demo-data/fallback_results.json
cp python/output/nearest_neighbors.json web/public/demo-data/nearest_neighbors.json
cp python/output/index_info.json web/public/demo-data/index_info.json

echo "✅ Files copied to web/public/demo-data/"
echo ""
echo "🎉 Done! Restart your dev server to see labels."
echo ""
echo "Next steps:"
echo "  1. Restart dev server: cd web && npm run dev"
echo "  2. Open http://localhost:3000"
echo "  3. Search and verify labels are showing"
