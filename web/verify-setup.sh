#!/bin/bash
# Verification script for Med-MIR setup

echo "🔍 Verifying Med-MIR Setup..."
echo ""

# Check data files
echo "📁 Checking data files..."
MISSING=0

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(ls -lh "$1" | awk '{print $5}')
        echo "  ✅ $1 ($SIZE)"
    else
        echo "  ❌ $1 (MISSING)"
        MISSING=$((MISSING + 1))
    fi
}

check_file "public/demo-data/embeddings.bin"
check_file "public/demo-data/metadata.json"
check_file "public/demo-data/fallback_results.json"
check_file "public/demo-data/nearest_neighbors.json"
check_file "public/demo-data/index_info.json"

# Check images
if [ -d "public/demo-data/images" ]; then
    IMG_COUNT=$(ls public/demo-data/images/*.webp 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✅ public/demo-data/images/ ($IMG_COUNT images)"
else
    echo "  ❌ public/demo-data/images/ (MISSING)"
    MISSING=$((MISSING + 1))
fi

echo ""

# Check dependencies
echo "📦 Checking dependencies..."
if [ -d "node_modules" ]; then
    echo "  ✅ node_modules/ (dependencies installed)"
else
    echo "  ⚠️  node_modules/ (not installed)"
    echo "     Run: npm install (or pnpm install)"
    MISSING=$((MISSING + 1))
fi

echo ""

# Summary
if [ $MISSING -eq 0 ]; then
    echo "✅ All files verified! Ready to test."
    echo ""
    echo "Next steps:"
    echo "  1. npm install (if not done)"
    echo "  2. npm run dev"
    echo "  3. Open http://localhost:3000"
else
    echo "⚠️  $MISSING item(s) missing. Please fix before testing."
fi
