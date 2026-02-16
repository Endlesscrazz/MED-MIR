# Browser Benchmark Guide

Use this to generate runtime measurements in the actual browser environment.

## Route
- Open: `/benchmark`

## What it measures
- `index_load`: load of embeddings + metadata + fallback + neighbors
- `worker_init_text`: text model worker initialization
- `fallback_lookup`: cached query path lookup latency
- `ai_text_embedding`: text encoder inference time
- `ai_text_vector_search`: vector search over all embeddings
- `ai_text_total`: text embedding + vector search
- `ai_vision_preprocess`: image fetch + normalization/preprocess
- `ai_vision_embedding`: vision encoder inference time
- `ai_vision_vector_search`: vector search over all embeddings
- `ai_vision_total`: preprocess + embedding + vector search

## How to run
1. Start app:
   - `cd web && npm run dev`
2. Open:
   - `http://localhost:3000/benchmark`
3. Click:
   - `Run Benchmark`
4. Download:
   - `browser_benchmark_report.json`
   - `browser_benchmark_report.md`

## Recommended test matrix
- Run each benchmark 3 times and keep median.
- Test at least:
  - your local dev machine
  - one lower-power machine if available
  - one mobile browser if relevant to demo scope

## Notes
- First run can be slower due to model download/compile/cache warmup.
- Image-query timing includes lazy loading of vision model if not cached.
