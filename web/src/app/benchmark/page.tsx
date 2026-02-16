'use client';

import { useState } from 'react';
import { loadIndexData, findTopK } from '@/lib/binary-loader';
import { checkFallback } from '@/lib/search';
import { preprocessImage } from '@/lib/image-processor';

type BenchReport = {
  run_at: string;
  dataset: {
    num_images: number;
    embedding_dim: number;
    fallback_queries: number;
  };
  timings_ms: {
    index_load: number;
    worker_init_text: number;
    fallback_lookup: number;
    ai_text_embedding: number;
    ai_text_vector_search: number;
    ai_text_total: number;
    ai_vision_preprocess: number;
    ai_vision_embedding: number;
    ai_vision_vector_search: number;
    ai_vision_total: number;
  };
  checks: {
    fallback_query: string;
    fallback_hit_count: number;
    ai_text_query: string;
    ai_text_top1_score: number | null;
    ai_vision_image: string;
    ai_vision_top1_score: number | null;
  };
};

type WorkerResolve = (embedding: Float32Array) => void;

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function BenchmarkPage() {
  const dataBaseUrl = process.env.NEXT_PUBLIC_DATA_URL || '/demo-data';
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<BenchReport | null>(null);
  const [status, setStatus] = useState('Idle');

  const runBenchmark = async () => {
    setIsRunning(true);
    setError(null);
    setReport(null);

    let worker: Worker | null = null;
    const pending = new Map<string, WorkerResolve>();

    try {
      setStatus('Loading index data...');
      const tIndex0 = performance.now();
      const indexData = await loadIndexData();
      const tIndex1 = performance.now();

      setStatus('Initializing inference worker (text model)...');
      const tInit0 = performance.now();
      worker = new Worker('/workers/inference.worker.js');

      const initDone = new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error('Worker init timeout')), 180000);
        worker!.onmessage = (event: MessageEvent) => {
          const { type, id, payload } = event.data || {};
          if (type === 'INIT_COMPLETE' && id === 'text') {
            window.clearTimeout(timeout);
            resolve();
            return;
          }
          if (type === 'EMBED_RESULT' && id && pending.has(id)) {
            pending.get(id)!(new Float32Array(payload.embedding));
            pending.delete(id);
            return;
          }
          if (type === 'EMBED_ERROR') {
            reject(new Error(payload?.error || 'Worker embedding error'));
          }
        };
      });
      worker.postMessage({ type: 'INIT', payload: { dataBaseUrl } });
      await initDone;
      const tInit1 = performance.now();

      setStatus('Running fallback benchmark...');
      const fallbackQuery = 'pneumonia';
      const tFb0 = performance.now();
      const fallbackHit = checkFallback(fallbackQuery, indexData.fallbackResults) || [];
      const tFb1 = performance.now();

      setStatus('Running AI text benchmark...');
      const aiTextQuery = 'left basilar reticular opacity with mild volume loss';
      const textId = `text-${Date.now()}`;
      const textPromise = new Promise<Float32Array>((resolve) => pending.set(textId, resolve));
      const tText0 = performance.now();
      worker.postMessage({ type: 'EMBED_TEXT', id: textId, payload: { text: aiTextQuery } });
      const textEmbedding = await textPromise;
      const tText1 = performance.now();
      const tTextSearch0 = performance.now();
      const textTopK = findTopK(textEmbedding, indexData.embeddings, indexData.numImages, 512, 10);
      const tTextSearch1 = performance.now();

      setStatus('Running AI vision benchmark...');
      const visionImage = indexData.metadata[0]?.url;
      if (!visionImage) throw new Error('No image metadata available for vision benchmark');

      const tVisPrep0 = performance.now();
      const imgResp = await fetch(visionImage);
      if (!imgResp.ok) throw new Error(`Failed to fetch benchmark image: ${imgResp.statusText}`);
      const imgBlob = await imgResp.blob();
      const imgFile = new File([imgBlob], 'benchmark.webp', { type: imgBlob.type || 'image/webp' });
      const pixelValues = await preprocessImage(imgFile);
      const tVisPrep1 = performance.now();

      const visionId = `vision-${Date.now()}`;
      const visionPromise = new Promise<Float32Array>((resolve) => pending.set(visionId, resolve));
      const tVis0 = performance.now();
      worker.postMessage({ type: 'EMBED_IMAGE', id: visionId, payload: { pixelValues } });
      const visionEmbedding = await visionPromise;
      const tVis1 = performance.now();
      const tVisSearch0 = performance.now();
      const visionTopK = findTopK(visionEmbedding, indexData.embeddings, indexData.numImages, 512, 10);
      const tVisSearch1 = performance.now();

      const benchmarkReport: BenchReport = {
        run_at: new Date().toISOString(),
        dataset: {
          num_images: indexData.numImages,
          embedding_dim: indexData.embeddingDim,
          fallback_queries: Object.keys(indexData.fallbackResults).length,
        },
        timings_ms: {
          index_load: +(tIndex1 - tIndex0).toFixed(3),
          worker_init_text: +(tInit1 - tInit0).toFixed(3),
          fallback_lookup: +(tFb1 - tFb0).toFixed(3),
          ai_text_embedding: +(tText1 - tText0).toFixed(3),
          ai_text_vector_search: +(tTextSearch1 - tTextSearch0).toFixed(3),
          ai_text_total: +(tTextSearch1 - tText0).toFixed(3),
          ai_vision_preprocess: +(tVisPrep1 - tVisPrep0).toFixed(3),
          ai_vision_embedding: +(tVis1 - tVis0).toFixed(3),
          ai_vision_vector_search: +(tVisSearch1 - tVisSearch0).toFixed(3),
          ai_vision_total: +(tVisSearch1 - tVisPrep0).toFixed(3),
        },
        checks: {
          fallback_query: fallbackQuery,
          fallback_hit_count: fallbackHit.length,
          ai_text_query: aiTextQuery,
          ai_text_top1_score: textTopK.length ? +textTopK[0][1].toFixed(6) : null,
          ai_vision_image: indexData.metadata[0]?.filename || 'unknown',
          ai_vision_top1_score: visionTopK.length ? +visionTopK[0][1].toFixed(6) : null,
        },
      };

      setReport(benchmarkReport);
      setStatus('Benchmark complete.');
    } catch (e: any) {
      setError(e?.message || 'Benchmark failed');
      setStatus('Benchmark failed.');
    } finally {
      if (worker) worker.terminate();
      setIsRunning(false);
    }
  };

  return (
    <div className="container py-8">
      <h1 className="mb-2 text-3xl font-bold">Browser Benchmark</h1>
      <p className="mb-6 text-muted-foreground">
        Measures real browser-side performance for index load, worker initialization, fallback path, text inference, and vision inference.
      </p>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <button
          data-testid="run-benchmark"
          onClick={runBenchmark}
          disabled={isRunning}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          {isRunning ? 'Running...' : 'Run Benchmark'}
        </button>
        {report && (
          <>
            <button
              onClick={() => downloadText('browser_benchmark_report.json', JSON.stringify(report, null, 2))}
              className="rounded-lg border px-4 py-2 text-sm"
            >
              Download JSON
            </button>
            <button
              onClick={() =>
                downloadText(
                  'browser_benchmark_report.md',
                  `# Browser Benchmark Report\n\n\`\`\`json\n${JSON.stringify(report, null, 2)}\n\`\`\`\n`
                )
              }
              className="rounded-lg border px-4 py-2 text-sm"
            >
              Download Markdown
            </button>
          </>
        )}
      </div>

      <p className="mb-4 text-sm text-muted-foreground">Status: {status}</p>

      {error && (
        <div className="mb-4 rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {report && (
        <pre data-testid="benchmark-report" className="overflow-auto rounded-lg border bg-card p-4 text-xs">
          {JSON.stringify(report, null, 2)}
        </pre>
      )}
    </div>
  );
}
