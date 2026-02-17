'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { SearchBox } from '@/components/SearchBox';
import { ResultGrid } from '@/components/ResultGrid';
import { LoadingState } from '@/components/LoadingState';
import { useSearch } from '@/lib/hooks/useSearch';
import { QUERY_SUGGESTIONS } from '@/lib/utils';
import {
  Activity,
  BookOpen,
  Brain,
  ChevronDown,
  ChevronUp,
  Cpu,
  FlaskConical,
  Gauge,
  ImageIcon,
  Lock,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import Link from 'next/link';

type QuantBenchmark = {
  size_metrics?: {
    text_encoder?: { fp32_mb?: number; int8_mb?: number; reduction_pct?: number };
    vision_encoder?: { fp32_mb?: number; int8_mb?: number; reduction_pct?: number };
    combined?: { fp32_mb?: number; int8_mb?: number; reduction_pct?: number };
  };
  fidelity_metrics?: {
    text_encoder_cosine_fp32_vs_int8?: { mean?: number };
    vision_encoder_cosine_fp32_vs_int8?: { mean?: number };
  };
  latency_ms?: {
    text_fp32?: { mean?: number };
    text_int8?: { mean?: number };
    vision_fp32?: { mean?: number };
    vision_int8?: { mean?: number };
  };
};

/**
 * Main search page for Med-MIR.
 */
export default function HomePage() {
  const {
    search,
    searchByImage,
    findSimilar,
    searchState,
    modelState,
    indexState,
    isReady,
  } = useSearch();

  const [selectedSuggestion, setSelectedSuggestion] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [viewMode, setViewMode] = useState<'general' | 'technical'>('general');
  const [showIntroModal, setShowIntroModal] = useState(false);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchSectionRef = useRef<HTMLElement>(null);
  const [quantBench, setQuantBench] = useState<QuantBenchmark | null>(null);
  const dataBaseUrl = process.env.NEXT_PUBLIC_DATA_URL || '/demo-data';

  useEffect(() => {
    let cancelled = false;
    async function loadBenchmark() {
      try {
        const response = await fetch(`${dataBaseUrl}/quantization_benchmark.json`, { cache: 'no-store' });
        if (!response.ok) return;
        const data = (await response.json()) as QuantBenchmark;
        if (!cancelled) setQuantBench(data);
      } catch {
        // Keep fallback constants if benchmark file is unavailable.
      }
    }
    loadBenchmark();
    return () => {
      cancelled = true;
    };
  }, [dataBaseUrl]);

  const combinedReduction = useMemo(() => {
    const v = quantBench?.size_metrics?.combined?.reduction_pct;
    return typeof v === 'number' ? `${v.toFixed(2)}%` : '65.17%';
  }, [quantBench]);

  const combinedSizes = useMemo(() => {
    const fp32 = quantBench?.size_metrics?.combined?.fp32_mb;
    const int8 = quantBench?.size_metrics?.combined?.int8_mb;
    if (typeof fp32 === 'number' && typeof int8 === 'number') {
      return `${fp32.toFixed(2)} MB -> ${int8.toFixed(2)} MB`;
    }
    return '747.36 MB -> 260.30 MB';
  }, [quantBench]);

  const textFidelity = useMemo(() => {
    const v = quantBench?.fidelity_metrics?.text_encoder_cosine_fp32_vs_int8?.mean;
    return typeof v === 'number' ? v.toFixed(3) : '0.994';
  }, [quantBench]);

  const visionFidelity = useMemo(() => {
    const v = quantBench?.fidelity_metrics?.vision_encoder_cosine_fp32_vs_int8?.mean;
    return typeof v === 'number' ? v.toFixed(3) : '0.998';
  }, [quantBench]);

  const textSpeedup = useMemo(() => {
    const fp32 = quantBench?.latency_ms?.text_fp32?.mean;
    const int8 = quantBench?.latency_ms?.text_int8?.mean;
    if (typeof fp32 === 'number' && typeof int8 === 'number' && int8 > 0) {
      return `${(fp32 / int8).toFixed(2)}x faster`;
    }
    return '2.62x faster';
  }, [quantBench]);

  const visionSpeedup = useMemo(() => {
    const fp32 = quantBench?.latency_ms?.vision_fp32?.mean;
    const int8 = quantBench?.latency_ms?.vision_int8?.mean;
    if (typeof fp32 === 'number' && typeof int8 === 'number' && int8 > 0) {
      return `${(fp32 / int8).toFixed(2)}x faster`;
    }
    return '2.67x faster';
  }, [quantBench]);

  const noveltyStats = useMemo(
    () =>
      viewMode === 'general'
        ? [
            {
              label: 'Model compressed',
              value: combinedReduction,
              detail: combinedSizes,
              icon: Cpu,
            },
            {
              label: 'Text quality retained',
              value: textFidelity,
              detail: 'Closeness between original and quantized text embeddings',
              icon: Activity,
            },
            {
              label: 'Image quality retained',
              value: visionFidelity,
              detail: 'Closeness between original and quantized image embeddings',
              icon: ImageIcon,
            },
            {
              label: 'Top-10 retrieval success',
              value: '91.1% strict',
              detail: 'Semantic success: 100.0% on 495-image baseline',
              icon: ShieldCheck,
            },
          ]
        : [
            {
              label: 'Model Size Reduction',
              value: combinedReduction,
              detail: `${combinedSizes} (text + vision INT8)`,
              icon: Cpu,
            },
            {
              label: 'Text Encoder Cosine',
              value: textFidelity,
              detail: 'Mean cosine similarity (FP32 vs INT8)',
              icon: Activity,
            },
            {
              label: 'Vision Encoder Cosine',
              value: visionFidelity,
              detail: 'Mean cosine similarity (FP32 vs INT8)',
              icon: ImageIcon,
            },
            {
              label: 'Retrieval Recall@10',
              value: '91.1%',
              detail: 'Strict: 91.1% | Semantic: 100.0% (495 images)',
              icon: ShieldCheck,
            },
          ],
    [combinedReduction, combinedSizes, textFidelity, viewMode, visionFidelity]
  );

  const handleSearch = useCallback(
    (query: string) => {
      if (query.trim()) {
        setSearchQuery(query.trim());
        search(query.trim());
      }
    },
    [search]
  );

  const handleImageUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setSearchQuery(`Image: ${file.name}`);
        setSelectedSuggestion(null);
        searchByImage(file);
      }
    },
    [searchByImage]
  );

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
    setSelectedSuggestion(null);
    search('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [search]);

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      setSelectedSuggestion(suggestion);
      handleSearch(suggestion);
    },
    [handleSearch]
  );

  const handleFindSimilar = useCallback(
    (imageId: number) => {
      findSimilar(imageId);
    },
    [findSimilar]
  );

  const handleHeroSampleQuery = useCallback(() => {
    handleSuggestionClick('pneumonia');
    searchSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [handleSuggestionClick]);

  return (
    <div className="container py-8">
      {showIntroModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4">
          <div className="w-full max-w-2xl rounded-2xl border bg-card p-6 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-4">
              <h2 className="text-xl font-semibold">Med-MIR in 30 seconds</h2>
              <button
                onClick={() => setShowIntroModal(false)}
                className="text-sm font-medium text-muted-foreground hover:text-foreground"
              >
                Close
              </button>
            </div>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                Med-MIR helps you retrieve similar chest X-ray cases by text (symptoms/findings) or by image upload.
              </p>
              <p>
                The core novelty is local inference: quantized BiomedCLIP text and vision encoders run in your browser, so
                user uploads are not sent to a cloud inference API.
              </p>
              <p>
                This is an educational and research retrieval tool. It is not a clinical diagnostic system and should not be
                used for medical decision making.
              </p>
            </div>
          </div>
        </div>
      )}

      <section className="mb-10 overflow-hidden rounded-3xl border bg-gradient-to-br from-sky-50 via-white to-emerald-50 p-8 shadow-sm">
        <div className="mx-auto max-w-4xl text-center">
          <h2 className="sr-only">Medical Image Retrieval</h2>
          <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Find similar chest X-ray cases in seconds
          </h1>
          <p className="mx-auto mb-6 max-w-2xl text-lg text-muted-foreground">
            Search by symptoms or upload an X-ray. Results are generated locally in your browser to keep your data private.
          </p>
          <div className="mb-6 flex flex-wrap items-center justify-center gap-3">
            <button
              onClick={handleHeroSampleQuery}
              className="inline-flex h-11 items-center gap-2 rounded-xl bg-primary px-5 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <BookOpen className="h-4 w-4" />
              Try a sample query
            </button>
            <button
              onClick={triggerUpload}
              disabled={searchState.isLoading}
              className="inline-flex h-11 items-center gap-2 rounded-xl border bg-card px-5 font-medium transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
            >
              <Upload className="h-4 w-4" />
              Upload an X-ray
            </button>
            <button
              onClick={() => setShowIntroModal(true)}
              className="inline-flex h-11 items-center gap-2 rounded-xl border bg-background px-5 font-medium transition-colors hover:bg-muted"
            >
              <FlaskConical className="h-4 w-4" />
              Learn in 30 sec
            </button>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
            <span className="rounded-full border bg-white/80 px-3 py-1">Private by design</span>
            <span className="rounded-full border bg-white/80 px-3 py-1">No cloud inference</span>
            <span className="rounded-full border bg-white/80 px-3 py-1">Runs in browser</span>
          </div>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="mb-4 text-2xl font-semibold tracking-tight">What this tool does</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <article className="rounded-2xl border bg-card p-5">
            <h3 className="mb-2 font-semibold">1) Type a query</h3>
            <p className="text-sm text-muted-foreground">Example: pleural effusion, cardiomegaly, or pneumonia pattern.</p>
          </article>
          <article className="rounded-2xl border bg-card p-5">
            <h3 className="mb-2 font-semibold">2) Or upload an image</h3>
            <p className="text-sm text-muted-foreground">Use a chest X-ray to retrieve visually similar study examples.</p>
          </article>
          <article className="rounded-2xl border bg-card p-5">
            <h3 className="mb-2 font-semibold">3) Get ranked matches</h3>
            <p className="text-sm text-muted-foreground">View similar cases with similarity scores and quick comparisons.</p>
          </article>
        </div>
        <p className="mt-3 text-sm text-muted-foreground">
          Med-MIR is a retrieval system for education and research. It is not a diagnosis tool.
        </p>
      </section>

      <section className="mb-10 grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="mb-3 text-2xl font-semibold tracking-tight">Why this is different</h2>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <Lock className="mt-0.5 h-4 w-4 text-primary" />
              Most demos send uploads to cloud APIs. Med-MIR keeps inference local in-browser.
            </li>
            <li className="flex items-start gap-2">
              <Brain className="mt-0.5 h-4 w-4 text-primary" />
              We quantized a SOTA biomedical model (BiomedCLIP) to INT8 ONNX for practical local runtime.
            </li>
            <li className="flex items-start gap-2">
              <Gauge className="mt-0.5 h-4 w-4 text-primary" />
              Quality stays high while model footprint and latency drop significantly.
            </li>
          </ul>
        </div>
        <div className="rounded-2xl border bg-gradient-to-br from-cyan-50 to-blue-50 p-6">
          <h3 className="mb-3 text-lg font-semibold">Snapshot</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border bg-white/90 p-4">
              <p className="text-xs text-muted-foreground">Model compression</p>
              <p className="text-xl font-semibold">{combinedReduction}</p>
            </div>
            <div className="rounded-xl border bg-white/90 p-4">
              <p className="text-xs text-muted-foreground">Retrieval quality</p>
              <p className="text-xl font-semibold">R@10 91.1%</p>
            </div>
            <div className="rounded-xl border bg-white/90 p-4">
              <p className="text-xs text-muted-foreground">Text speedup</p>
              <p className="text-xl font-semibold">{textSpeedup}</p>
            </div>
            <div className="rounded-xl border bg-white/90 p-4">
              <p className="text-xs text-muted-foreground">Vision speedup</p>
              <p className="text-xl font-semibold">{visionSpeedup}</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-10">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-2xl font-semibold tracking-tight">Metrics and verifiability</h2>
          <div className="inline-flex rounded-xl border bg-muted p-1">
            <button
              onClick={() => setViewMode('general')}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                viewMode === 'general' ? 'bg-background shadow-sm' : 'text-muted-foreground'
              }`}
            >
              Simple view
            </button>
            <button
              onClick={() => setViewMode('technical')}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                viewMode === 'technical' ? 'bg-background shadow-sm' : 'text-muted-foreground'
              }`}
            >
              Technical view
            </button>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-sky-50 via-white to-emerald-50 p-6 shadow-sm">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border bg-white/70 px-3 py-1 text-xs font-medium text-foreground backdrop-blur">
            <FlaskConical className="h-3.5 w-3.5 text-primary" />
            Quantized BiomedCLIP running in browser
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {noveltyStats.map(({ label, value, detail, icon: Icon }) => (
              <div key={label} className="rounded-xl border bg-white/80 p-4 backdrop-blur">
                <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
                  <Icon className="h-4 w-4 text-primary" />
                  {label}
                </div>
                <div className="mb-1 text-lg font-semibold">{value}</div>
                <div className="text-xs text-muted-foreground">{detail}</div>
              </div>
            ))}
          </div>

          <button
            onClick={() => setShowAdvancedMetrics((v) => !v)}
            className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
          >
            {showAdvancedMetrics ? 'Hide technical details' : 'Show technical details'}
            {showAdvancedMetrics ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>

          {showAdvancedMetrics && (
            <div className="mt-4 rounded-xl border bg-white/80 p-4 text-sm text-muted-foreground">
              <p className="mb-2">
                Quantization setup: BiomedCLIP text and vision encoders exported to ONNX and dynamically quantized to INT8.
              </p>
              <ul className="space-y-1">
                <li>Combined size: {combinedSizes} (FP32 to INT8)</li>
                <li>Cosine fidelity: text {textFidelity}, vision {visionFidelity}</li>
                <li>Latency speedup: text {textSpeedup}, vision {visionSpeedup}</li>
                <li>Retrieval baseline: strict Recall@10 = 91.1%, semantic Recall@10 = 100.0% (495 images)</li>
              </ul>
            </div>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>Latest benchmark: 2026-02-10 run</span>
            <Link href="/metrics" className="font-medium text-primary hover:underline">
              View Metrics Dashboard
            </Link>
            <Link href="/benchmark" className="font-medium text-primary hover:underline">
              Run Browser Benchmark
            </Link>
            <a
              href={`${dataBaseUrl}/quantization_benchmark.json`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary hover:underline"
            >
              View Source JSON
            </a>
          </div>
        </div>
      </section>

      {!isReady && <LoadingState modelState={modelState} indexState={indexState} />}

      {isReady && (
        <>
          <section ref={searchSectionRef} className="mb-8 flex flex-col items-center gap-4 sm:flex-row">
            <div className="w-full flex-1">
              <SearchBox
                onSearch={handleSearch}
                isLoading={searchState.isLoading}
                initialValue={searchQuery}
                placeholder="Describe pathology (e.g., 'pleural effusion')"
              />
            </div>

            <div className="flex shrink-0 gap-2">
              <input type="file" ref={fileInputRef} onChange={handleImageUpload} accept="image/*" className="hidden" />
              <button
                onClick={triggerUpload}
                disabled={searchState.isLoading}
                className="flex h-14 items-center gap-2 rounded-xl border bg-card px-6 font-medium shadow-sm transition-all hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
              >
                <ImageIcon className="h-5 w-5 text-primary" />
                <span className="hidden md:inline">Search by Image</span>
              </button>
            </div>
          </section>

          {modelState.isLoading && !modelState.isLoaded && (
            <section className="mb-8 animate-slide-in">
              <div className="rounded-lg border border-primary/50 bg-primary/10 p-4">
                <div className="flex items-center gap-3">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-primary/20">
                    <div className="h-full bg-primary transition-all duration-300" style={{ width: `${modelState.progress}%` }} />
                  </div>
                  <span className="text-sm font-medium text-primary">{Math.round(modelState.progress)}%</span>
                </div>
                <p className="mt-2 text-sm text-muted-foreground">Initializing Vision Encoder... First load takes ~30-60s.</p>
              </div>
            </section>
          )}

          {searchState.results.length === 0 && !searchState.isLoading && (
            <section className="mb-8">
              <h2 className="mb-4 text-sm font-medium text-muted-foreground">Try these common queries:</h2>
              <div className="flex flex-wrap gap-2">
                {QUERY_SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className={`rounded-full border px-4 py-2 text-sm transition-colors hover:bg-accent hover:text-accent-foreground ${
                      selectedSuggestion === suggestion ? 'bg-primary text-primary-foreground' : 'bg-background'
                    }`}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </section>
          )}

          {searchState.query && (
            <section className="mb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>
                    Showing results for:{' '}
                    <strong className="inline-block max-w-[200px] truncate align-bottom text-foreground">{searchState.query}</strong>
                  </span>
                  {searchState.searchType && (
                    <span className="rounded-full bg-muted px-2 py-0.5 text-xs">
                      {searchState.searchType === 'fallback' ? 'Cached' : 'AI Inference'}
                    </span>
                  )}
                  {searchState.inferenceTime !== null && <span className="text-xs">{searchState.inferenceTime.toFixed(0)}ms</span>}
                </div>
                <button onClick={handleClearSearch} className="text-sm font-medium text-primary hover:underline">
                  Clear Results
                </button>
              </div>
            </section>
          )}

          {searchState.error && (
            <section className="mb-8">
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-center">
                <p className="text-sm font-medium text-destructive">{searchState.error}</p>
              </div>
            </section>
          )}

          <section>
            <ResultGrid results={searchState.results} isLoading={searchState.isLoading} onFindSimilar={handleFindSimilar} />
          </section>
        </>
      )}
    </div>
  );
}
