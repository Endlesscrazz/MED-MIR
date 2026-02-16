'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { SearchBox } from '@/components/SearchBox';
import { ResultGrid } from '@/components/ResultGrid';
import { LoadingState } from '@/components/LoadingState';
import { useSearch } from '@/lib/hooks/useSearch';
import { QUERY_SUGGESTIONS } from '@/lib/utils';
import { Activity, Cpu, FlaskConical, ImageIcon, ShieldCheck } from 'lucide-react';
import Link from 'next/link';

type QuantBenchmark = {
  size_metrics?: {
    combined?: { reduction_pct?: number };
  };
  fidelity_metrics?: {
    text_encoder_cosine_fp32_vs_int8?: { mean?: number };
    vision_encoder_cosine_fp32_vs_int8?: { mean?: number };
  };
};

/**
 * Main search page for Med-MIR.
 */
export default function HomePage() {
  const {
    search,
    searchByImage, // Destructure the new function
    findSimilar,
    searchState,
    modelState,
    indexState,
    isReady,
  } = useSearch();

  const [selectedSuggestion, setSelectedSuggestion] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
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

  const textFidelity = useMemo(() => {
    const v = quantBench?.fidelity_metrics?.text_encoder_cosine_fp32_vs_int8?.mean;
    return typeof v === 'number' ? v.toFixed(3) : '0.994';
  }, [quantBench]);

  const visionFidelity = useMemo(() => {
    const v = quantBench?.fidelity_metrics?.vision_encoder_cosine_fp32_vs_int8?.mean;
    return typeof v === 'number' ? v.toFixed(3) : '0.998';
  }, [quantBench]);

  const noveltyStats = [
    {
      label: 'Model Size Reduction',
      value: combinedReduction,
      detail: '747.36 MB -> 260.30 MB (text + vision INT8)',
      icon: Cpu,
    },
    {
      label: 'Text Encoder Fidelity',
      value: textFidelity,
      detail: 'Cosine similarity (FP32 vs INT8 embeddings)',
      icon: Activity,
    },
    {
      label: 'Vision Encoder Fidelity',
      value: visionFidelity,
      detail: 'Cosine similarity (FP32 vs INT8 embeddings)',
      icon: ImageIcon,
    },
    {
      label: 'Retrieval Quality',
      value: 'R@10 91.1%',
      detail: 'Strict 91.1% | Semantic 100.0% (495-image baseline)',
      icon: ShieldCheck,
    },
  ] as const;

  // Handle text search submission
  const handleSearch = useCallback((query: string) => {
    if (query.trim()) {
      setSearchQuery(query.trim());
      search(query.trim());
    }
  }, [search]);

  // Handle image upload submission
  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSearchQuery(`Image: ${file.name}`);
      setSelectedSuggestion(null);
      searchByImage(file); // Trigger the vision encoder flow
    }
  }, [searchByImage]);

  // Trigger file browser
  const triggerUpload = () => {
    fileInputRef.current?.click();
  };
  
  // Handle clear search
  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
    setSelectedSuggestion(null);
    search('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [search]);

  // Handle suggestion click
  const handleSuggestionClick = useCallback((suggestion: string) => {
    setSelectedSuggestion(suggestion);
    handleSearch(suggestion);
  }, [handleSearch]);

  // Handle find similar
  const handleFindSimilar = useCallback((imageId: number) => {
    findSimilar(imageId);
  }, [findSimilar]);

  return (
    <div className="container py-8">
      {/* Hero Section */}
      <section className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight">
          Medical Image Retrieval
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          Search through medical images using natural language or by uploading an X-ray. 
          All AI inference runs locally in your browser — your data never leaves your device.
        </p>
      </section>

      {/* Novelty + Performance Snapshot */}
      <section className="mb-10">
        <div className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-sky-50 via-white to-emerald-50 p-6 shadow-sm">
          <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-sky-200/30 blur-2xl" />
          <div className="pointer-events-none absolute -bottom-20 -left-16 h-40 w-40 rounded-full bg-emerald-200/30 blur-2xl" />

          <div className="relative">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border bg-white/70 px-3 py-1 text-xs font-medium text-foreground backdrop-blur">
              <FlaskConical className="h-3.5 w-3.5 text-primary" />
              Engineered Novelty
            </div>
            <h2 className="mb-2 text-2xl font-semibold tracking-tight">
              Quantized BiomedCLIP Running Directly in the Browser
            </h2>
            <p className="mb-6 max-w-3xl text-sm text-muted-foreground">
              Med-MIR deploys a state-of-the-art biomedical model as quantized ONNX encoders (text + vision) for local inference.
              No GPU backend, no cloud inference API, and no patient-image upload to external inference servers.
            </p>

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
        </div>
      </section>

      {/* Loading State */}
      {!isReady && (
        <LoadingState
          modelState={modelState}
          indexState={indexState}
        />
      )}

      {/* Search Interface */}
      {isReady && (
        <>
          {/* Search Controls */}
          <section className="mb-8 flex flex-col items-center gap-4 sm:flex-row">
            <div className="w-full flex-1">
              <SearchBox
                onSearch={handleSearch}
                isLoading={searchState.isLoading}
                initialValue={searchQuery}
                placeholder="Describe pathology (e.g., 'pleural effusion')"
              />
            </div>
            
            <div className="flex shrink-0 gap-2">
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleImageUpload} 
                accept="image/*" 
                className="hidden" 
              />
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
          
          {/* Model Loading Progress (Visible when Vision model is lazy-loading) */}
          {modelState.isLoading && !modelState.isLoaded && (
            <section className="mb-8 animate-slide-in">
              <div className="rounded-lg border border-primary/50 bg-primary/10 p-4">
                <div className="flex items-center gap-3">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-primary/20">
                    <div 
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${modelState.progress}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-primary">
                    {Math.round(modelState.progress)}%
                  </span>
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Initializing Vision Encoder... First load takes ~30-60s.
                </p>
              </div>
            </section>
          )}

          {/* Query Suggestions (Only if no results) */}
          {searchState.results.length === 0 && !searchState.isLoading && (
            <section className="mb-8">
              <h2 className="mb-4 text-sm font-medium text-muted-foreground">
                Try these common queries:
              </h2>
              <div className="flex flex-wrap gap-2">
                {QUERY_SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className={`rounded-full border px-4 py-2 text-sm transition-colors hover:bg-accent hover:text-accent-foreground ${
                      selectedSuggestion === suggestion
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-background'
                    }`}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Search Status & Clear Button */}
          {searchState.query && (
            <section className="mb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>
                    Showing results for: <strong className="text-foreground truncate max-w-[200px] inline-block align-bottom">{searchState.query}</strong>
                  </span>
                  {searchState.searchType && (
                    <span className="rounded-full bg-muted px-2 py-0.5 text-xs">
                      {searchState.searchType === 'fallback' ? 'Cached' : 'AI Inference'}
                    </span>
                  )}
                  {searchState.inferenceTime !== null && (
                    <span className="text-xs">
                      {searchState.inferenceTime.toFixed(0)}ms
                    </span>
                  )}
                </div>
                <button
                  onClick={handleClearSearch}
                  className="text-sm font-medium text-primary hover:underline"
                >
                  Clear Results
                </button>
              </div>
            </section>
          )}

          {/* Error State */}
          {searchState.error && (
            <section className="mb-8">
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-center">
                <p className="text-sm text-destructive font-medium">{searchState.error}</p>
              </div>
            </section>
          )}

          {/* Results Grid */}
          <section>
            <ResultGrid
              results={searchState.results}
              isLoading={searchState.isLoading}
              onFindSimilar={handleFindSimilar}
            />
          </section>
        </>
      )}
    </div>
  );
}
