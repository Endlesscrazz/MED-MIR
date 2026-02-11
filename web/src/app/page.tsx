'use client';

import { useState, useEffect, useCallback } from 'react';
import { SearchBox } from '@/components/SearchBox';
import { ResultGrid } from '@/components/ResultGrid';
import { LoadingState } from '@/components/LoadingState';
import { useSearch } from '@/lib/hooks/useSearch';
import { QUERY_SUGGESTIONS } from '@/lib/utils';

/**
 * Main search page for Med-MIR.
 * 
 * Provides the primary interface for:
 * - Text-to-image search
 * - Viewing search results
 * - Finding similar images
 */
export default function HomePage() {
  const {
    search,
    findSimilar,
    searchState,
    modelState,
    indexState,
    isReady,
  } = useSearch();

  const [selectedSuggestion, setSelectedSuggestion] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Handle search submission
  const handleSearch = useCallback((query: string) => {
    if (query.trim()) {
      setSearchQuery(query.trim());
      search(query.trim());
    }
  }, [search]);
  
  // Handle clear search
  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
    setSelectedSuggestion(null);
    search('');
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
          Search through medical images using natural language queries. 
          All AI inference runs locally in your browser — your data never leaves your device.
        </p>
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
          {/* Search Box */}
          <section className="mb-8">
            <SearchBox
              onSearch={handleSearch}
              isLoading={searchState.isLoading}
              initialValue={searchQuery}
              placeholder="Describe what you're looking for (e.g., 'pleural effusion', 'normal chest xray')"
            />
          </section>
          
          {/* Model Loading Progress (when model is loading) */}
          {modelState.isLoading && !modelState.isLoaded && (
            <section className="mb-8">
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
                  Loading AI model... This may take 2-3 minutes on first load. The model will be cached for future use.
                </p>
              </div>
            </section>
          )}

          {/* Query Suggestions */}
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

          {/* Search Status */}
          {searchState.query && (
            <section className="mb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>
                    Showing results for: <strong className="text-foreground">{searchState.query}</strong>
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
                  className="rounded-lg border px-4 py-2 text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  Clear Search
                </button>
              </div>
            </section>
          )}

          {/* Error State */}
          {searchState.error && (
            <section className="mb-8">
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">{searchState.error}</p>
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
