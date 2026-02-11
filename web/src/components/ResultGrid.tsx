'use client';

import { ResultCard, ResultCardSkeleton } from './ResultCard';
import type { SearchResult } from '@/lib/types';
import { ImageOff } from 'lucide-react';

interface ResultGridProps {
  /** Search results to display */
  results: SearchResult[];
  /** Whether results are loading */
  isLoading?: boolean;
  /** Callback when "Find Similar" is clicked */
  onFindSimilar?: (imageId: number) => void;
  /** Number of skeleton items to show when loading */
  skeletonCount?: number;
}

/**
 * Grid display of search results.
 * 
 * Features:
 * - Responsive grid layout
 * - Loading skeletons
 * - Empty state
 * - Ranked results
 */
export function ResultGrid({
  results,
  isLoading = false,
  onFindSimilar,
  skeletonCount = 8,
}: ResultGridProps) {
  // Loading state
  if (isLoading) {
    return (
      <div className="results-grid">
        {Array.from({ length: skeletonCount }).map((_, i) => (
          <ResultCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  // Empty state
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="mb-4 rounded-full bg-muted p-4">
          <ImageOff className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="mb-2 text-lg font-semibold">No results yet</h3>
        <p className="max-w-sm text-sm text-muted-foreground">
          Enter a search query to find relevant medical images. Try queries like
          "pneumonia", "cardiomegaly", or "normal chest xray".
        </p>
      </div>
    );
  }

  // Results grid
  return (
    <div className="results-grid">
      {results.map((result, index) => (
        <ResultCard
          key={result.image.id}
          result={result}
          rank={index + 1}
          onFindSimilar={onFindSimilar}
        />
      ))}
    </div>
  );
}
