/**
 * Search logic for Med-MIR.
 * 
 * Implements the hybrid search strategy:
 * 1. Check fallback results for common queries (instant)
 * 2. Fall back to ONNX inference for novel queries
 */

import type { 
  SearchResult, 
  FallbackResults, 
  ImageMetadata,
} from './types';
import { findTopK } from './binary-loader';

/**
 * Configuration for search operations.
 */
export interface SearchConfig {
  /** Number of results to return */
  topK: number;
  /** Minimum similarity threshold (0-1) */
  minScore?: number;
}

const DEFAULT_CONFIG: SearchConfig = {
  topK: 10,
  minScore: 0,
};

/**
 * Check if query matches a fallback result.
 * 
 * @param query - Search query
 * @param fallbackResults - Pre-computed fallback results
 * @returns Matching results or null
 */
export function checkFallback(
  query: string,
  fallbackResults: FallbackResults
): Array<{ id: number; score: number }> | null {
  // Normalize query
  const normalized = query.toLowerCase().trim();
  
  // Direct match
  if (fallbackResults[normalized]) {
    return fallbackResults[normalized];
  }
  
  // Try partial matching for common variations
  const variations = [
    normalized,
    normalized.replace(/\s+/g, ' '),
    normalized.replace(/xray/g, 'x-ray'),
    normalized.replace(/x-ray/g, 'xray'),
  ];
  
  for (const variation of variations) {
    if (fallbackResults[variation]) {
      return fallbackResults[variation];
    }
  }
  
  return null;
}

/**
 * Perform vector similarity search.
 * 
 * @param queryEmbedding - Query vector
 * @param imageEmbeddings - All image embeddings
 * @param numImages - Number of images
 * @param embeddingDim - Embedding dimension
 * @param config - Search configuration
 * @returns Top-K results
 */
export function vectorSearch(
  queryEmbedding: Float32Array,
  imageEmbeddings: Float32Array,
  numImages: number,
  embeddingDim: number,
  config: SearchConfig = DEFAULT_CONFIG
): Array<{ id: number; score: number }> {
  const results = findTopK(
    queryEmbedding,
    imageEmbeddings,
    numImages,
    embeddingDim,
    config.topK
  );
  
  // Filter by minimum score if specified
  const filtered = config.minScore
    ? results.filter(([_, score]) => score >= config.minScore!)
    : results;
  
  return filtered.map(([id, score]) => ({ id, score }));
}

/**
 * Convert raw results to SearchResult array with metadata.
 * 
 * @param results - Array of { id, score }
 * @param metadata - Image metadata array
 * @returns SearchResult array
 */
export function toSearchResults(
  results: Array<{ id: number; score: number }>,
  metadata: ImageMetadata[]
): SearchResult[] {
  return results.map(({ id, score }) => ({
    image: metadata[id],
    score,
  }));
}

/**
 * Group search results by label.
 * 
 * @param results - Search results
 * @returns Map of label to results
 */
export function groupByLabel(
  results: SearchResult[]
): Map<string, SearchResult[]> {
  const groups = new Map<string, SearchResult[]>();
  
  for (const result of results) {
    const label = result.image.ground_truth_label;
    const existing = groups.get(label) || [];
    existing.push(result);
    groups.set(label, existing);
  }
  
  return groups;
}

/**
 * Calculate diversity score for results.
 * Higher score means more diverse labels in results.
 * 
 * @param results - Search results
 * @returns Diversity score (0-1)
 */
export function calculateDiversity(results: SearchResult[]): number {
  if (results.length === 0) return 0;
  
  const uniqueLabels = new Set(results.map(r => r.image.ground_truth_label));
  return uniqueLabels.size / results.length;
}

/**
 * Get statistics about search results.
 * 
 * @param results - Search results
 * @returns Statistics object
 */
export function getResultStats(results: SearchResult[]): {
  count: number;
  avgScore: number;
  minScore: number;
  maxScore: number;
  diversity: number;
  labelCounts: Record<string, number>;
} {
  if (results.length === 0) {
    return {
      count: 0,
      avgScore: 0,
      minScore: 0,
      maxScore: 0,
      diversity: 0,
      labelCounts: {},
    };
  }
  
  const scores = results.map(r => r.score);
  const labelCounts: Record<string, number> = {};
  
  for (const result of results) {
    const label = result.image.ground_truth_label;
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  }
  
  return {
    count: results.length,
    avgScore: scores.reduce((a, b) => a + b, 0) / scores.length,
    minScore: Math.min(...scores),
    maxScore: Math.max(...scores),
    diversity: calculateDiversity(results),
    labelCounts,
  };
}
