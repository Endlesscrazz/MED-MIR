/**
 * Binary data loader utilities for Med-MIR.
 * 
 * Handles loading and parsing of binary embedding files
 * and JSON data files for the search index.
 */

import type { 
  ImageMetadata, 
  FallbackResults, 
  NearestNeighbors,
  IndexInfo,
  Metrics,
  HardCases,
} from './types';

/**
 * Configuration for data loading.
 */
export interface DataConfig {
  /** Base URL for data files (e.g., GitHub Pages URL) */
  baseUrl: string;
  /** Whether to use demo data (local development) */
  useDemo: boolean;
}

/**
 * Default configuration - uses local demo-data in development.
 */
const DEFAULT_CONFIG: DataConfig = {
  baseUrl: process.env.NEXT_PUBLIC_DATA_URL || '/demo-data',
  useDemo: process.env.NODE_ENV === 'development',
};

/**
 * Loaded index data structure.
 */
export interface IndexData {
  embeddings: Float32Array;
  metadata: ImageMetadata[];
  fallbackResults: FallbackResults;
  nearestNeighbors: NearestNeighbors;
  indexInfo: IndexInfo;
  embeddingDim: number;
  numImages: number;
}

/**
 * Load binary embeddings file.
 * 
 * @param url - URL to embeddings.bin
 * @param expectedSize - Expected number of embeddings (for validation)
 * @returns Float32Array of embeddings
 */
export async function loadEmbeddings(
  url: string,
  expectedSize?: number
): Promise<Float32Array> {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to load embeddings: ${response.statusText}`);
  }
  
  const buffer = await response.arrayBuffer();
  const embeddings = new Float32Array(buffer);
  
  // Validate size if expected size provided
  if (expectedSize !== undefined) {
    const embeddingDim = embeddings.length / expectedSize;
    if (!Number.isInteger(embeddingDim)) {
      console.warn(
        `Embedding dimensions don't divide evenly: ${embeddings.length} / ${expectedSize}`
      );
    }
  }
  
  return embeddings;
}

/**
 * Load JSON data file with type safety.
 * 
 * @param url - URL to JSON file
 * @returns Parsed JSON data
 */
export async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.statusText}`);
  }
  
  return response.json() as Promise<T>;
}

/**
 * Load all index data required for search.
 * 
 * @param config - Data configuration
 * @param onProgress - Progress callback (0-1)
 * @returns Complete index data
 */
export async function loadIndexData(
  config: DataConfig = DEFAULT_CONFIG,
  onProgress?: (progress: number) => void
): Promise<IndexData> {
  const baseUrl = config.baseUrl;
  const totalSteps = 5;
  let completedSteps = 0;
  
  const updateProgress = () => {
    completedSteps++;
    onProgress?.(completedSteps / totalSteps);
  };
  
  try {
    // Load index info first to get dimensions
    const indexInfo = await loadJson<IndexInfo>(`${baseUrl}/index_info.json`);
    updateProgress();
    
    // Load metadata
    const metadata = await loadJson<ImageMetadata[]>(`${baseUrl}/metadata.json`);
    updateProgress();
    
    // Load embeddings with size validation
    const embeddings = await loadEmbeddings(
      `${baseUrl}/embeddings.bin`,
      metadata.length
    );
    updateProgress();
    
    // Load fallback results
    const fallbackResults = await loadJson<FallbackResults>(
      `${baseUrl}/fallback_results.json`
    );
    updateProgress();
    
    // Load nearest neighbors
    const nearestNeighbors = await loadJson<NearestNeighbors>(
      `${baseUrl}/nearest_neighbors.json`
    );
    updateProgress();
    
    // Calculate embedding dimension
    const embeddingDim = embeddings.length / metadata.length;
    
    // Fix image URLs to be absolute paths from public directory
    const metadataWithFixedUrls = metadata.map((img) => ({
      ...img,
      url: img.url.startsWith('/') ? img.url : `${baseUrl}/${img.url}`,
    }));
    
    return {
      embeddings,
      metadata: metadataWithFixedUrls,
      fallbackResults,
      nearestNeighbors,
      indexInfo,
      embeddingDim,
      numImages: metadata.length,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Failed to load index data: ${message}`);
  }
}

/**
 * Load metrics data for the dashboard.
 * 
 * @param config - Data configuration
 * @returns Metrics data
 */
export async function loadMetrics(
  config: DataConfig = DEFAULT_CONFIG
): Promise<Metrics> {
  return loadJson<Metrics>(`${config.baseUrl}/metrics.json`);
}

/**
 * Load hard cases data for analysis page.
 * 
 * @param config - Data configuration
 * @returns Hard cases data
 */
export async function loadHardCases(
  config: DataConfig = DEFAULT_CONFIG
): Promise<HardCases> {
  return loadJson<HardCases>(`${config.baseUrl}/hard_cases.json`);
}

/**
 * Get a single embedding vector from the embeddings array.
 * 
 * @param embeddings - Full embeddings array
 * @param index - Image index
 * @param dim - Embedding dimension
 * @returns Single embedding vector
 */
export function getEmbedding(
  embeddings: Float32Array,
  index: number,
  dim: number
): Float32Array {
  const start = index * dim;
  return embeddings.slice(start, start + dim);
}

/**
 * Compute dot product between two vectors.
 * This equals cosine similarity for L2-normalized vectors.
 * 
 * @param a - First vector
 * @param b - Second vector
 * @returns Dot product (similarity score)
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }
  
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  
  return sum;
}

/**
 * Find top-K most similar images given a query embedding.
 * 
 * @param queryEmbedding - Query vector
 * @param imageEmbeddings - All image embeddings
 * @param numImages - Number of images
 * @param dim - Embedding dimension
 * @param k - Number of results to return
 * @returns Array of [index, score] tuples, sorted by score descending
 */
export function findTopK(
  queryEmbedding: Float32Array,
  imageEmbeddings: Float32Array,
  numImages: number,
  dim: number,
  k: number
): Array<[number, number]> {
  const scores: Array<[number, number]> = [];
  
  for (let i = 0; i < numImages; i++) {
    const imageEmbedding = getEmbedding(imageEmbeddings, i, dim);
    const score = dotProduct(queryEmbedding, imageEmbedding);
    scores.push([i, score]);
  }
  
  // Sort by score descending
  scores.sort((a, b) => b[1] - a[1]);
  
  return scores.slice(0, k);
}

/**
 * L2 normalize a vector in place.
 * 
 * @param vector - Vector to normalize
 * @returns The normalized vector (same reference)
 */
export function l2Normalize(vector: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vector.length; i++) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }
  
  return vector;
}
