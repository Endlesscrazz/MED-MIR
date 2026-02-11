'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { 
  SearchResult, 
  SearchState, 
  ModelState,
  ImageMetadata,
  FallbackResults,
  NearestNeighbors,
} from '@/lib/types';
import { loadIndexData, findTopK, type IndexData } from '@/lib/binary-loader';
import { generateId } from '@/lib/utils';

/**
 * Index loading state.
 */
interface IndexState {
  isLoaded: boolean;
  isLoading: boolean;
  progress: number;
  error: string | null;
}

/**
 * Initial search state.
 */
const INITIAL_SEARCH_STATE: SearchState = {
  query: '',
  results: [],
  isLoading: false,
  error: null,
  searchType: null,
  inferenceTime: null,
};

/**
 * Initial model state.
 */
const INITIAL_MODEL_STATE: ModelState = {
  isLoaded: false,
  isLoading: false,
  progress: 0,
  error: null,
};

/**
 * Initial index state.
 */
const INITIAL_INDEX_STATE: IndexState = {
  isLoaded: false,
  isLoading: false,
  progress: 0,
  error: null,
};

/**
 * Custom hook for search functionality.
 * 
 * Handles:
 * - Index data loading
 * - Model initialization (via Web Worker)
 * - Text-to-image search with fallback strategy
 * - Image-to-image similarity search
 */
export function useSearch() {
  // State
  const [searchState, setSearchState] = useState<SearchState>(INITIAL_SEARCH_STATE);
  const [modelState, setModelState] = useState<ModelState>(INITIAL_MODEL_STATE);
  const [indexState, setIndexState] = useState<IndexState>(INITIAL_INDEX_STATE);
  
  // Refs
  const indexDataRef = useRef<IndexData | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const pendingRequestsRef = useRef<Map<string, (embedding: Float32Array) => void>>(new Map());

  /**
   * Initialize the Web Worker for ONNX inference.
   */
  const initWorker = useCallback(() => {
    if (typeof window === 'undefined' || workerRef.current) return;

    setModelState((prev) => ({ ...prev, isLoading: true, progress: 0 }));

    try {
      // Create worker from public/ directory — served statically, no webpack.
      // The worker loads onnxruntime-web from CDN and runs BiomedCLIP directly.
      workerRef.current = new Worker('/workers/inference.worker.js');

      // Handle messages from worker
      workerRef.current.onmessage = (event) => {
        const { type, id, payload } = event.data;

        switch (type) {
          case 'INIT_COMPLETE':
            setModelState({
              isLoaded: true,
              isLoading: false,
              progress: 100,
              error: null,
            });
            break;

          case 'INIT_ERROR':
            setModelState({
              isLoaded: false,
              isLoading: false,
              progress: 0,
              error: payload?.error || 'Failed to initialize model',
            });
            break;

          case 'PROGRESS':
            setModelState((prev) => ({
              ...prev,
              progress: (payload?.progress || 0) * 100,
            }));
            break;

          case 'EMBED_RESULT':
            if (id && payload?.embedding) {
              const callback = pendingRequestsRef.current.get(id);
              if (callback) {
                // Convert array back to Float32Array if needed
                const embedding = payload.embedding instanceof Float32Array
                  ? payload.embedding
                  : new Float32Array(payload.embedding);
                callback(embedding);
                pendingRequestsRef.current.delete(id);
              }
            }
            break;

          case 'EMBED_ERROR':
            if (id) {
              pendingRequestsRef.current.delete(id);
            }
            setSearchState((prev) => ({
              ...prev,
              isLoading: false,
              error: payload?.error || 'Inference failed',
            }));
            break;
        }
      };

      // Handle worker errors
      workerRef.current.onerror = (error) => {
        console.error('Worker error:', error);
        setModelState({
          isLoaded: false,
          isLoading: false,
          progress: 0,
          error: 'Worker initialization failed',
        });
      };

      // Initialize the worker
      workerRef.current.postMessage({ type: 'INIT' });
    } catch (error) {
      console.error('Failed to create worker:', error);
      setModelState({
        isLoaded: false,
        isLoading: false,
        progress: 0,
        error: 'Failed to create inference worker',
      });
    }
  }, []);

  /**
   * Load index data on mount.
   */
  useEffect(() => {
    async function loadIndex() {
      setIndexState({ isLoaded: false, isLoading: true, progress: 0, error: null });

      try {
        const data = await loadIndexData(undefined, (progress) => {
          setIndexState((prev) => ({ ...prev, progress }));
        });

        indexDataRef.current = data;
        setIndexState({ isLoaded: true, isLoading: false, progress: 1, error: null });

        // Initialize worker after index is loaded
        initWorker();
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load index';
        setIndexState({ isLoaded: false, isLoading: false, progress: 0, error: message });
      }
    }

    loadIndex();

    // Cleanup
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [initWorker]);

  /**
   * Get embedding from worker.
   */
  const getEmbedding = useCallback((text: string): Promise<Float32Array> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = generateId();
      pendingRequestsRef.current.set(id, resolve);

      // Timeout after 120 seconds (first WASM inference on a large model is slow)
      setTimeout(() => {
        if (pendingRequestsRef.current.has(id)) {
          pendingRequestsRef.current.delete(id);
          reject(new Error('Inference timed out. The model may still be warming up — please try again.'));
        }
      }, 120000);

      workerRef.current.postMessage({
        type: 'EMBED_TEXT',
        id,
        payload: { text },
      });
    });
  }, []);

  /**
   * Convert results to SearchResult array.
   */
  const toSearchResults = useCallback(
    (results: Array<{ id: number; score: number }>): SearchResult[] => {
      const indexData = indexDataRef.current;
      if (!indexData) return [];

      return results.map(({ id, score }) => ({
        image: indexData.metadata[id],
        score,
      }));
    },
    []
  );

  /**
   * Search with fallback-first strategy.
   */
  const search = useCallback(
    async (query: string) => {
      const indexData = indexDataRef.current;
      if (!indexData) {
        setSearchState((prev) => ({
          ...prev,
          error: 'Index not loaded',
        }));
        return;
      }

      // Handle clear search
      if (!query.trim()) {
        setSearchState({
          query: '',
          results: [],
          isLoading: false,
          error: null,
          searchType: null,
          inferenceTime: null,
        });
        return;
      }

      setSearchState({
        query,
        results: [],
        isLoading: true,
        error: null,
        searchType: null,
        inferenceTime: null,
      });

      const startTime = performance.now();

      // Check fallback results first (exact match, then fuzzy match)
      const normalizedQuery = query.toLowerCase().trim();
      let fallbackMatch = indexData.fallbackResults[normalizedQuery];

      // Fuzzy fallback: if no exact match, check if all query words
      // appear in any pre-computed fallback key. This handles cases like
      // "chest xray" matching "normal chest xray", or "lung" matching
      // "lung nodule", etc.
      if (!fallbackMatch) {
        const queryWords = normalizedQuery.split(/\s+/);
        const fallbackKeys = Object.keys(indexData.fallbackResults);

        for (const key of fallbackKeys) {
          const keyWords = key.split(/\s+/);
          const allQueryWordsMatch = queryWords.every((qw) =>
            keyWords.some((kw) => kw === qw || kw.startsWith(qw) || qw.startsWith(kw))
          );
          if (allQueryWordsMatch) {
            fallbackMatch = indexData.fallbackResults[key];
            break;
          }
        }
      }

      if (fallbackMatch) {
        // Fast path: use pre-computed results
        const results = toSearchResults(fallbackMatch);
        const endTime = performance.now();

        setSearchState({
          query,
          results,
          isLoading: false,
          error: null,
          searchType: 'fallback',
          inferenceTime: endTime - startTime,
        });
        return;
      }

      // Slow path: use AI inference
      if (!modelState.isLoaded) {
        // Try to initialize the model if not already loading
        if (!modelState.isLoading) {
          initWorker();
        }
        const progressPct = Math.round(modelState.progress);
        const progressMsg = progressPct > 0
          ? ` (${progressPct}% loaded)`
          : '';
        setSearchState((prev) => ({
          ...prev,
          isLoading: false,
          error: `AI model is still loading${progressMsg}. Common queries like "pneumonia" or "cardiomegaly" work instantly. For custom queries, please wait for the model to finish loading.`,
        }));
        return;
      }

      try {
        const queryEmbedding = await getEmbedding(query);
        
        const topK = findTopK(
          queryEmbedding,
          indexData.embeddings,
          indexData.numImages,
          indexData.embeddingDim,
          10
        );

        const results = topK.map(([id, score]) => ({
          image: indexData.metadata[id],
          score,
        }));

        const endTime = performance.now();

        setSearchState({
          query,
          results,
          isLoading: false,
          error: null,
          searchType: 'inference',
          inferenceTime: endTime - startTime,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Search failed';
        setSearchState((prev) => ({
          ...prev,
          isLoading: false,
          error: message,
        }));
      }
    },
    [modelState.isLoaded, modelState.isLoading, modelState.progress, getEmbedding, toSearchResults, initWorker]
  );

  /**
   * Find similar images using pre-computed nearest neighbors.
   */
  const findSimilar = useCallback((imageId: number) => {
    const indexData = indexDataRef.current;
    if (!indexData) {
      setSearchState((prev) => ({
        ...prev,
        error: 'Index not loaded',
      }));
      return;
    }

    const startTime = performance.now();

    // Look up pre-computed neighbors
    const neighbors = indexData.nearestNeighbors[String(imageId)];

    if (!neighbors) {
      setSearchState((prev) => ({
        ...prev,
        error: `No similar images found for ID ${imageId}`,
      }));
      return;
    }

    const results = toSearchResults(neighbors);
    const endTime = performance.now();

    setSearchState({
      query: `Similar to image #${imageId}`,
      results,
      isLoading: false,
      error: null,
      searchType: 'fallback',
      inferenceTime: endTime - startTime,
    });
  }, [toSearchResults]);

  /**
   * Check if system is ready for search.
   */
  const isReady = indexState.isLoaded;

  return {
    search,
    findSimilar,
    searchState,
    modelState,
    indexState,
    isReady,
  };
}
