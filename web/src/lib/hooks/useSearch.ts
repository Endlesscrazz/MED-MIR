'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { SearchResult, SearchState, ModelState } from '@/lib/types';
import { loadIndexData, findTopK, type IndexData } from '@/lib/binary-loader';
import { generateId } from '@/lib/utils';
import { preprocessImage } from '@/lib/image-processor';
import { checkFallback } from '@/lib/search';

export function useSearch() {
  const dataBaseUrl = process.env.NEXT_PUBLIC_DATA_URL || '/demo-data';

  const [searchState, setSearchState] = useState<SearchState>({
    query: '',
    results: [],
    isLoading: false,
    error: null,
    searchType: null,
    inferenceTime: null
  });

  const [modelState, setModelState] = useState<ModelState>({
    isLoaded: false,
    isLoading: false,
    progress: 0,
    error: null
  });

  const [indexState, setIndexState] = useState({
    isLoaded: false,
    isLoading: false,
    progress: 0,
    error: null
  });
  
  const indexDataRef = useRef<IndexData | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const pendingRequestsRef = useRef<Map<string, (embedding: Float32Array) => void>>(new Map());

  const initWorker = useCallback(() => {
    if (typeof window === 'undefined' || workerRef.current) return;
    
    workerRef.current = new Worker('/workers/inference.worker.js');
    
    workerRef.current.onmessage = (event) => {
      const { type, id, payload } = event.data;
      switch (type) {
        case 'INIT_COMPLETE':
          // The worker sends 'text' or 'vision' as id to tell us which model is ready
          if (id === 'text') {
            setModelState(prev => ({ ...prev, isLoaded: true, isLoading: false }));
          }
          break;
        case 'PROGRESS':
          setModelState(prev => ({ ...prev, progress: (payload?.progress || 0) * 100 }));
          break;
        case 'EMBED_RESULT':
          const callback = pendingRequestsRef.current.get(id);
          if (callback) {
            callback(new Float32Array(payload.embedding));
            pendingRequestsRef.current.delete(id);
          }
          break;
        case 'EMBED_ERROR':
          setSearchState(prev => ({ ...prev, isLoading: false, error: payload?.error }));
          break;
      }
    };

    workerRef.current.postMessage({ type: 'INIT', payload: { dataBaseUrl } });
  }, [dataBaseUrl]);

  useEffect(() => {
    async function load() {
      try {
        const data = await loadIndexData();
        indexDataRef.current = data;
        setIndexState(prev => ({ ...prev, isLoaded: true }));
        initWorker();
      } catch (e: any) {
        setIndexState(prev => ({ ...prev, error: e.message }));
      }
    }
    load();
  }, [initWorker]);

  const performVectorSearch = useCallback((embedding: Float32Array, queryText: string, searchType: 'fallback' | 'inference') => {
    if (!indexDataRef.current) return;
    const startTime = performance.now();
    
    // 512 is the dimension for BiomedCLIP
    const topK = findTopK(embedding, indexDataRef.current.embeddings, indexDataRef.current.numImages, 512, 10);
    const results = topK.map(([id, score]) => ({
      image: indexDataRef.current!.metadata[id],
      score
    }));
    
    setSearchState({
      query: queryText,
      results,
      isLoading: false,
      error: null,
      searchType: searchType,
      inferenceTime: performance.now() - startTime
    });
  }, []);

  // 1. Natural Language Search
  const search = useCallback(async (text: string) => {
    if (!text.trim()) {
      setSearchState({ query: '', results: [], isLoading: false, error: null, searchType: null, inferenceTime: null });
      return;
    }

    setSearchState(prev => ({ ...prev, isLoading: true, query: text }));

    // Fast path: serve precomputed results for common clinical queries.
    const indexData = indexDataRef.current;
    if (indexData) {
      const fallbackHit = checkFallback(text, indexData.fallbackResults);
      if (fallbackHit) {
        const startTime = performance.now();
        const results = fallbackHit
          .map(({ id, score }) => ({
            image: indexData.metadata[id],
            score,
          }))
          .filter((r) => Boolean(r.image));

        setSearchState({
          query: text,
          results,
          isLoading: false,
          error: null,
          searchType: 'fallback',
          inferenceTime: performance.now() - startTime,
        });
        return;
      }
    }
    
    const id = generateId();
    const promise = new Promise<Float32Array>(res => pendingRequestsRef.current.set(id, res));
    
    workerRef.current?.postMessage({ type: 'EMBED_TEXT', id, payload: { text } });
    
    const embedding = await promise;
    performVectorSearch(embedding, text, 'inference');
  }, [performVectorSearch]);

  // 2. Image Upload Search (The new Dual-Encoder part)
  const searchByImage = useCallback(async (file: File) => {
    setSearchState(prev => ({ ...prev, isLoading: true, query: `Image: ${file.name}` }));
    try {
      const pixelValues = await preprocessImage(file);
      const id = generateId();
      const promise = new Promise<Float32Array>(res => pendingRequestsRef.current.set(id, res));
      
      workerRef.current?.postMessage({ type: 'EMBED_IMAGE', id, payload: { pixelValues } });
      
      const embedding = await promise;
      performVectorSearch(embedding, `Visually similar to ${file.name}`, 'inference');
    } catch (e: any) {
      setSearchState(prev => ({ ...prev, isLoading: false, error: e.message }));
    }
  }, [performVectorSearch]);

  // 3. Find Similar (Pre-computed Lookup) - This was missing!
  const findSimilar = useCallback((imageId: number) => {
    if (!indexDataRef.current) return;
    const startTime = performance.now();

    const neighbors = indexDataRef.current.nearestNeighbors[String(imageId)];
    if (!neighbors) {
      setSearchState(prev => ({ ...prev, error: `No similar images pre-computed for ID ${imageId}` }));
      return;
    }

    const results = neighbors.map(({ id, score }) => ({
      image: indexDataRef.current!.metadata[id],
      score
    }));

    setSearchState({
      query: `Visually similar to #${imageId}`,
      results,
      isLoading: false,
      error: null,
      searchType: 'fallback',
      inferenceTime: performance.now() - startTime
    });
  }, []);

  return {
    search,
    searchByImage,
    findSimilar, 
    searchState,
    modelState,
    indexState,
    isReady: indexState.isLoaded
  };
}
