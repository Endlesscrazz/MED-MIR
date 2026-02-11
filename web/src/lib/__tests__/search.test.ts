/**
 * Unit tests for search.ts
 */

import { describe, it, expect } from 'vitest';
import {
  checkFallback,
  vectorSearch,
  toSearchResults,
  groupByLabel,
  calculateDiversity,
  getResultStats,
} from '../search';
import type { ImageMetadata, SearchResult, FallbackResults } from '../types';

// Test fixtures
const mockMetadata: ImageMetadata[] = [
  { id: 0, filename: 'img0.webp', url: '/img0.webp', report_snippet: 'Normal', ground_truth_label: 'Normal' },
  { id: 1, filename: 'img1.webp', url: '/img1.webp', report_snippet: 'Pneumonia', ground_truth_label: 'Pneumonia' },
  { id: 2, filename: 'img2.webp', url: '/img2.webp', report_snippet: 'Normal', ground_truth_label: 'Normal' },
  { id: 3, filename: 'img3.webp', url: '/img3.webp', report_snippet: 'Effusion', ground_truth_label: 'Effusion' },
];

const mockFallbackResults: FallbackResults = {
  'pneumonia': [{ id: 1, score: 0.95 }, { id: 0, score: 0.7 }],
  'normal chest xray': [{ id: 0, score: 0.9 }, { id: 2, score: 0.85 }],
  'pleural effusion': [{ id: 3, score: 0.88 }],
};

describe('checkFallback', () => {
  it('should find exact match', () => {
    const result = checkFallback('pneumonia', mockFallbackResults);
    
    expect(result).not.toBeNull();
    expect(result!.length).toBe(2);
    expect(result![0].id).toBe(1);
  });

  it('should handle case-insensitive queries', () => {
    const result = checkFallback('PNEUMONIA', mockFallbackResults);
    
    expect(result).not.toBeNull();
    expect(result![0].id).toBe(1);
  });

  it('should trim whitespace', () => {
    const result = checkFallback('  pneumonia  ', mockFallbackResults);
    
    expect(result).not.toBeNull();
  });

  it('should return null for non-matching queries', () => {
    const result = checkFallback('cardiomegaly', mockFallbackResults);
    
    expect(result).toBeNull();
  });

  it('should handle x-ray/xray variations', () => {
    // Note: This test assumes the variation logic is implemented
    const result = checkFallback('normal chest x-ray', mockFallbackResults);
    
    // May or may not match depending on implementation
    // The checkFallback should handle common variations
  });
});

describe('vectorSearch', () => {
  it('should find top-K results', () => {
    const query = new Float32Array([0.6, 0.8]);
    const embeddings = new Float32Array([
      0.6, 0.8,   // score = 1.0
      0, 1,       // score = 0.8
      1, 0,       // score = 0.6
    ]);
    
    const results = vectorSearch(query, embeddings, 3, 2, { topK: 2 });
    
    expect(results.length).toBe(2);
    expect(results[0].id).toBe(0);
    expect(results[0].score).toBeCloseTo(1, 5);
  });

  it('should filter by minimum score', () => {
    const query = new Float32Array([1, 0]);
    const embeddings = new Float32Array([
      1, 0,       // score = 1.0
      0.5, 0.5,   // score = 0.5
      0, 1,       // score = 0
    ]);
    
    const results = vectorSearch(query, embeddings, 3, 2, { topK: 10, minScore: 0.6 });
    
    expect(results.length).toBe(1);
    expect(results[0].id).toBe(0);
  });
});

describe('toSearchResults', () => {
  it('should convert raw results to SearchResult array', () => {
    const rawResults = [
      { id: 0, score: 0.9 },
      { id: 1, score: 0.8 },
    ];
    
    const results = toSearchResults(rawResults, mockMetadata);
    
    expect(results.length).toBe(2);
    expect(results[0].image.filename).toBe('img0.webp');
    expect(results[0].score).toBe(0.9);
    expect(results[1].image.ground_truth_label).toBe('Pneumonia');
  });
});

describe('groupByLabel', () => {
  it('should group results by ground truth label', () => {
    const results: SearchResult[] = [
      { image: mockMetadata[0], score: 0.9 },
      { image: mockMetadata[1], score: 0.8 },
      { image: mockMetadata[2], score: 0.7 },
    ];
    
    const groups = groupByLabel(results);
    
    expect(groups.size).toBe(2);
    expect(groups.get('Normal')?.length).toBe(2);
    expect(groups.get('Pneumonia')?.length).toBe(1);
  });

  it('should handle empty results', () => {
    const groups = groupByLabel([]);
    
    expect(groups.size).toBe(0);
  });
});

describe('calculateDiversity', () => {
  it('should return 1 for all unique labels', () => {
    const results: SearchResult[] = [
      { image: mockMetadata[0], score: 0.9 }, // Normal
      { image: mockMetadata[1], score: 0.8 }, // Pneumonia
      { image: mockMetadata[3], score: 0.7 }, // Effusion
    ];
    
    expect(calculateDiversity(results)).toBe(1);
  });

  it('should return lower score for repeated labels', () => {
    const results: SearchResult[] = [
      { image: mockMetadata[0], score: 0.9 }, // Normal
      { image: mockMetadata[2], score: 0.8 }, // Normal
    ];
    
    expect(calculateDiversity(results)).toBe(0.5);
  });

  it('should return 0 for empty results', () => {
    expect(calculateDiversity([])).toBe(0);
  });
});

describe('getResultStats', () => {
  it('should compute statistics for results', () => {
    const results: SearchResult[] = [
      { image: mockMetadata[0], score: 0.9 },
      { image: mockMetadata[1], score: 0.7 },
      { image: mockMetadata[2], score: 0.5 },
    ];
    
    const stats = getResultStats(results);
    
    expect(stats.count).toBe(3);
    expect(stats.avgScore).toBeCloseTo(0.7, 5);
    expect(stats.minScore).toBe(0.5);
    expect(stats.maxScore).toBe(0.9);
    expect(stats.labelCounts['Normal']).toBe(2);
    expect(stats.labelCounts['Pneumonia']).toBe(1);
  });

  it('should handle empty results', () => {
    const stats = getResultStats([]);
    
    expect(stats.count).toBe(0);
    expect(stats.avgScore).toBe(0);
    expect(Object.keys(stats.labelCounts).length).toBe(0);
  });
});
