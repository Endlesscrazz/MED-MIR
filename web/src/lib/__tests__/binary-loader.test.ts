/**
 * Unit tests for binary-loader.ts
 */

import { describe, it, expect } from 'vitest';
import { 
  dotProduct, 
  l2Normalize, 
  getEmbedding,
  findTopK,
} from '../binary-loader';

describe('dotProduct', () => {
  it('should compute dot product of two vectors', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expect(dotProduct(a, b)).toBe(32);
  });

  it('should return 0 for orthogonal vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    
    expect(dotProduct(a, b)).toBe(0);
  });

  it('should return 1 for identical normalized vectors', () => {
    const vec = new Float32Array([0.6, 0.8]); // Already normalized
    
    expect(dotProduct(vec, vec)).toBeCloseTo(1, 5);
  });

  it('should throw for mismatched dimensions', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2]);
    
    expect(() => dotProduct(a, b)).toThrow();
  });
});

describe('l2Normalize', () => {
  it('should normalize a vector to unit length', () => {
    const vec = new Float32Array([3, 4]);
    l2Normalize(vec);
    
    // 3-4-5 triangle: normalized should be [0.6, 0.8]
    expect(vec[0]).toBeCloseTo(0.6, 5);
    expect(vec[1]).toBeCloseTo(0.8, 5);
    
    // Verify unit length
    const norm = Math.sqrt(vec[0] ** 2 + vec[1] ** 2);
    expect(norm).toBeCloseTo(1, 5);
  });

  it('should handle zero vector gracefully', () => {
    const vec = new Float32Array([0, 0, 0]);
    l2Normalize(vec);
    
    expect(vec[0]).toBe(0);
    expect(vec[1]).toBe(0);
    expect(vec[2]).toBe(0);
  });

  it('should return the same reference', () => {
    const vec = new Float32Array([1, 1, 1]);
    const result = l2Normalize(vec);
    
    expect(result).toBe(vec);
  });
});

describe('getEmbedding', () => {
  it('should extract correct embedding from array', () => {
    // 3 embeddings of dimension 2
    const embeddings = new Float32Array([1, 2, 3, 4, 5, 6]);
    
    const first = getEmbedding(embeddings, 0, 2);
    expect(Array.from(first)).toEqual([1, 2]);
    
    const second = getEmbedding(embeddings, 1, 2);
    expect(Array.from(second)).toEqual([3, 4]);
    
    const third = getEmbedding(embeddings, 2, 2);
    expect(Array.from(third)).toEqual([5, 6]);
  });

  it('should handle higher dimensions', () => {
    // 2 embeddings of dimension 4
    const embeddings = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    
    const first = getEmbedding(embeddings, 0, 4);
    expect(Array.from(first)).toEqual([1, 2, 3, 4]);
    
    const second = getEmbedding(embeddings, 1, 4);
    expect(Array.from(second)).toEqual([5, 6, 7, 8]);
  });
});

describe('findTopK', () => {
  it('should find top-K most similar embeddings', () => {
    // Create normalized query
    const query = new Float32Array([0.6, 0.8]);
    
    // Create normalized image embeddings (3 images, dim=2)
    const imageEmbeddings = new Float32Array([
      0.6, 0.8,    // Image 0: identical to query (score = 1)
      -0.6, -0.8,  // Image 1: opposite (score = -1)
      0.8, 0.6,    // Image 2: similar (score = 0.96)
    ]);
    
    const results = findTopK(query, imageEmbeddings, 3, 2, 2);
    
    expect(results.length).toBe(2);
    expect(results[0][0]).toBe(0); // First result is image 0
    expect(results[0][1]).toBeCloseTo(1, 5);
    expect(results[1][0]).toBe(2); // Second result is image 2
    expect(results[1][1]).toBeCloseTo(0.96, 2);
  });

  it('should return K or fewer results', () => {
    const query = new Float32Array([1, 0]);
    const imageEmbeddings = new Float32Array([1, 0, 0, 1]);
    
    // Request 5 but only 2 images exist
    const results = findTopK(query, imageEmbeddings, 2, 2, 5);
    
    expect(results.length).toBe(2);
  });

  it('should sort results by score descending', () => {
    const query = new Float32Array([1, 0]);
    const imageEmbeddings = new Float32Array([
      0, 1,    // Image 0: score = 0
      1, 0,    // Image 1: score = 1
      0.5, 0.5, // Image 2: score = 0.5 (not normalized, but ok for test)
    ]);
    
    const results = findTopK(query, imageEmbeddings, 3, 2, 3);
    
    expect(results[0][0]).toBe(1); // Highest score first
    expect(results[1][0]).toBe(2);
    expect(results[2][0]).toBe(0);
  });
});

describe('cosine similarity property', () => {
  it('should compute cosine similarity via dot product of normalized vectors', () => {
    // Two vectors at 45 degree angle
    const a = l2Normalize(new Float32Array([1, 0]));
    const b = l2Normalize(new Float32Array([1, 1]));
    
    const similarity = dotProduct(a, b);
    
    // cos(45°) ≈ 0.707
    expect(similarity).toBeCloseTo(Math.cos(Math.PI / 4), 5);
  });
});
