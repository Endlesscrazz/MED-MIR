/**
 * Image-to-image similarity using pre-computed nearest neighbors.
 * 
 * This module provides instant "Find Similar" functionality
 * without requiring inference by using pre-computed results.
 */

import type { 
  NearestNeighbors, 
  ImageMetadata, 
  SearchResult,
} from './types';

/**
 * Find similar images using pre-computed nearest neighbors.
 * 
 * @param imageId - ID of the source image
 * @param neighbors - Pre-computed nearest neighbors map
 * @param metadata - Image metadata array
 * @param limit - Maximum number of results
 * @returns Array of similar images with scores
 */
export function findSimilarImages(
  imageId: number,
  neighbors: NearestNeighbors,
  metadata: ImageMetadata[],
  limit?: number
): SearchResult[] {
  const key = String(imageId);
  const neighborList = neighbors[key];
  
  if (!neighborList) {
    return [];
  }
  
  const results = neighborList.map(({ id, score }) => ({
    image: metadata[id],
    score,
  }));
  
  return limit ? results.slice(0, limit) : results;
}

/**
 * Get the source image metadata for a similarity search.
 * 
 * @param imageId - ID of the source image
 * @param metadata - Image metadata array
 * @returns Image metadata or null if not found
 */
export function getSourceImage(
  imageId: number,
  metadata: ImageMetadata[]
): ImageMetadata | null {
  if (imageId < 0 || imageId >= metadata.length) {
    return null;
  }
  return metadata[imageId];
}

/**
 * Check if two images are similar based on pre-computed neighbors.
 * 
 * @param imageId1 - First image ID
 * @param imageId2 - Second image ID
 * @param neighbors - Pre-computed nearest neighbors map
 * @returns Similarity score if related, null otherwise
 */
export function getSimilarity(
  imageId1: number,
  imageId2: number,
  neighbors: NearestNeighbors
): number | null {
  const neighborList = neighbors[String(imageId1)];
  
  if (!neighborList) {
    return null;
  }
  
  const match = neighborList.find(n => n.id === imageId2);
  return match?.score ?? null;
}

/**
 * Find images that are mutually similar (both ways).
 * 
 * @param imageId - Source image ID
 * @param neighbors - Pre-computed nearest neighbors map
 * @param metadata - Image metadata array
 * @returns Array of mutually similar images
 */
export function findMutualNeighbors(
  imageId: number,
  neighbors: NearestNeighbors,
  metadata: ImageMetadata[]
): SearchResult[] {
  const myNeighbors = neighbors[String(imageId)];
  
  if (!myNeighbors) {
    return [];
  }
  
  const mutual: SearchResult[] = [];
  
  for (const neighbor of myNeighbors) {
    const theirNeighbors = neighbors[String(neighbor.id)];
    
    if (theirNeighbors) {
      const backReference = theirNeighbors.find(n => n.id === imageId);
      
      if (backReference) {
        // Calculate mutual score as average of both directions
        const mutualScore = (neighbor.score + backReference.score) / 2;
        mutual.push({
          image: metadata[neighbor.id],
          score: mutualScore,
        });
      }
    }
  }
  
  // Sort by mutual score
  mutual.sort((a, b) => b.score - a.score);
  
  return mutual;
}

/**
 * Get similarity clusters - groups of images that are all similar to each other.
 * 
 * @param imageId - Starting image ID
 * @param neighbors - Pre-computed nearest neighbors map
 * @param metadata - Image metadata array
 * @param depth - How many hops to explore (default 2)
 * @returns Set of related image IDs
 */
export function getSimilarityCluster(
  imageId: number,
  neighbors: NearestNeighbors,
  metadata: ImageMetadata[],
  depth: number = 2
): Set<number> {
  const cluster = new Set<number>([imageId]);
  const toExplore = [imageId];
  
  for (let d = 0; d < depth && toExplore.length > 0; d++) {
    const currentBatch = [...toExplore];
    toExplore.length = 0;
    
    for (const id of currentBatch) {
      const neighborList = neighbors[String(id)];
      
      if (neighborList) {
        // Only consider high-confidence neighbors
        const highConfidence = neighborList.filter(n => n.score > 0.5);
        
        for (const neighbor of highConfidence) {
          if (!cluster.has(neighbor.id)) {
            cluster.add(neighbor.id);
            toExplore.push(neighbor.id);
          }
        }
      }
    }
  }
  
  return cluster;
}

/**
 * Find images with the same label that are most similar.
 * 
 * @param imageId - Source image ID
 * @param neighbors - Pre-computed nearest neighbors map
 * @param metadata - Image metadata array
 * @returns Similar images with matching label
 */
export function findSimilarWithSameLabel(
  imageId: number,
  neighbors: NearestNeighbors,
  metadata: ImageMetadata[]
): SearchResult[] {
  const sourceImage = metadata[imageId];
  
  if (!sourceImage) {
    return [];
  }
  
  const similar = findSimilarImages(imageId, neighbors, metadata);
  
  return similar.filter(
    result => result.image.ground_truth_label === sourceImage.ground_truth_label
  );
}

/**
 * Find images with different labels that are unexpectedly similar.
 * These could indicate potential confusion cases.
 * 
 * @param imageId - Source image ID
 * @param neighbors - Pre-computed nearest neighbors map
 * @param metadata - Image metadata array
 * @param minScore - Minimum similarity to consider "unexpected"
 * @returns Similar images with different labels
 */
export function findUnexpectedSimilar(
  imageId: number,
  neighbors: NearestNeighbors,
  metadata: ImageMetadata[],
  minScore: number = 0.6
): SearchResult[] {
  const sourceImage = metadata[imageId];
  
  if (!sourceImage) {
    return [];
  }
  
  const similar = findSimilarImages(imageId, neighbors, metadata);
  
  return similar.filter(
    result => 
      result.image.ground_truth_label !== sourceImage.ground_truth_label &&
      result.score >= minScore
  );
}
