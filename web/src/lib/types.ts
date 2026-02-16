/**
 * Type definitions for Med-MIR application.
 * 
 * These interfaces define the structure of data used throughout
 * the application, ensuring type safety and documentation.
 */

/**
 * Metadata for a single medical image in the index.
 */
export interface ImageMetadata {
  /** Unique numeric identifier (index in embeddings array) */
  id: number;
  /** Filename of the image (e.g., "image_001.webp") */
  filename: string;
  /** URL to the image (relative or absolute) */
  url: string;
  /** Snippet from the radiology report */
  report_snippet: string;
  /** Ground truth diagnostic label */
  ground_truth_label: string;
  /** Source of the label (derived, raw, verified, or unknown) */
  label_source?: 'dataset' | 'derived' | 'raw' | 'verified' | 'unknown';
  /** Whether label was manually verified */
  label_verified?: boolean;
}

/**
 * Search result combining image metadata with similarity score.
 */
export interface SearchResult {
  /** The image metadata */
  image: ImageMetadata;
  /** Cosine similarity score (0-1, higher is better) */
  score: number;
}

/**
 * Pre-computed fallback result for common queries.
 */
export interface FallbackResult {
  /** Image ID (index in metadata array) */
  id: number;
  /** Pre-computed similarity score */
  score: number;
}

/**
 * Structure of fallback_results.json.
 * Maps lowercase query strings to arrays of results.
 */
export interface FallbackResults {
  [query: string]: FallbackResult[];
}

/**
 * Structure of nearest_neighbors.json.
 * Maps image ID (as string) to arrays of similar images.
 */
export interface NearestNeighbors {
  [imageId: string]: FallbackResult[];
}

/**
 * Overall retrieval metrics.
 */
export interface OverallMetrics {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  mAP: number;
  MRR: number;
}

/**
 * Per-label metrics.
 */
export interface LabelMetrics {
  recall_at_10: number;
  num_images: number;
  num_queries: number;
  /** Whether the dataset contains any images with this label */
  has_images?: boolean;
}

/**
 * Dataset statistics.
 */
export interface DatasetStats {
  num_images: number;
  num_labels: number;
  label_distribution: Record<string, number>;
}

/**
 * Adjusted metrics (only viable queries — labels that exist in dataset).
 */
export interface AdjustedMetrics extends OverallMetrics {
  num_viable_queries: number;
  excluded_labels: string[];
}

/**
 * Semantic (soft) metrics — counts medically-related retrievals as hits.
 */
export interface SemanticMetrics {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  MRR: number;
  note: string;
}

/**
 * Full metrics structure from metrics.json.
 */
export interface Metrics {
  overall: OverallMetrics;
  adjusted?: AdjustedMetrics;
  semantic?: SemanticMetrics;
  per_label: Record<string, LabelMetrics>;
  dataset: DatasetStats;
  evaluation: {
    num_queries: number;
    num_viable_queries?: number;
    num_impossible_queries?: number;
    k_values: number[];
  };
}

/**
 * Single hard case example.
 */
export interface HardCaseExample {
  query: string;
  expected_label: string;
  retrieved_label: string;
  confusion_type: string;
  /** Whether expected and retrieved are semantically related */
  is_semantic_match?: boolean;
  top_results: {
    id: number;
    label: string;
    score: number;
    filename: string;
  }[];
}

/**
 * Confusion case summary.
 */
export interface ConfusionCase {
  query: string;
  expected: string;
  got: string;
  /** Whether expected and retrieved labels are semantically related */
  is_semantic_match?: boolean;
}

/**
 * Full hard cases structure from hard_cases.json.
 */
export interface HardCases {
  total_failures: number;
  /** Number of failures that are actually semantically related */
  semantic_matches?: number;
  /** Number of truly wrong failures */
  true_failures?: number;
  failure_rate: number;
  /** Failure rate excluding semantic matches */
  true_failure_rate?: number;
  /** Number of queries for labels absent from dataset */
  impossible_queries?: number;
  by_confusion_type: Record<string, ConfusionCase[]>;
  examples: HardCaseExample[];
}

/**
 * Index information from index_info.json.
 */
export interface IndexInfo {
  num_images: number;
  embedding_dim: number;
  model: string;
  num_fallback_queries: number;
  top_k: number;
  nearest_neighbors_k: number;
}

/**
 * Worker message types for inference worker communication.
 */
export type WorkerMessageType = 
  | 'INIT'
  | 'INIT_COMPLETE'
  | 'INIT_ERROR'
  | 'EMBED_TEXT'
  | 'EMBED_RESULT'
  | 'EMBED_ERROR'
  | 'STATUS'
  | 'PROGRESS';

/**
 * Message sent to the inference worker.
 */
export interface WorkerRequest {
  type: WorkerMessageType;
  id?: string;
  payload?: {
    text?: string;
    modelPath?: string;
  };
}

/**
 * Message received from the inference worker.
 */
export interface WorkerResponse {
  type: WorkerMessageType;
  id?: string;
  payload?: {
    embedding?: Float32Array;
    error?: string;
    progress?: number;
    status?: string;
  };
}

/**
 * Search state for the UI.
 */
export interface SearchState {
  query: string;
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  searchType: 'fallback' | 'inference' | null;
  inferenceTime: number | null;
}

/**
 * Model loading state.
 */
export interface ModelState {
  isLoaded: boolean;
  isLoading: boolean;
  progress: number;
  error: string | null;
}
