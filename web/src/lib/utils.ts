import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Combines class names with Tailwind merge support.
 * 
 * @param inputs - Class values to merge
 * @returns Merged class string
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a similarity score for display.
 * 
 * @param score - Raw similarity score (0-1)
 * @returns Formatted string (e.g., "0.85")
 */
export function formatScore(score: number): string {
  return score.toFixed(2);
}

/**
 * Get CSS class for score badge based on value.
 * 
 * @param score - Similarity score (0-1)
 * @returns CSS class name for styling
 */
export function getScoreClass(score: number): string {
  if (score >= 0.7) return 'score-high';
  if (score >= 0.4) return 'score-medium';
  return 'score-low';
}

/**
 * Format percentage for display.
 * 
 * @param value - Decimal value (0-1)
 * @returns Formatted percentage string
 */
export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format bytes to human readable size.
 * 
 * @param bytes - Size in bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format milliseconds to human readable duration.
 * 
 * @param ms - Duration in milliseconds
 * @returns Formatted string (e.g., "1.5s" or "150ms")
 */
export function formatDuration(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${Math.round(ms)}ms`;
}

/**
 * Debounce a function call.
 * 
 * @param fn - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Generate a unique ID for request tracking.
 * 
 * @returns Unique string ID
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Truncate text to a maximum length.
 * 
 * @param text - Text to truncate
 * @param maxLength - Maximum length
 * @returns Truncated text with ellipsis if needed
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

/**
 * Check if running in browser environment.
 * 
 * @returns True if in browser
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined';
}

/**
 * Check if Web Workers are supported.
 * 
 * @returns True if Web Workers are available
 */
export function supportsWebWorkers(): boolean {
  return isBrowser() && typeof Worker !== 'undefined';
}

/**
 * Common clinical query suggestions.
 */
export const QUERY_SUGGESTIONS = [
  'Normal chest xray',
  'Pneumonia',
  'Pleural effusion',
  'Cardiomegaly',
  'Atelectasis',
  'Pulmonary edema',
  'Pneumothorax',
  'Lung nodule',
  'Consolidation',
  'Emphysema',
] as const;
