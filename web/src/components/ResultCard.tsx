'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Search, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { cn, truncateText } from '@/lib/utils';
import type { SearchResult } from '@/lib/types';

interface ResultCardProps {
  /** Search result to display */
  result: SearchResult;
  /** Callback when "Find Similar" is clicked */
  onFindSimilar?: (imageId: number) => void;
  /** Optional rank number */
  rank?: number;
}

/**
 * Individual search result card.
 * 
 * Displays:
 * - Medical image thumbnail
 * - Similarity score badge
 * - Ground truth label
 * - Report snippet (expandable)
 * - "Find Similar" button
 */
export function ResultCard({ result, onFindSimilar, rank }: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const { image, score } = result;
  const hasReport = image.report_snippet && image.report_snippet !== 'No report available';
  const labelText = image.ground_truth_label === 'Unknown'
    ? 'Label unavailable'
    : image.ground_truth_label;
  const labelSource = image.label_source ?? 'unknown';
  const labelSourceText = labelSource === 'dataset'
    ? 'NIH ChestX-ray14 label'
    : labelSource === 'verified'
      ? 'Verified'
      : labelSource === 'raw'
        ? 'Source: dataset label'
        : labelSource === 'derived'
          ? 'Source: derived from report'
          : 'Source: unavailable';

  return (
    <div className="result-card group relative overflow-hidden">
      {/* Rank Badge */}
      {rank !== undefined && (
        <div className="absolute left-2 top-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-background/90 text-xs font-bold shadow">
          {rank}
        </div>
      )}

      {/* Similarity Score Badge + Layman Tooltip */}
      <div className="absolute right-2 top-2 z-20">
        <div
          tabIndex={0}
          aria-label={`Match score ${Math.round(score * 100)} percent. Hover or focus for explanation.`}
          className="peer flex cursor-help items-center gap-1 rounded-full bg-black/80 px-2.5 py-1 shadow-md backdrop-blur-sm outline-none"
        >
          <span className="text-[10px] font-medium text-white/70">Match</span>
          <span className="text-xs font-bold text-white">{Math.round(score * 100)}%</span>
        </div>
        <div
          role="tooltip"
          className="pointer-events-none absolute right-0 mt-2 w-64 translate-y-1 rounded-lg border bg-background/95 p-3 text-[11px] leading-relaxed text-foreground shadow-xl opacity-0 backdrop-blur-sm transition-all duration-150 peer-hover:translate-y-0 peer-hover:opacity-100 peer-focus-visible:translate-y-0 peer-focus-visible:opacity-100"
        >
          Match % tells how visually/textually similar this result is to your query compared against other images in this
          dataset. Higher means more similar. Use it to rank results (for example, 26% is a stronger match than 24%), not
          as a diagnosis probability.
        </div>
      </div>

      {/* Image */}
      <div className="relative mb-3 aspect-square overflow-hidden rounded-lg bg-black">
        {imageError ? (
          <div className="flex h-full items-center justify-center bg-muted">
            <span className="text-sm text-muted-foreground">Image unavailable</span>
          </div>
        ) : (
          <Image
            src={image.url}
            alt={`Medical image: ${image.ground_truth_label}`}
            fill
            className="xray-image"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
            onError={() => setImageError(true)}
          />
        )}
      </div>

      {/* Label */}
      <div className="mb-2 flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center rounded-md bg-secondary px-2 py-1 text-xs font-medium">
            {labelText}
          </span>
          {image.label_verified && (
            <span className="inline-flex items-center rounded-md bg-green-100 px-2 py-1 text-xs font-medium text-green-800">
              Verified
            </span>
          )}
        </div>
        <span className="text-xs text-muted-foreground">{labelSourceText}</span>
      </div>

      {/* Report Snippet */}
      {hasReport && (
        <div className="mb-3">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex w-full items-center justify-between text-left text-sm text-muted-foreground hover:text-foreground"
          >
            <span>Report</span>
            {isExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
          <p
            className={cn(
              'mt-1 text-sm text-muted-foreground transition-all',
              isExpanded ? '' : 'line-clamp-2'
            )}
          >
            {isExpanded ? image.report_snippet : truncateText(image.report_snippet, 100)}
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2">
        {onFindSimilar && (
          <button
            onClick={() => onFindSimilar(image.id)}
            className={cn(
              'flex flex-1 items-center justify-center gap-2 rounded-lg border py-2 text-sm',
              'transition-colors hover:bg-accent hover:text-accent-foreground'
            )}
          >
            <Search className="h-4 w-4" />
            Find Similar
          </button>
        )}
        <a
          href={image.url}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'flex items-center justify-center rounded-lg border p-2',
            'transition-colors hover:bg-accent hover:text-accent-foreground'
          )}
          aria-label="View full image"
        >
          <ExternalLink className="h-4 w-4" />
        </a>
      </div>

      {/* Metadata Footer */}
      <div className="mt-3 flex items-center justify-between border-t pt-3 text-xs text-muted-foreground">
        <span>ID: {image.id}</span>
        <span>{image.filename}</span>
      </div>
    </div>
  );
}

/**
 * Loading skeleton for ResultCard.
 */
export function ResultCardSkeleton() {
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="skeleton mb-3 aspect-square rounded-lg" />
      <div className="skeleton mb-2 h-6 w-20 rounded-md" />
      <div className="skeleton mb-1 h-4 w-full rounded" />
      <div className="skeleton mb-3 h-4 w-3/4 rounded" />
      <div className="skeleton h-9 w-full rounded-lg" />
    </div>
  );
}
