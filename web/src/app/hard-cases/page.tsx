'use client';

import { useEffect, useState, useRef } from 'react';
import { loadHardCases } from '@/lib/binary-loader';
import type { HardCases, HardCaseExample } from '@/lib/types';
import { formatPercentage, formatScore, getScoreClass, cn } from '@/lib/utils';

/**
 * Hard cases analysis page.
 *
 * Displays cases where the model makes mistakes, including:
 * - Overall failure statistics (strict vs true failures)
 * - Semantic-match annotations
 * - Confusion patterns between labels
 * - Specific failure examples
 */
export default function HardCasesPage() {
  const [hardCases, setHardCases] = useState<HardCases | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedExample, setSelectedExample] = useState<HardCaseExample | null>(null);

  useEffect(() => {
    async function fetchHardCases() {
      try {
        const data = await loadHardCases();
        setHardCases(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load hard cases');
      } finally {
        setIsLoading(false);
      }
    }
    fetchHardCases();
  }, []);

  if (isLoading) {
    return (
      <div className="container py-8">
        <div className="flex items-center justify-center py-12">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      </div>
    );
  }

  if (error || !hardCases) {
    return (
      <div className="container py-8">
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-6">
          <h2 className="mb-2 text-lg font-semibold text-destructive">Error Loading Data</h2>
          <p className="text-sm text-destructive/80">{error || 'No hard cases data available'}</p>
          <p className="mt-4 text-sm text-muted-foreground">
            Run the Python evaluation pipeline to generate hard_cases.json
          </p>
        </div>
      </div>
    );
  }

  const semanticMatches = hardCases.semantic_matches ?? 0;
  const trueFailures = hardCases.true_failures ?? hardCases.total_failures;
  const impossibleQueries = hardCases.impossible_queries ?? 0;
  const trueFailureRate = hardCases.true_failure_rate ?? hardCases.failure_rate;

  return (
    <div className="container py-8">
      <div className="mb-8 flex items-start gap-3">
        <div>
          <h1 className="mb-2 text-3xl font-bold">Hard Case Analysis</h1>
          <p className="text-muted-foreground">
            Understanding where the model makes mistakes — and where &quot;mistakes&quot; are actually medically reasonable
          </p>
        </div>
        <InfoTooltip />
      </div>

      {/* Summary Stats */}
      <section className="mb-12">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <StatCard
            title="Strict Failures"
            value={String(hardCases.total_failures)}
            description="queries where top label ≠ expected"
            color="text-destructive"
          />
          <StatCard
            title="Semantic Matches"
            value={String(semanticMatches)}
            description="failures that are medically related"
            color="text-amber-600 dark:text-amber-400"
          />
          <StatCard
            title="True Failures"
            value={String(trueFailures)}
            description="genuinely wrong retrievals"
            color="text-destructive"
          />
          <StatCard
            title="True Failure Rate"
            value={formatPercentage(trueFailureRate)}
            description="excluding semantic matches"
            color="text-foreground"
          />
          {impossibleQueries > 0 && (
            <StatCard
              title="Impossible Queries"
              value={String(impossibleQueries)}
              description="label absent from dataset"
              color="text-muted-foreground"
            />
          )}
        </div>

        {/* Explanation */}
        <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-50 p-4 text-sm dark:bg-amber-900/20">
          <p className="font-medium text-amber-800 dark:text-amber-300">
            Why strict failures are inflated:
          </p>
          <ul className="mt-2 list-inside list-disc space-y-1 text-amber-700 dark:text-amber-400">
            {impossibleQueries > 0 && (
              <li>
                {impossibleQueries} queries test for labels with <strong>zero images</strong> in the dataset — these
                are guaranteed failures.
              </li>
            )}
            {semanticMatches > 0 && (
              <li>
                {semanticMatches} &quot;failures&quot; retrieved <strong>medically related</strong> labels (e.g.,
                Nodule→Opacity, Edema→Effusion) — clinically reasonable retrievals.
              </li>
            )}
          </ul>
        </div>
      </section>

      {/* Confusion Patterns */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Confusion Patterns</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Labels that the model frequently confuses with each other.{' '}
          <span className="text-amber-600 dark:text-amber-400">Amber</span> = semantically related.
        </p>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Object.entries(hardCases.by_confusion_type).map(([type, cases]) => {
            const semanticCount = cases.filter((c) => c.is_semantic_match).length;
            return (
              <div key={type} className="rounded-lg border bg-card p-4">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold">{type === 'unknown' ? 'Uncategorized' : type}</h3>
                  <div className="flex gap-2">
                    {semanticCount > 0 && (
                      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                        {semanticCount} semantic
                      </span>
                    )}
                    <span className="rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive">
                      {cases.length} cases
                    </span>
                  </div>
                </div>
                <div className="space-y-2">
                  {cases.slice(0, 4).map((c, i) => (
                    <div key={i} className="text-sm">
                      <span className="text-muted-foreground">&quot;{c.query}&quot;</span>
                      <div className="mt-1 flex items-center gap-2 text-xs">
                        <span className="text-green-600 dark:text-green-400">Expected: {c.expected}</span>
                        <span className={c.is_semantic_match ? 'text-amber-600 dark:text-amber-400' : 'text-destructive'}>
                          Got: {c.got}
                          {c.is_semantic_match && ' ≈'}
                        </span>
                      </div>
                    </div>
                  ))}
                  {cases.length > 4 && (
                    <p className="text-xs text-muted-foreground">+{cases.length - 4} more cases</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Example Failures */}
      <section>
        <h2 className="mb-4 text-xl font-semibold">Detailed Examples</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Click on an example to see the full retrieval results.{' '}
          <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
            ≈ semantic
          </span>{' '}
          marks medically-related retrievals.
        </p>

        <div className="grid gap-4 lg:grid-cols-2">
          {/* Example List */}
          <div className="max-h-[600px] space-y-2 overflow-y-auto pr-2">
            {hardCases.examples.map((example, i) => (
              <button
                key={i}
                onClick={() => setSelectedExample(example)}
                className={cn(
                  'w-full rounded-lg border p-4 text-left transition-colors hover:bg-accent',
                  selectedExample === example && 'border-primary bg-accent',
                )}
              >
                <div className="flex items-center gap-2">
                  <p className="font-medium">&quot;{example.query}&quot;</p>
                  {example.is_semantic_match && (
                    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                      ≈ semantic
                    </span>
                  )}
                </div>
                <div className="mt-2 flex items-center gap-4 text-sm">
                  <span className="text-muted-foreground">
                    Expected: <span className="text-green-600 dark:text-green-400">{example.expected_label}</span>
                  </span>
                  <span className="text-muted-foreground">
                    Got:{' '}
                    <span className={example.is_semantic_match ? 'text-amber-600 dark:text-amber-400' : 'text-destructive'}>
                      {example.retrieved_label}
                    </span>
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {example.confusion_type === 'unknown' ? 'Uncategorized' : example.confusion_type}
                </p>
              </button>
            ))}
          </div>

          {/* Selected Example Detail */}
          <div className="rounded-lg border bg-card p-4">
            {selectedExample ? (
              <>
                <h3 className="mb-4 font-semibold">
                  Results for: &quot;{selectedExample.query}&quot;
                  {selectedExample.is_semantic_match && (
                    <span className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-normal text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                      ≈ semantic match
                    </span>
                  )}
                </h3>
                <div className="space-y-3">
                  {selectedExample.top_results.map((result, i) => (
                    <div
                      key={i}
                      className={cn(
                        'flex items-center justify-between rounded-lg border p-3',
                        result.label.toLowerCase() === selectedExample.expected_label.toLowerCase()
                          ? 'border-green-500/30 bg-green-500/5'
                          : i === 0
                            ? selectedExample.is_semantic_match
                              ? 'border-amber-500/30 bg-amber-500/5'
                              : 'border-destructive/30 bg-destructive/5'
                            : '',
                      )}
                    >
                      <div>
                        <span className="text-sm font-medium">#{i + 1}</span>
                        <span className="ml-2 text-sm">{result.filename}</span>
                        <p className="mt-1 text-sm text-muted-foreground">Label: {result.label}</p>
                      </div>
                      <span className={cn('score-badge', getScoreClass(result.score))}>
                        {formatScore(result.score)}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 rounded-lg bg-muted p-3">
                  <p className="text-sm">
                    <strong>Analysis:</strong> The model returned{' '}
                    <span className={selectedExample.is_semantic_match ? 'text-amber-600 dark:text-amber-400' : 'text-destructive'}>
                      {selectedExample.retrieved_label}
                    </span>{' '}
                    instead of{' '}
                    <span className="text-green-600 dark:text-green-400">{selectedExample.expected_label}</span>.
                    {selectedExample.is_semantic_match && (
                      <> This is a <strong>medically reasonable</strong> retrieval — both labels are clinically related.</>
                    )}
                    {!selectedExample.is_semantic_match && selectedExample.confusion_type !== 'unknown' && (
                      <> This is a known confusion pattern: {selectedExample.confusion_type}.</>
                    )}
                  </p>
                </div>
              </>
            ) : (
              <div className="flex h-full items-center justify-center text-muted-foreground">
                Select an example to see details
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

/** Stat card component. */
function StatCard({
  title,
  value,
  description,
  color = 'text-foreground',
}: {
  title: string;
  value: string;
  description: string;
  color?: string;
}) {
  return (
    <div className="rounded-lg border bg-card p-6">
      <p className="text-sm font-medium text-muted-foreground">{title}</p>
      <p className={cn('mt-2 text-3xl font-bold', color)}>{value}</p>
      <p className="mt-1 text-sm text-muted-foreground">{description}</p>
    </div>
  );
}

/**
 * Info tooltip that explains what Hard Cases are and why they matter.
 * Appears as an ℹ button; a popover shows on hover / click.
 */
function InfoTooltip() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((prev) => !prev)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-muted text-sm font-semibold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        aria-label="What are Hard Cases?"
      >
        i
      </button>

      {open && (
        <div className="absolute right-0 top-9 z-50 w-80 rounded-lg border bg-card p-4 shadow-lg text-sm leading-relaxed text-muted-foreground sm:w-96">
          <h4 className="mb-2 font-semibold text-foreground">What are Hard Cases?</h4>
          <p className="mb-2">
            Hard cases are queries where the model&apos;s <strong>top-1 retrieved image</strong> does not carry the
            expected label. They reveal the model&apos;s weaknesses and the boundaries of its understanding.
          </p>
          <h4 className="mb-1 font-semibold text-foreground">Why does this matter?</h4>
          <ul className="mb-2 list-inside list-disc space-y-1">
            <li>Shows which diagnoses are most challenging for the model to distinguish.</li>
            <li>
              Highlights <span className="text-amber-600 dark:text-amber-400">semantic matches</span> — cases where the
              &quot;wrong&quot; result is actually medically related (e.g., Edema ↔ Effusion), meaning the retrieval is
              still clinically useful.
            </li>
            <li>Helps identify areas where more training data or fine-tuning could improve performance.</li>
          </ul>
          <h4 className="mb-1 font-semibold text-foreground">How to read this page</h4>
          <p>
            <strong>Confusion Patterns</strong> show which label pairs are most commonly confused.{' '}
            <strong>Detailed Examples</strong> let you click into specific failures and see the exact ranked results,
            similarity scores, and whether the confusion is semantically reasonable.
          </p>
        </div>
      )}
    </div>
  );
}
