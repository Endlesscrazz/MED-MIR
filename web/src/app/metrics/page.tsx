'use client';

import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { loadMetrics } from '@/lib/binary-loader';
import type { Metrics } from '@/lib/types';
import { formatPercentage } from '@/lib/utils';

/** Which metric view the user is looking at */
type MetricView = 'strict' | 'adjusted' | 'semantic';

/**
 * Metrics dashboard page.
 *
 * Displays model performance metrics including:
 * - Three tiers: Strict, Adjusted (fair), and Semantic (soft) metrics
 * - Per-label performance (annotated with missing-label warnings)
 * - Dataset distribution
 * - Methodology explanation
 */
export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [view, setView] = useState<MetricView>('adjusted');

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const data = await loadMetrics();
        setMetrics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load metrics');
      } finally {
        setIsLoading(false);
      }
    }
    fetchMetrics();
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

  if (error || !metrics) {
    return (
      <div className="container py-8">
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-6">
          <h2 className="mb-2 text-lg font-semibold text-destructive">Error Loading Metrics</h2>
          <p className="text-sm text-destructive/80">{error || 'No metrics data available'}</p>
          <p className="mt-4 text-sm text-muted-foreground">
            Run the Python evaluation pipeline to generate metrics.json
          </p>
        </div>
      </div>
    );
  }

  // Pick the active metric set based on the view selector
  const activeMetrics =
    view === 'adjusted' && metrics.adjusted
      ? metrics.adjusted
      : view === 'semantic' && metrics.semantic
        ? {
            recall_at_1: metrics.semantic.recall_at_1,
            recall_at_5: metrics.semantic.recall_at_5,
            recall_at_10: metrics.semantic.recall_at_10,
            recall_at_20: metrics.semantic.recall_at_20,
            mAP: metrics.overall.mAP, // semantic view doesn't compute its own mAP
            MRR: metrics.semantic.MRR,
          }
        : metrics.overall;

  const recallData = [
    { name: 'Recall@1', value: activeMetrics.recall_at_1 },
    { name: 'Recall@5', value: activeMetrics.recall_at_5 },
    { name: 'Recall@10', value: activeMetrics.recall_at_10 },
    { name: 'Recall@20', value: activeMetrics.recall_at_20 },
  ];

  const perLabelData = Object.entries(metrics.per_label)
    .map(([label, data]) => ({
      name: label,
      recall: data.recall_at_10,
      images: data.num_images,
      hasImages: data.has_images !== false && data.num_images > 0,
    }))
    .sort((a, b) => b.recall - a.recall || b.images - a.images);

  const distributionData = Object.entries(metrics.dataset.label_distribution)
    .map(([label, count]) => ({ name: label, value: count }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 12);

  const COLORS = [
    '#0077B6', '#00B4D8', '#90E0EF', '#03045E', '#023E8A',
    '#0096C7', '#48CAE4', '#ADE8F4', '#CAF0F8', '#012A4A',
    '#005F73', '#0A9396',
  ];

  const excludedLabels = metrics.adjusted?.excluded_labels ?? [];

  return (
    <div className="container py-8">
      <h1 className="mb-2 text-3xl font-bold">Reliability Metrics</h1>
      <p className="mb-8 text-muted-foreground">
        BiomedCLIP retrieval evaluation on {metrics.dataset.num_images} medical images
      </p>

      {/* ── View Selector ─────────────────────────────────────── */}
      <section className="mb-8">
        <div className="inline-flex rounded-lg border bg-muted p-1">
          {(['strict', 'adjusted', 'semantic'] as MetricView[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                view === v
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {v === 'strict' ? 'Strict' : v === 'adjusted' ? 'Adjusted (Fair)' : 'Semantic (Soft)'}
            </button>
          ))}
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          {view === 'strict' && (
            <>All {metrics.evaluation.num_queries} queries — includes labels with zero images in the dataset.</>
          )}
          {view === 'adjusted' && (
            <>
              {metrics.evaluation.num_viable_queries ?? '?'} viable queries only — excludes{' '}
              {excludedLabels.length > 0 ? excludedLabels.join(', ') : 'labels'} (0 images in dataset).
            </>
          )}
          {view === 'semantic' && (
            <>Counts medically-related retrievals as correct (e.g., Nodule → Opacity, Edema → Effusion).</>
          )}
        </p>
      </section>

      {/* ── Overall Metrics Cards ─────────────────────────────── */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Overall Performance</h2>
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
          <MetricCard title="Recall@1" value={formatPercentage(activeMetrics.recall_at_1)} description="Top result is relevant" />
          <MetricCard title="Recall@5" value={formatPercentage(activeMetrics.recall_at_5)} description="Relevant in top 5" />
          <MetricCard title="Recall@10" value={formatPercentage(activeMetrics.recall_at_10)} description="Relevant in top 10" />
          <MetricCard title="Recall@20" value={formatPercentage(activeMetrics.recall_at_20)} description="Relevant in top 20" />
          <MetricCard title="mAP" value={formatPercentage(activeMetrics.mAP)} description="Mean Avg Precision" />
          <MetricCard title="MRR" value={formatPercentage(activeMetrics.MRR)} description="Mean Reciprocal Rank" />
        </div>
      </section>

      {/* ── Side-by-Side Comparison ───────────────────────────── */}
      {metrics.adjusted && metrics.semantic && (
        <section className="mb-12">
          <h2 className="mb-4 text-xl font-semibold">Metric Comparison</h2>
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium">Metric</th>
                  <th className="px-4 py-3 text-right font-medium">Strict</th>
                  <th className="px-4 py-3 text-right font-medium">Adjusted</th>
                  <th className="px-4 py-3 text-right font-medium">Semantic</th>
                </tr>
              </thead>
              <tbody>
                {(['recall_at_1', 'recall_at_5', 'recall_at_10', 'recall_at_20'] as const).map((key) => {
                  const label = key.replace('recall_at_', 'Recall@');
                  return (
                    <tr key={key} className="border-b last:border-0">
                      <td className="px-4 py-2 font-medium">{label}</td>
                      <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.overall[key])}</td>
                      <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.adjusted![key])}</td>
                      <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.semantic![key])}</td>
                    </tr>
                  );
                })}
                <tr className="border-b last:border-0">
                  <td className="px-4 py-2 font-medium">MRR</td>
                  <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.overall.MRR)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.adjusted.MRR)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatPercentage(metrics.semantic.MRR)}</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-medium">Queries</td>
                  <td className="px-4 py-2 text-right font-mono">{metrics.evaluation.num_queries}</td>
                  <td className="px-4 py-2 text-right font-mono">{metrics.adjusted.num_viable_queries}</td>
                  <td className="px-4 py-2 text-right font-mono">{metrics.evaluation.num_queries}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ── Recall Chart ──────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Recall by K</h2>
        <div className="rounded-lg border bg-card p-4">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={recallData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" className="text-sm" />
              <YAxis domain={[0, 1]} tickFormatter={(v: number) => formatPercentage(v)} className="text-sm" />
              <Tooltip
                formatter={(value: number) => formatPercentage(value)}
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                }}
              />
              <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ── Per-Label Performance ─────────────────────────────── */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Performance by Label</h2>
        {excludedLabels.length > 0 && (
          <p className="mb-3 rounded-md border border-amber-500/30 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
            ⚠ Labels with 0 images in dataset ({excludedLabels.join(', ')}) will always show 0% — these are excluded from the Adjusted metrics.
          </p>
        )}
        <div className="rounded-lg border bg-card p-4">
          <ResponsiveContainer width="100%" height={Math.max(400, perLabelData.length * 40)}>
            <BarChart data={perLabelData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => formatPercentage(v)} />
              <YAxis type="category" dataKey="name" width={120} tick={({ x, y, payload }: { x: number; y: number; payload: { value: string } }) => {
                const item = perLabelData.find((d) => d.name === payload.value);
                const missing = item && !item.hasImages;
                return (
                  <text x={x} y={y} dy={4} textAnchor="end" fontSize={12} fill={missing ? '#e11d48' : 'currentColor'}>
                    {payload.value}{missing ? ' ⊘' : ''} ({item?.images ?? 0})
                  </text>
                );
              }} />
              <Tooltip
                formatter={(value: number, name: string) => [
                  name === 'recall' ? formatPercentage(value) : String(value),
                  name === 'recall' ? 'Recall@10' : 'Images',
                ]}
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                }}
              />
              <Bar dataKey="recall" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ── Dataset Distribution ──────────────────────────────── */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Dataset Distribution</h2>
        <div className="grid gap-8 lg:grid-cols-3">
          <div className="rounded-lg border bg-card p-4 lg:col-span-2">
            <h3 className="mb-3 text-sm font-medium text-muted-foreground">Images per Label (multi-label counts)</h3>
            <ResponsiveContainer width="100%" height={Math.max(350, distributionData.length * 32)}>
              <BarChart data={distributionData} layout="vertical" margin={{ left: 10, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" horizontal={false} />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 13 }} />
                <Tooltip
                  formatter={(value: number) => [`${value} images`, 'Count']}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {distributionData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-4 font-semibold">Dataset Statistics</h3>
            <div className="space-y-3">
              <StatRow label="Total Images" value={String(metrics.dataset.num_images)} />
              <StatRow label="Unique Labels" value={String(metrics.dataset.num_labels)} />
              <StatRow label="Evaluation Queries" value={String(metrics.evaluation.num_queries)} />
              {metrics.evaluation.num_viable_queries != null && (
                <StatRow label="Viable Queries" value={String(metrics.evaluation.num_viable_queries)} />
              )}
              {metrics.evaluation.num_impossible_queries != null && metrics.evaluation.num_impossible_queries > 0 && (
                <StatRow label="Impossible Queries" value={`${metrics.evaluation.num_impossible_queries} (label not in dataset)`} />
              )}
              <StatRow label="K Values Tested" value={metrics.evaluation.k_values.join(', ')} />
            </div>
          </div>
        </div>
      </section>

      {/* ── Methodology Explanation ───────────────────────────── */}
      <section className="mb-12">
        <h2 className="mb-4 text-xl font-semibold">Methodology</h2>
        <div className="space-y-5 rounded-lg border bg-card p-6 text-sm leading-relaxed text-muted-foreground">
          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">What is this measuring?</h3>
            <p>
              This page evaluates how accurately the AI model retrieves relevant medical images when given a text description.
              For example, if you search &quot;pneumonia chest X-ray&quot;, do the top results actually contain images
              diagnosed with pneumonia?
            </p>
          </div>

          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">Model</h3>
            <p>
              <strong>BiomedCLIP</strong> (PubMedBERT + ViT-B/16) — a state-of-the-art biomedical vision-language model
              pre-trained on 15 million biomedical image-text pairs from PubMed Central. It understands both medical images
              and clinical text in a shared embedding space, enabling cross-modal retrieval.
            </p>
          </div>

          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">How We Evaluate</h3>
            <p>
              We generate text embeddings for <strong>{metrics.evaluation.num_queries} clinical queries</strong> across{' '}
              <strong>{Object.keys(metrics.per_label).length} diagnostic categories</strong>, then rank all{' '}
              {metrics.dataset.num_images} images by cosine similarity. A retrieval is &quot;correct&quot; if the top-K
              results contain at least one image whose label matches the query.
            </p>
          </div>

          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">Understanding the Three Metric Tiers</h3>
            <div className="mt-2 space-y-3">
              <div className="rounded-md border-l-4 border-blue-500 bg-blue-50 p-3 dark:bg-blue-900/10">
                <p className="font-semibold text-foreground">Strict</p>
                <p>Exact label match on all queries. The harshest evaluation — does the model return exactly the right diagnosis?</p>
              </div>
              <div className="rounded-md border-l-4 border-emerald-500 bg-emerald-50 p-3 dark:bg-emerald-900/10">
                <p className="font-semibold text-foreground">Adjusted (Fair)</p>
                <p>
                  Same as Strict, but excludes queries whose target label has zero images in the dataset. This is the fairest
                  comparison because the model cannot succeed on labels it has never seen.
                </p>
              </div>
              <div className="rounded-md border-l-4 border-amber-500 bg-amber-50 p-3 dark:bg-amber-900/10">
                <p className="font-semibold text-foreground">Semantic (Soft)</p>
                <p>
                  Counts medically-related retrievals as correct. For instance, returning &quot;Effusion&quot; for a query
                  about &quot;Edema&quot; is counted as a hit because both involve fluid accumulation in or around the lungs.
                  This reflects real clinical utility — a doctor would still find such results useful.
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">Key Metrics Explained</h3>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <tbody>
                  <tr className="border-b">
                    <td className="py-2 pr-4 font-medium text-foreground">Recall@K</td>
                    <td className="py-2">Of all queries, what fraction found at least one correct image in the top K results?</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-2 pr-4 font-medium text-foreground">mAP</td>
                    <td className="py-2">Mean Average Precision — rewards placing correct results higher in the ranking.</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-medium text-foreground">MRR</td>
                    <td className="py-2">Mean Reciprocal Rank — how high is the first correct result on average? (1.0 = always first)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h3 className="mb-1 text-base font-semibold text-foreground">Dataset &amp; Labels</h3>
            <p>
              Ground-truth labels come from the <strong>NIH ChestX-ray14</strong> dataset — a widely-used, peer-reviewed
              benchmark of 112,120 frontal-view chest X-rays with 14 disease labels and &quot;No Finding&quot;, extracted
              from radiology reports using NLP by the NIH Clinical Center. All labels are <strong>dataset-verified</strong>,
              meaning they originate directly from the published dataset rather than manual annotation.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

/** Metric card component. */
function MetricCard({ title, value, description }: { title: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <p className="text-sm font-medium text-muted-foreground">{title}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

/** Simple stat row. */
function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b pb-2 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}
