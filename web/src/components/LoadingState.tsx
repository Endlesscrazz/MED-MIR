'use client';

import { Progress } from '@/components/ui/progress';
import type { ModelState } from '@/lib/types';
import { Loader2, CheckCircle, AlertCircle, Download, Brain, Database } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingStep {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'pending' | 'loading' | 'complete' | 'error';
  progress?: number;
  errorMessage?: string;
}

interface LoadingStateProps {
  /** Model loading state */
  modelState: ModelState;
  /** Index loading state */
  indexState: {
    isLoaded: boolean;
    isLoading: boolean;
    progress: number;
    error: string | null;
  };
}

/**
 * Full-screen loading state shown while initializing.
 * 
 * Shows progress for:
 * - Index data loading (embeddings, metadata)
 * - ONNX model loading
 */
export function LoadingState({ modelState, indexState }: LoadingStateProps) {
  const steps: LoadingStep[] = [
    {
      id: 'index',
      label: 'Loading search index',
      icon: Database,
      status: indexState.error
        ? 'error'
        : indexState.isLoaded
        ? 'complete'
        : indexState.isLoading
        ? 'loading'
        : 'pending',
      progress: indexState.progress * 100,
      errorMessage: indexState.error || undefined,
    },
    {
      id: 'model',
      label: 'Loading AI model',
      icon: Brain,
      status: modelState.error
        ? 'error'
        : modelState.isLoaded
        ? 'complete'
        : modelState.isLoading
        ? 'loading'
        : 'pending',
      progress: modelState.progress * 100,
      errorMessage: modelState.error || undefined,
    },
  ];

  const overallProgress =
    steps.reduce((acc, step) => acc + (step.progress || 0), 0) / steps.length;

  return (
    <div className="mx-auto max-w-md py-12">
      <div className="rounded-xl border bg-card p-6 shadow-lg">
        {/* Header */}
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
            <Download className="h-8 w-8 animate-pulse text-primary" />
          </div>
          <h2 className="text-xl font-semibold">Initializing Med-MIR</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Preparing client-side AI for privacy-preserving search
          </p>
        </div>

        {/* Overall Progress */}
        <div className="mb-6">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Overall Progress</span>
            <span className="font-medium">{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>

        {/* Steps */}
        <div className="space-y-4">
          {steps.map((step) => (
            <LoadingStepItem key={step.id} step={step} />
          ))}
        </div>

        {/* Footer Note */}
        <div className="mt-6 rounded-lg bg-muted p-3 text-center text-xs text-muted-foreground">
          <p>
            First load may take 30-60 seconds to download the AI model.
            <br />
            Subsequent visits will be faster (model is cached).
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Individual loading step item.
 */
function LoadingStepItem({ step }: { step: LoadingStep }) {
  const Icon = step.icon;

  return (
    <div className="flex items-start gap-3">
      {/* Status Icon */}
      <div
        className={cn(
          'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full',
          step.status === 'complete' && 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400',
          step.status === 'loading' && 'bg-primary/10 text-primary',
          step.status === 'error' && 'bg-destructive/10 text-destructive',
          step.status === 'pending' && 'bg-muted text-muted-foreground'
        )}
      >
        {step.status === 'complete' ? (
          <CheckCircle className="h-5 w-5" />
        ) : step.status === 'loading' ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : step.status === 'error' ? (
          <AlertCircle className="h-5 w-5" />
        ) : (
          <Icon className="h-5 w-5" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <span
            className={cn(
              'text-sm font-medium',
              step.status === 'pending' && 'text-muted-foreground'
            )}
          >
            {step.label}
          </span>
          {step.status === 'loading' && step.progress !== undefined && (
            <span className="text-xs text-muted-foreground">
              {Math.round(step.progress)}%
            </span>
          )}
        </div>

        {/* Progress bar for loading state */}
        {step.status === 'loading' && step.progress !== undefined && (
          <Progress value={step.progress} className="mt-2 h-1" />
        )}

        {/* Error message */}
        {step.status === 'error' && step.errorMessage && (
          <p className="mt-1 text-xs text-destructive">{step.errorMessage}</p>
        )}
      </div>
    </div>
  );
}
