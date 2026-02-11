'use client';

import { useState, useEffect, useCallback, FormEvent, KeyboardEvent } from 'react';
import { Search, Loader2, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SearchBoxProps {
  /** Callback when search is submitted */
  onSearch: (query: string) => void;
  /** Whether search is currently loading */
  isLoading?: boolean;
  /** Placeholder text */
  placeholder?: string;
  /** Initial value */
  initialValue?: string;
  /** Custom className */
  className?: string;
}

/**
 * Search input component with submit button.
 * 
 * Features:
 * - Enter key to submit
 * - Clear button when text present
 * - Loading state indicator
 */
export function SearchBox({
  onSearch,
  isLoading = false,
  placeholder = 'Search medical images...',
  initialValue = '',
  className,
}: SearchBoxProps) {
  const [query, setQuery] = useState(initialValue);
  
  // Sync with external value changes (e.g., clear search)
  useEffect(() => {
    setQuery(initialValue);
  }, [initialValue]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSearch(query.trim());
      }
    },
    [query, isLoading, onSearch]
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (query.trim() && !isLoading) {
          onSearch(query.trim());
        }
      }
    },
    [query, isLoading, onSearch]
  );

  const handleClear = useCallback(() => {
    setQuery('');
  }, []);

  return (
    <form onSubmit={handleSubmit} className={cn('w-full', className)}>
      <div className="relative flex items-center">
        {/* Search Icon */}
        <div className="pointer-events-none absolute left-4 flex items-center">
          <Search className="h-5 w-5 text-muted-foreground" />
        </div>

        {/* Input */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          className={cn(
            'h-14 w-full rounded-xl border bg-background pl-12 pr-24 text-base shadow-sm',
            'placeholder:text-muted-foreground',
            'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'transition-shadow'
          )}
        />

        {/* Clear Button */}
        {query && !isLoading && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-20 rounded-full p-1 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            aria-label="Clear search"
          >
            <X className="h-4 w-4" />
          </button>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className={cn(
            'absolute right-2 flex h-10 items-center gap-2 rounded-lg px-4',
            'bg-primary text-primary-foreground',
            'font-medium transition-colors',
            'hover:bg-primary/90',
            'disabled:cursor-not-allowed disabled:opacity-50'
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="hidden sm:inline">Searching</span>
            </>
          ) : (
            <>
              <Search className="h-4 w-4" />
              <span className="hidden sm:inline">Search</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
}
