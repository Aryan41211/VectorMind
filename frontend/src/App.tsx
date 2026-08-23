/**
 * VectorMind — application shell and search orchestration.
 *
 * Layout is single-column and centred: this app does one thing, and a
 * sidebar or dashboard chrome would imply otherwise. The search control
 * is the page until results exist, then moves up to make room for them.
 */

import { useCallback, useRef, useState } from 'react';
import { ApiError, searchImage, searchText } from './api/client';
import { AboutPanel } from './components/AboutPanel';
import { HealthIndicator } from './components/HealthIndicator';
import { AlertIcon, ImageIcon, SearchIcon } from './components/Icon';
import { ImageUploader } from './components/ImageUploader';
import { ResultGrid } from './components/ResultGrid';
import { ResultSkeleton } from './components/ResultSkeleton';
import { SearchBar } from './components/SearchBar';
import { ThemeToggle } from './components/ThemeToggle';
import { useTheme } from './hooks/useTheme';
import type { SearchResponse } from './types/search';

type Mode = 'text' | 'image';

const TOP_K = 12;

export default function App() {
  const { theme, setTheme } = useTheme();
  const [mode, setMode] = useState<Mode>('text');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Guards against a slow first request resolving after a faster second
  // one and overwriting newer results with stale ones.
  const requestId = useRef(0);

  const run = useCallback(async (search: () => Promise<SearchResponse>) => {
    const id = ++requestId.current;
    setIsLoading(true);
    setError(null);
    try {
      const response = await search();
      if (id !== requestId.current) return;
      setResult(response);
    } catch (caught) {
      if (id !== requestId.current) return;
      setResult(null);
      setError(
        caught instanceof ApiError
          ? caught.userMessage
          : caught instanceof Error
            ? caught.message
            : 'Search failed'
      );
    } finally {
      if (id === requestId.current) setIsLoading(false);
    }
  }, []);

  const handleTextSearch = useCallback(
    (query: string) => run(() => searchText({ query, top_k: TOP_K })),
    [run]
  );

  const handleImageSearch = useCallback(
    (file: File) => run(() => searchImage(file, TOP_K)),
    [run]
  );

  const switchMode = (next: Mode) => {
    if (next === mode) return;
    requestId.current++; // discard anything in flight
    setMode(next);
    setResult(null);
    setError(null);
    setIsLoading(false);
  };

  const isIdle = !isLoading && !result && !error;

  return (
    <div className="min-h-screen flex flex-col">
      <header
        className="sticky top-0 z-40 border-b border-subtle backdrop-blur-md"
        style={{ background: 'color-mix(in oklch, var(--surface) 85%, transparent)' }}
      >
        <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between gap-4">
          <div className="flex items-baseline gap-2.5 min-w-0">
            <span className="text-[15px] font-semibold tracking-tight text-primary">
              VectorMind
            </span>
            <span className="hidden sm:inline text-xs text-tertiary truncate">
              multimodal search, trained from scratch
            </span>
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <HealthIndicator />
            <ThemeToggle theme={theme} onChange={setTheme} />
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-6xl mx-auto px-5 pb-20">
        <div className={isIdle ? 'pt-20 md:pt-28' : 'pt-10'}>
          {isIdle && (
            <div className="text-center mb-10 animate-in">
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-primary">
                Search photographs by meaning
              </h1>
              <p className="mt-3 text-sm text-secondary max-w-lg mx-auto leading-relaxed">
                Describe a scene and get the closest photographs, or upload an
                image and get the closest captions — through one shared
                embedding space.
              </p>
            </div>
          )}

          <div className="flex justify-center mb-6">
            <ModeToggle mode={mode} onChange={switchMode} />
          </div>

          {mode === 'text' ? (
            <SearchBar onSearch={handleTextSearch} isLoading={isLoading} />
          ) : (
            <ImageUploader onUpload={handleImageSearch} isLoading={isLoading} />
          )}

          {error && (
            <div
              role="alert"
              className="mt-8 max-w-2xl mx-auto flex items-start gap-3 p-4 rounded-[--radius-card] border animate-in"
              style={{
                borderColor: 'var(--border-subtle)',
                background: 'var(--surface-raised)',
              }}
            >
              <AlertIcon
                className="w-5 h-5 shrink-0 mt-px"
                style={{ color: 'var(--danger)' }}
              />
              <div className="min-w-0">
                <p className="text-sm font-medium text-primary">
                  Search failed
                </p>
                <p className="text-sm text-secondary mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {isLoading && <ResultSkeleton />}
          {!isLoading && <ResultGrid response={result} />}
          {isIdle && <AboutPanel />}
        </div>
      </main>

      <footer className="border-t border-subtle">
        <div className="max-w-6xl mx-auto px-5 py-5 flex flex-wrap items-center justify-between gap-3 text-xs text-tertiary">
          <span>
            Dual encoder trained from scratch on Flickr30k · 24M parameters ·
            RTX 4050
          </span>
          <a
            href="https://github.com/Aryan41211/VectorMind"
            target="_blank"
            rel="noreferrer noopener"
            className="text-secondary transition-colors hover:text-primary"
          >
            Source and write-up →
          </a>
        </div>
      </footer>
    </div>
  );
}

function ModeToggle({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  const options: { value: Mode; label: string; Icon: typeof SearchIcon }[] = [
    { value: 'text', label: 'Text', Icon: SearchIcon },
    { value: 'image', label: 'Image', Icon: ImageIcon },
  ];

  return (
    <div
      role="radiogroup"
      aria-label="Search mode"
      className="inline-flex items-center gap-0.5 p-0.5 rounded-lg border border-subtle"
      style={{ background: 'var(--surface-sunken)' }}
    >
      {options.map(({ value, label, Icon }) => {
        const active = mode === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(value)}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors"
            style={{
              background: active ? 'var(--surface-raised)' : 'transparent',
              color: active ? 'var(--text-primary)' : 'var(--text-tertiary)',
              boxShadow: active ? 'var(--shadow-card)' : 'none',
            }}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        );
      })}
    </div>
  );
}
