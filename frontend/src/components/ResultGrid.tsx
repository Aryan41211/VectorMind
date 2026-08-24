/**
 * Ranked retrieval results.
 *
 * The design goal is that the photographs carry the page. Chrome around
 * each result stays quiet: rank and score sit on the image rather than in
 * a caption bar, so the grid reads as a wall of pictures rather than a
 * table with thumbnails.
 *
 * Score is shown on every card deliberately. This model is honestly
 * mid-quality, and a visitor who can see that result #1 scored 0.94 and
 * result #8 scored 0.71 can calibrate what the ranking means — which is
 * more informative than hiding the number and letting the order imply
 * more confidence than exists.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { SearchResponse, SearchResult } from '../types/search';
import { CloseIcon, ImageIcon } from './Icon';

interface ResultGridProps {
  response: SearchResponse | null;
}

export function ResultGrid({ response }: ResultGridProps) {
  const [selected, setSelected] = useState<SearchResult | null>(null);

  if (!response) return null;

  const { results, query, search_type, total_results, latency_ms } = response;
  const direction =
    search_type === 'text_to_image' ? 'text → image' : 'image → text';

  return (
    <>
      <section className="w-full max-w-6xl mx-auto mt-10 animate-in">
        <header className="flex flex-wrap items-baseline justify-between gap-3 pb-4 mb-6 border-b border-subtle">
          <h2 className="text-sm text-secondary">
            <span className="text-primary font-medium">{total_results}</span>{' '}
            {total_results === 1 ? 'result' : 'results'}
            {query && (
              <>
                {' for '}
                <span className="text-primary font-medium">“{query}”</span>
              </>
            )}
          </h2>
          <div className="flex items-center gap-3 text-xs text-tertiary tabular">
            <span>{direction}</span>
            <span aria-hidden>·</span>
            <span>{latency_ms.toFixed(0)}ms</span>
          </div>
        </header>

        {results.length === 0 ? (
          <p className="text-center text-sm text-secondary py-16">
            Nothing matched closely enough. The corpus is 31,783 Flickr30k
            photographs — everyday scenes of people, animals, and streets.
          </p>
        ) : (
          <ul className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map((result) => (
              <li key={`${result.index}-${result.rank}`}>
                <ResultCard result={result} onSelect={() => setSelected(result)} />
              </li>
            ))}
          </ul>
        )}
      </section>

      {selected && (
        <ResultDialog result={selected} onClose={() => setSelected(null)} />
      )}
    </>
  );
}

function ResultCard({
  result,
  onSelect,
}: {
  result: SearchResult;
  onSelect: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const showImage = Boolean(result.image_url) && !failed;

  return (
    <button
      type="button"
      onClick={onSelect}
      className="group w-full text-left rounded-card overflow-hidden border border-subtle transition-all hover:shadow-lifted focus-visible:shadow-lifted"
      style={{ background: 'var(--surface-raised)' }}
    >
      <div className="relative aspect-square surface-sunken overflow-hidden">
        {showImage ? (
          <img
            src={result.image_url}
            alt={result.caption ?? `Search result ${result.rank}`}
            loading="lazy"
            decoding="async"
            onError={() => setFailed(true)}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <ImageIcon className="w-8 h-8 text-tertiary" />
          </div>
        )}

        {/* Rank and score float over the image so the card is pure picture
            until you look for the numbers. */}
        <span
          className="absolute top-2 left-2 text-[11px] font-medium tabular px-1.5 py-0.5 rounded backdrop-blur-sm"
          style={{ background: 'oklch(0 0 0 / 0.55)', color: 'oklch(1 0 0)' }}
        >
          #{result.rank}
        </span>
        <span
          className="absolute top-2 right-2 text-[11px] font-mono tabular px-1.5 py-0.5 rounded backdrop-blur-sm"
          style={{ background: 'oklch(0 0 0 / 0.55)', color: 'oklch(1 0 0)' }}
        >
          {result.score.toFixed(3)}
        </span>

        {result.caption && (
          <p
            className="absolute inset-x-0 bottom-0 p-2.5 text-[11px] leading-snug line-clamp-3 opacity-0 translate-y-1 transition-all duration-200 group-hover:opacity-100 group-hover:translate-y-0 group-focus-visible:opacity-100"
            style={{
              background:
                'linear-gradient(to top, oklch(0 0 0 / 0.85), oklch(0 0 0 / 0))',
              color: 'oklch(1 0 0)',
            }}
          >
            {result.caption}
          </p>
        )}
      </div>
    </button>
  );
}

function ResultDialog({
  result,
  onClose,
}: {
  result: SearchResult;
  onClose: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  // Move focus into the dialog and restore it on close, so keyboard users
  // are not dropped at the top of the document.
  useEffect(() => {
    previouslyFocused.current = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => previouslyFocused.current?.focus();
  }, []);

  // Escape closes; the body must not scroll behind the overlay.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [onClose]);

  const stop = useCallback(
    (event: React.MouseEvent) => event.stopPropagation(),
    []
  );

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Result ${result.rank} detail`}
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in"
      style={{ background: 'oklch(0 0 0 / 0.72)', backdropFilter: 'blur(4px)' }}
    >
      <div
        onClick={stop}
        className="surface border border-subtle rounded-card shadow-lifted w-full max-w-3xl max-h-[90vh] overflow-y-auto"
      >
        <header className="sticky top-0 z-10 flex items-center justify-between gap-4 px-5 py-3 border-b border-subtle surface">
          <div className="flex items-center gap-3 text-sm tabular">
            <span className="font-medium text-primary">Rank {result.rank}</span>
            <span className="text-tertiary" aria-hidden>·</span>
            <span className="font-mono text-accent">
              {result.score.toFixed(4)}
            </span>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close detail view"
            className="p-1.5 rounded-lg text-secondary transition-colors hover:text-primary"
          >
            <CloseIcon />
          </button>
        </header>

        {result.image_url && !failed ? (
          <img
            src={result.image_url}
            alt={result.caption ?? `Result ${result.rank}`}
            onError={() => setFailed(true)}
            className="w-full max-h-[55vh] object-contain surface-sunken"
          />
        ) : (
          <div className="h-56 flex items-center justify-center surface-sunken">
            <ImageIcon className="w-10 h-10 text-tertiary" />
          </div>
        )}

        <dl className="px-5 py-4 grid gap-3 text-sm">
          {result.caption && (
            <div>
              <dt className="text-xs uppercase tracking-wide text-tertiary mb-1">
                Caption
              </dt>
              <dd className="text-primary leading-relaxed">{result.caption}</dd>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            {result.filename && (
              <div>
                <dt className="text-xs uppercase tracking-wide text-tertiary mb-1">
                  File
                </dt>
                <dd className="font-mono text-xs text-secondary break-all">
                  {result.filename}
                </dd>
              </div>
            )}
            <div>
              <dt className="text-xs uppercase tracking-wide text-tertiary mb-1">
                Index position
              </dt>
              <dd className="font-mono text-xs text-secondary tabular">
                {result.index}
              </dd>
            </div>
          </div>
        </dl>
      </div>
    </div>
  );
}
