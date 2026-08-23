/**
 * Text query input with example suggestions.
 *
 * The examples are not decoration. A visitor has no way to know that the
 * corpus is 3,179 Flickr30k photographs, so an unprompted query is as
 * likely to be "quarterly revenue chart" as anything the model can
 * answer — and an empty result set reads as a broken demo rather than an
 * out-of-domain query.
 */

import { useEffect, useRef, useState, type FormEvent } from 'react';
import { SearchIcon } from './Icon';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

/**
 * Chosen to span what the model actually does well and badly, per the
 * Phase 5 qualitative analysis: scene-level concepts it handles, and
 * fine-grained actions it does not.
 */
const EXAMPLE_QUERIES = [
  'a dog playing in a park',
  'a person riding a bike',
  'children playing in a playground',
  'a busy city street with cars',
  'a chef cooking in a kitchen',
];

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // "/" focuses search, the convention on every site with a search box.
  // Guarded so it does not steal the key while someone is typing.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.metaKey || event.ctrlKey) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || target?.isContentEditable) {
        return;
      }
      event.preventDefault();
      inputRef.current?.focus();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const submit = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSearch(trimmed);
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit(query);
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    submit(example);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit}>
        <label htmlFor="search-query" className="sr-only">
          Describe an image to search for
        </label>
        <div
          className="flex items-center gap-3 surface border border-subtle rounded-xl px-4 h-14 shadow-card transition-shadow focus-within:shadow-lifted"
          style={{ borderColor: 'var(--border-subtle)' }}
        >
          <SearchIcon className="w-5 h-5 shrink-0 text-tertiary" />
          <input
            id="search-query"
            ref={inputRef}
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={isLoading}
            placeholder="Describe an image…"
            autoComplete="off"
            spellCheck={false}
            className="flex-1 bg-transparent text-[15px] text-primary placeholder:text-tertiary outline-none disabled:opacity-50"
          />
          <kbd
            aria-hidden
            className="hidden sm:block text-[11px] font-mono px-1.5 py-0.5 rounded border border-subtle text-tertiary"
          >
            /
          </kbd>
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="shrink-0 px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: 'var(--accent)',
              color: 'var(--accent-contrast)',
            }}
          >
            {isLoading ? 'Searching…' : 'Search'}
          </button>
        </div>
      </form>

      <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
        <span className="text-xs text-tertiary mr-1">Try</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => handleExampleClick(example)}
            disabled={isLoading}
            className="text-xs px-3 py-1.5 rounded-full border border-subtle text-secondary transition-colors disabled:opacity-40 hover:text-primary"
            style={{ background: 'var(--surface-raised)' }}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
