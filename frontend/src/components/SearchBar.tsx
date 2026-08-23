/**
 * SearchBar Component — Text query input with example queries
 */

import { useState, type FormEvent } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

const EXAMPLE_QUERIES = [
  'a dog playing in a park',
  'a person riding a bike',
  'children playing in a playground',
  'a busy city street with cars',
  'a chef cooking in a kitchen',
];

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim());
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    onSearch(example);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit}>
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for images... (e.g., 'a dog playing in the park')"
            disabled={isLoading}
            className="flex-1 px-4 py-3 text-lg border border-gray-300 rounded-lg
                       focus:outline-none focus:ring-2 focus:ring-blue-500
                       disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="px-6 py-3 text-lg font-medium text-white bg-blue-600
                       rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2
                       focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed
                       transition-colors"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        <span className="text-sm text-gray-500 self-center">Try:</span>
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => handleExampleClick(q)}
            disabled={isLoading}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700
                       rounded-full transition-colors disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
