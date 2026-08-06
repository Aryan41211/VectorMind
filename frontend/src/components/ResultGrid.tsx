/**
 * ResultGrid Component — Ranked retrieval results
 */

import type { SearchResult, SearchResponse } from '../types/search';

interface ResultGridProps {
  response: SearchResponse | null;
}

export function ResultGrid({ response }: ResultGridProps) {
  if (!response) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mt-8">
      <div className="mb-4 text-sm text-gray-600">
        <span className="font-medium">Query:</span> {response.query}
        <span className="mx-2">|</span>
        <span className="font-medium">Type:</span> {response.search_type}
        <span className="mx-2">|</span>
        <span className="font-medium">Results:</span> {response.total_results}
        <span className="mx-2">|</span>
        <span className="font-medium">Latency:</span> {response.latency_ms.toFixed(1)}ms
      </div>

      {response.results.length === 0 ? (
        <div className="text-center text-gray-500 py-8">
          No results found
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {response.results.map((result, idx) => (
            <ResultCard key={`${result.index}-${idx}`} result={result} rank={idx + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs font-medium text-gray-500">#{rank}</span>
        <span className="text-xs font-mono text-blue-600">
          {result.score.toFixed(3)}
        </span>
      </div>

      {result.image_path && (
        <div className="mb-2 text-sm text-gray-600 truncate">
          <span className="font-medium">Image:</span> {result.image_path}
        </div>
      )}

      {result.caption && (
        <div className="text-sm text-gray-700">
          <span className="font-medium">Caption:</span> {result.caption}
        </div>
      )}

      <div className="mt-2 text-xs text-gray-400">
        Index: {result.index}
      </div>
    </div>
  );
}
