/**
 * ResultGrid Component — Ranked retrieval results with real images
 */

import { useState } from 'react';
import type { SearchResult, SearchResponse } from '../types/search';

interface ResultGridProps {
  response: SearchResponse | null;
}

export function ResultGrid({ response }: ResultGridProps) {
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  if (!response) return null;

  return (
    <>
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
            {response.results.map((result) => (
              <ResultCard
                key={`${result.index}-${result.rank}`}
                result={result}
                onClick={() => setSelectedResult(result)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Modal */}
      {selectedResult && (
        <ImageModal
          result={selectedResult}
          onClose={() => setSelectedResult(null)}
        />
      )}
    </>
  );
}

function ResultCard({
  result,
  onClick,
}: {
  result: SearchResult;
  onClick: () => void;
}) {
  const [imgError, setImgError] = useState(false);

  return (
    <div
      className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
      onClick={onClick}
    >
      {/* Image */}
      {result.image_url && !imgError ? (
        <div className="aspect-square bg-gray-100">
          <img
            src={result.image_url}
            alt={result.caption || `Result ${result.rank}`}
            className="w-full h-full object-cover"
            loading="lazy"
            onError={() => setImgError(true)}
          />
        </div>
      ) : (
        <div className="aspect-square bg-gray-100 flex items-center justify-center">
          <svg className="w-12 h-12 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
      )}

      {/* Info */}
      <div className="p-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium text-gray-500">#{result.rank}</span>
          <span className="text-xs font-mono text-blue-600">
            {result.score.toFixed(3)}
          </span>
        </div>

        {result.filename && (
          <div className="text-xs text-gray-500 truncate mb-1">
            {result.filename}
          </div>
        )}

        {result.caption && (
          <div className="text-sm text-gray-700 line-clamp-2">
            {result.caption}
          </div>
        )}
      </div>
    </div>
  );
}

function ImageModal({
  result,
  onClose,
}: {
  result: SearchResult;
  onClose: () => void;
}) {
  const [imgError, setImgError] = useState(false);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl max-w-3xl w-full mx-4 max-h-[90vh] overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-500">
              Rank #{result.rank}
            </span>
            <span className="text-sm font-mono text-blue-600">
              Score: {result.score.toFixed(3)}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Image */}
        {result.image_url && !imgError ? (
          <div className="bg-gray-100">
            <img
              src={result.image_url}
              alt={result.caption || `Result ${result.rank}`}
              className="w-full max-h-[50vh] object-contain"
              onError={() => setImgError(true)}
            />
          </div>
        ) : (
          <div className="bg-gray-100 h-64 flex items-center justify-center">
            <svg className="w-16 h-16 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        )}

        {/* Details */}
        <div className="p-4 space-y-2">
          {result.filename && (
            <div className="text-sm">
              <span className="font-medium text-gray-700">Filename:</span>{' '}
              <span className="text-gray-600">{result.filename}</span>
            </div>
          )}
          <div className="text-sm">
            <span className="font-medium text-gray-700">Dataset Index:</span>{' '}
            <span className="text-gray-600">{result.index}</span>
          </div>
          {result.caption && (
            <div className="text-sm">
              <span className="font-medium text-gray-700">Caption:</span>{' '}
              <span className="text-gray-600">{result.caption}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
