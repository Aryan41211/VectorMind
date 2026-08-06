/**
 * VectorMind App — Main application component
 */

import { useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { ImageUploader } from './components/ImageUploader';
import { ResultGrid } from './components/ResultGrid';
import { searchText, searchImage } from './api/client';
import type { SearchResponse } from './types/search';

type SearchMode = 'text' | 'image';

function App() {
  const [mode, setMode] = useState<SearchMode>('text');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTextSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await searchText({ query, top_k: 10 });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageSearch = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await searchImage(file, 10);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">VectorMind</h1>
          <p className="mt-1 text-gray-600">
            Multimodal semantic search — find images by text, or captions by image
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Mode Toggle */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              onClick={() => setMode('text')}
              className={`px-6 py-2 text-sm font-medium transition-colors ${
                mode === 'text'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Text → Image
            </button>
            <button
              onClick={() => setMode('image')}
              className={`px-6 py-2 text-sm font-medium transition-colors ${
                mode === 'image'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Image → Caption
            </button>
          </div>
        </div>

        {/* Search Input */}
        {mode === 'text' ? (
          <SearchBar onSearch={handleTextSearch} isLoading={isLoading} />
        ) : (
          <ImageUploader onUpload={handleImageSearch} isLoading={isLoading} />
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 max-w-2xl mx-auto p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Results */}
        <ResultGrid response={result} />
      </main>

      <footer className="mt-auto py-4 text-center text-sm text-gray-500">
        VectorMind — Trained from scratch on Flickr30k
      </footer>
    </div>
  );
}

export default App;
