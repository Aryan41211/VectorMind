/**
 * VectorMind App — Main application component
 */

import { useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { ImageUploader } from './components/ImageUploader';
import { ResultGrid } from './components/ResultGrid';
import { HealthIndicator } from './components/HealthIndicator';
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
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">VectorMind</h1>
              <p className="mt-1 text-gray-600">
                Multimodal semantic search — find images by text, or captions by image
              </p>
            </div>
            <HealthIndicator />
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto px-4 py-8 w-full">
        {/* Mode Toggle */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              onClick={() => { setMode('text'); setResult(null); setError(null); }}
              className={`px-6 py-2 text-sm font-medium transition-colors ${
                mode === 'text'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Text → Image
            </button>
            <button
              onClick={() => { setMode('image'); setResult(null); setError(null); }}
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
          <div className="mt-6 max-w-2xl mx-auto p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2 text-red-700">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">Search Error:</span> {error}
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-3 text-gray-600">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Searching...</span>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !result && !error && (
          <div className="mt-12 text-center text-gray-500">
            <svg className="mx-auto h-16 w-16 mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <p className="text-lg">Enter a query to search</p>
            <p className="text-sm mt-1">Try the examples above, or type your own query</p>
          </div>
        )}

        {/* About This Demo */}
        {!isLoading && !result && !error && (
          <div className="mt-12 max-w-2xl mx-auto">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-3">About This Demo</h2>
              <p className="text-sm text-gray-600 mb-4">
                This is a research prototype of a dual-encoder multimodal model trained
                from scratch on Flickr30k (~31k images, 5 captions each). It learns a
                shared embedding space for images and text via contrastive learning
                (CLIP-style architecture), without any pretrained vision-language weights.
              </p>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-900">Val R@10:</span>{' '}
                  <span className="text-gray-600">20.23%</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900">Test R@10:</span>{' '}
                  <span className="text-gray-600">19.63%</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900">Image encoder:</span>{' '}
                  <span className="text-gray-600">ResNet-18-style CNN (from scratch)</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900">Text encoder:</span>{' '}
                  <span className="text-gray-600">Transformer (6-layer)</span>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-100">
                <h3 className="font-medium text-gray-900 mb-2">Known Failure Patterns</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li><span className="font-medium">Action ambiguity (~35%):</span> Queries with vague actions ("a person doing something") often retrieve unrelated images because the model can't resolve underspecified motion.</li>
                  <li><span className="font-medium">Object specificity (~25%):</span> Distinguishing similar objects ("a blue boat" vs "a red boat") is unreliable — the model struggles with fine-grained attribute binding.</li>
                  <li><span className="font-medium">Context vs content (~20%):</span> Abstract or scene-level concepts ("a safe neighborhood") are hard to separate from visual content — the model conflates appearance with meaning.</li>
                  <li><span className="font-medium">Compositional complexity (~15%):</span> Multi-object spatial relations ("a cat on a mat next to a dog") are only marginally better than chance.</li>
                </ul>
              </div>
              <p className="mt-4 text-xs text-gray-400">
                These limitations are expected for a ~12M parameter model trained on 31k
                image-text pairs from scratch. Larger datasets and pretrained backbones
                are needed to close the gap with production CLIP models.
              </p>
            </div>
          </div>
        )}

        {/* Results */}
        <ResultGrid response={result} />
      </main>

      <footer className="bg-white border-t border-gray-200 py-4">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          VectorMind — Trained from scratch on Flickr30k • Test R@10: 19.6% I2T / 15.1% T2I
        </div>
      </footer>
    </div>
  );
}

export default App;
