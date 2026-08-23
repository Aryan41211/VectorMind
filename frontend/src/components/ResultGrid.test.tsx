/**
 * Tests for ResultGrid.
 *
 * The grid is where the index-duplication bug was visible to users: before
 * the image index was deduplicated, a single top-10 could render the same
 * photo five times. The key uniqueness test below is the frontend-side
 * guard for that class of failure.
 */

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ResultGrid } from './ResultGrid';
import type { SearchResponse, SearchResult } from '../types/search';

function result(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    rank: 1,
    index: 7,
    score: 0.84,
    filename: '022545.jpg',
    image_url: '/images/022545.jpg',
    caption: 'A crowd of people shopping at a street market.',
    ...overrides,
  };
}

function response(overrides: Partial<SearchResponse> = {}): SearchResponse {
  return {
    results: [result()],
    query: 'a busy market',
    search_type: 'text_to_image',
    total_results: 1,
    latency_ms: 14.23,
    ...overrides,
  };
}

describe('ResultGrid', () => {
  it('renders nothing before a search has run', () => {
    const { container } = render(<ResultGrid response={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows an empty state when a search returns no results', () => {
    render(<ResultGrid response={response({ results: [], total_results: 0 })} />);
    expect(screen.getByText('No results found')).toBeInTheDocument();
  });

  it('echoes the query back to the user', () => {
    render(<ResultGrid response={response()} />);
    expect(screen.getByText(/a busy market/)).toBeInTheDocument();
  });

  it('reports latency to one decimal place', () => {
    render(<ResultGrid response={response()} />);
    expect(screen.getByText(/14\.2ms/)).toBeInTheDocument();
  });

  it('renders one card per result', () => {
    const results = [
      result({ rank: 1, index: 1, filename: 'a.jpg', image_url: '/images/a.jpg' }),
      result({ rank: 2, index: 2, filename: 'b.jpg', image_url: '/images/b.jpg' }),
      result({ rank: 3, index: 3, filename: 'c.jpg', image_url: '/images/c.jpg' }),
    ];
    render(<ResultGrid response={response({ results, total_results: 3 })} />);
    expect(screen.getAllByRole('img')).toHaveLength(3);
  });

  it('gives every result a distinct React key', () => {
    // index+rank must be unique per row, or React silently drops cards.
    // Distinct images at distinct ranks is the post-deduplication shape.
    const results = Array.from({ length: 5 }, (_, i) =>
      result({
        rank: i + 1,
        index: i + 1,
        filename: `${i}.jpg`,
        image_url: `/images/${i}.jpg`,
      })
    );
    render(<ResultGrid response={response({ results, total_results: 5 })} />);
    const sources = screen.getAllByRole('img').map((img) => img.getAttribute('src'));
    expect(new Set(sources).size).toBe(5);
  });

  it('handles a result with no image_url without crashing', () => {
    const results = [result({ image_url: undefined, filename: undefined })];
    expect(() =>
      render(<ResultGrid response={response({ results })} />)
    ).not.toThrow();
  });

  it('shows the search direction', () => {
    render(
      <ResultGrid
        response={response({ search_type: 'image_to_text' })}
      />
    );
    expect(screen.getByText(/image_to_text/)).toBeInTheDocument();
  });
});
