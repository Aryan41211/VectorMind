/**
 * Tests for ResultGrid.
 *
 * The grid is where the index-duplication bug was visible to users: before
 * the image index was deduplicated, a single top-10 could render the same
 * photo five times. The key uniqueness test below is the frontend-side
 * guard for that class of failure.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

  it('shows an empty state that explains the corpus', () => {
    // "No results" alone reads as a broken demo. Naming the corpus tells
    // the visitor their query was simply out of domain.
    render(<ResultGrid response={response({ results: [], total_results: 0 })} />);
    expect(screen.getByText(/Flickr30k photographs/i)).toBeInTheDocument();
  });

  it('echoes the query back to the user', () => {
    render(<ResultGrid response={response()} />);
    expect(screen.getByText(/a busy market/)).toBeInTheDocument();
  });

  it('reports latency in whole milliseconds', () => {
    render(<ResultGrid response={response()} />);
    expect(screen.getByText('14ms')).toBeInTheDocument();
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

  it('names the search direction in words, not the wire value', () => {
    render(<ResultGrid response={response({ search_type: 'image_to_text' })} />);
    expect(screen.getByText('image → text')).toBeInTheDocument();
    expect(screen.queryByText(/image_to_text/)).not.toBeInTheDocument();
  });

  it('pluralises the result count', () => {
    // The count is split across elements for styling, so assert on the
    // heading's text rather than trying to match a single text node.
    const { rerender } = render(<ResultGrid response={response()} />);
    expect(screen.getByRole('heading').textContent).toMatch(/1 result for/);

    const many = [result({ rank: 1, index: 1 }), result({ rank: 2, index: 2 })];
    rerender(<ResultGrid response={response({ results: many, total_results: 2 })} />);
    expect(screen.getByRole('heading').textContent).toMatch(/2 results/);
  });

  it('opens a detail dialog when a result is activated', async () => {
    const user = userEvent.setup();
    render(<ResultGrid response={response()} />);
    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('closes the detail dialog on Escape', async () => {
    const user = userEvent.setup();
    render(<ResultGrid response={response()} />);
    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});
