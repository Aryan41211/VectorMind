/**
 * Tests for SearchBar.
 *
 * The behaviour worth pinning is the submit guard: an empty or
 * whitespace-only query must not reach the API, and neither must a second
 * submit while one is already in flight.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { SearchBar } from './SearchBar';

function setup(isLoading = false) {
  const onSearch = vi.fn();
  render(<SearchBar onSearch={onSearch} isLoading={isLoading} />);
  return { onSearch, user: userEvent.setup() };
}

describe('SearchBar', () => {
  it('submits the typed query', async () => {
    const { onSearch, user } = setup();
    await user.type(screen.getByRole('textbox'), 'a dog in a park');
    await user.keyboard('{Enter}');
    expect(onSearch).toHaveBeenCalledWith('a dog in a park');
  });

  it('trims surrounding whitespace before submitting', async () => {
    const { onSearch, user } = setup();
    await user.type(screen.getByRole('textbox'), '   a dog   ');
    await user.keyboard('{Enter}');
    expect(onSearch).toHaveBeenCalledWith('a dog');
  });

  it('does not submit an empty query', async () => {
    const { onSearch, user } = setup();
    await user.click(screen.getByRole('textbox'));
    await user.keyboard('{Enter}');
    expect(onSearch).not.toHaveBeenCalled();
  });

  it('does not submit a whitespace-only query', async () => {
    const { onSearch, user } = setup();
    await user.type(screen.getByRole('textbox'), '    ');
    await user.keyboard('{Enter}');
    expect(onSearch).not.toHaveBeenCalled();
  });

  it('does not submit while a search is already running', async () => {
    const { onSearch, user } = setup(true);
    const input = screen.getByRole('textbox');
    await user.type(input, 'a dog');
    await user.keyboard('{Enter}');
    expect(onSearch).not.toHaveBeenCalled();
  });

  it('offers example queries for visitors who do not know the dataset', () => {
    setup();
    expect(screen.getByText('a dog playing in a park')).toBeInTheDocument();
  });

  it('searches immediately when an example is clicked', async () => {
    const { onSearch, user } = setup();
    await user.click(screen.getByText('a person riding a bike'));
    expect(onSearch).toHaveBeenCalledWith('a person riding a bike');
  });
});
