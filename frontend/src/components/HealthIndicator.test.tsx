/**
 * Tests for HealthIndicator.
 *
 * The distinction that matters is "up but still loading" versus
 * "offline". They look identical to a visitor otherwise — both mean
 * searches fail — but only one is worth waiting through, and the demo is
 * most likely to be opened during exactly that window.
 */

import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HealthIndicator } from './HealthIndicator';
import * as client from '../api/client';
import type { HealthResponse } from '../types/search';

function health(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return {
    status: 'ok',
    model_loaded: true,
    index_loaded: true,
    device: 'cpu',
    num_indexed_images: 3179,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
});

describe('HealthIndicator', () => {
  it('reports the indexed corpus size once ready', async () => {
    vi.spyOn(client, 'getHealth').mockResolvedValue(health());
    render(<HealthIndicator />);
    await waitFor(() =>
      expect(screen.getByText('3,179 indexed')).toBeInTheDocument()
    );
  });

  it('distinguishes a loading backend from an offline one', async () => {
    vi.spyOn(client, 'getHealth').mockResolvedValue(
      health({ model_loaded: true, index_loaded: false })
    );
    render(<HealthIndicator />);
    await waitFor(() =>
      expect(screen.getByText('Starting up')).toBeInTheDocument()
    );
  });

  it('treats an unloaded model as starting, not ready', async () => {
    vi.spyOn(client, 'getHealth').mockResolvedValue(
      health({ model_loaded: false, index_loaded: true })
    );
    render(<HealthIndicator />);
    await waitFor(() =>
      expect(screen.getByText('Starting up')).toBeInTheDocument()
    );
  });

  it('reports offline when the API cannot be reached', async () => {
    vi.spyOn(client, 'getHealth').mockRejectedValue(
      new client.ApiError('offline')
    );
    render(<HealthIndicator />);
    await waitFor(() =>
      expect(screen.getByText('API offline')).toBeInTheDocument()
    );
  });

  it('announces status changes to assistive technology', async () => {
    vi.spyOn(client, 'getHealth').mockResolvedValue(health());
    const { container } = render(<HealthIndicator />);
    expect(container.querySelector('[aria-live="polite"]')).not.toBeNull();
  });

  it('stops polling after unmount', async () => {
    const spy = vi.spyOn(client, 'getHealth').mockResolvedValue(health());
    const { unmount } = render(<HealthIndicator />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    unmount();
    await vi.advanceTimersByTimeAsync(90_000);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
