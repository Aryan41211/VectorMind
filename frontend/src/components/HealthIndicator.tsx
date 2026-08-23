/**
 * Live API status, shown as a dot plus a short label.
 *
 * Worth the header space because the most likely failure a visitor meets
 * is not a bad result — it is a backend that is up but still loading its
 * checkpoint, which returns 503 and looks identical to "broken" without
 * this. The three states are deliberately distinct: offline, starting,
 * ready.
 */

import { useEffect, useState } from 'react';
import { getHealth } from '../api/client';
import type { HealthResponse } from '../types/search';

/** Slow enough to be invisible, fast enough to catch a cold start. */
const POLL_INTERVAL_MS = 30_000;

type Status = 'checking' | 'offline' | 'starting' | 'ready';

export function HealthIndicator() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<Status>('checking');

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const response = await getHealth();
        if (cancelled) return;
        setHealth(response);
        setStatus(
          response.model_loaded && response.index_loaded ? 'ready' : 'starting'
        );
      } catch {
        if (!cancelled) {
          setStatus('offline');
          setHealth(null);
        }
      }
    };

    check();
    const timer = setInterval(check, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const { color, label, detail } = describe(status, health);

  return (
    <div
      className="inline-flex items-center gap-2 text-xs"
      title={detail}
      aria-live="polite"
    >
      <span className="relative flex w-2 h-2" aria-hidden>
        {status === 'ready' && (
          <span
            className="absolute inline-flex w-full h-full rounded-full opacity-60 animate-ping"
            style={{ background: color }}
          />
        )}
        <span
          className="relative inline-flex w-2 h-2 rounded-full"
          style={{ background: color }}
        />
      </span>
      <span className="text-secondary">{label}</span>
    </div>
  );
}

function describe(
  status: Status,
  health: HealthResponse | null
): { color: string; label: string; detail: string } {
  switch (status) {
    case 'ready':
      return {
        color: 'var(--success)',
        label: `${(health?.num_indexed_images ?? 0).toLocaleString()} indexed`,
        detail: `Model and index loaded on ${health?.device ?? 'unknown device'}`,
      };
    case 'starting':
      return {
        color: 'var(--warning)',
        label: 'Starting up',
        detail:
          'The API is reachable but its model or index is not loaded yet. Searches will return 503 until it is.',
      };
    case 'offline':
      return {
        color: 'var(--danger)',
        label: 'API offline',
        detail: 'No response from the API. Is the backend running on port 8000?',
      };
    default:
      return {
        color: 'var(--text-tertiary)',
        label: 'Checking…',
        detail: 'Contacting the API',
      };
  }
}
