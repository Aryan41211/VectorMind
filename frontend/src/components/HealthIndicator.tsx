/**
 * HealthIndicator Component — Shows API connection status
 */

import { useEffect, useState } from 'react';
import { getHealth } from '../api/client';
import type { HealthResponse } from '../types/search';

export function HealthIndicator() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await getHealth();
        setHealth(response);
        setError(false);
      } catch {
        setError(true);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="inline-flex items-center gap-2 text-sm text-red-600">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        API offline
      </div>
    );
  }

  if (!health) {
    return (
      <div className="inline-flex items-center gap-2 text-sm text-gray-500">
        <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
        Connecting...
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 text-sm text-green-600">
      <span className="w-2 h-2 rounded-full bg-green-500" />
      {health.num_indexed_images.toLocaleString()} images indexed
    </div>
  );
}
