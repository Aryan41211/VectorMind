/**
 * Vitest setup: registers jest-dom matchers and resets state between tests.
 *
 * The API client reads a module-level API_BASE and the components fetch on
 * mount, so both fetch and timers are reset per test to keep cases isolated.
 */

import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
