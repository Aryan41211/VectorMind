/**
 * Application entry point.
 *
 * The ErrorBoundary sits outside StrictMode so it also catches errors
 * thrown during StrictMode's deliberate double-render in development —
 * those are real bugs and should surface, not blank the page.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import './index.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Root element #root is missing from index.html');
}

createRoot(container).render(
  <ErrorBoundary>
    <StrictMode>
      <App />
    </StrictMode>
  </ErrorBoundary>
);
