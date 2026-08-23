/**
 * Catches render errors so one broken result cannot blank the page.
 *
 * React unmounts the whole tree when a render throws, and a white screen
 * is indistinguishable from a dead server. This keeps the failure legible
 * and gives the visitor a way out.
 *
 * It does NOT catch errors inside event handlers or promises — those are
 * handled where they happen, by ApiError in the client.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertIcon } from './Icon';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // No error-reporting service is wired up for a portfolio demo, so the
    // console is the record. Keep the component stack — it is the part
    // that says which component threw.
    console.error('Render error:', error, info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        role="alert"
        className="min-h-screen flex items-center justify-center p-6"
      >
        <div className="surface border border-subtle rounded-card shadow-card max-w-md w-full p-8 text-center">
          <span
            className="inline-flex items-center justify-center w-12 h-12 rounded-full mb-4"
            style={{ background: 'var(--accent-subtle)', color: 'var(--danger)' }}
          >
            <AlertIcon className="w-6 h-6" />
          </span>
          <h1 className="text-lg font-semibold text-primary mb-2">
            Something broke while rendering
          </h1>
          <p className="text-sm text-secondary mb-6">
            The interface hit an unexpected error. Your search did not reach the
            server, so nothing was lost.
          </p>
          <pre className="text-left text-xs font-mono surface-sunken border border-subtle rounded-lg p-3 mb-6 overflow-x-auto text-tertiary">
            {error.message}
          </pre>
          <button
            type="button"
            onClick={this.handleReset}
            className="w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
            style={{
              background: 'var(--accent)',
              color: 'var(--accent-contrast)',
            }}
          >
            Try again
          </button>
        </div>
      </div>
    );
  }
}
