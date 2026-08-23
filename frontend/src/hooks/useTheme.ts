/**
 * Theme state: light, dark, or follow the operating system.
 *
 * Three states rather than two. A plain boolean toggle silently opts the
 * visitor out of their OS preference the first time they touch it, and
 * gives them no way back.
 */

import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'vectormind:theme';

function readStoredTheme(): Theme {
  // Private windows and blocked site-data make localStorage throw on
  // access, not just return null, so this cannot be an optional chain.
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      return stored;
    }
  } catch {
    /* fall through to the default */
  }
  return 'system';
}

function prefersDark(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-color-scheme: dark)').matches === true
  );
}

/** Apply the resolved theme to <html>, which is what the CSS keys off. */
function applyTheme(theme: Theme): void {
  const isDark = theme === 'dark' || (theme === 'system' && prefersDark());
  document.documentElement.classList.toggle('dark', isDark);
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme);

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* the theme still applies for this session */
    }
  }, [theme]);

  // While following the system, track changes to it live — a visitor whose
  // OS flips to dark at sunset should not have to reload.
  useEffect(() => {
    if (theme !== 'system') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyTheme('system');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);

  const cycleTheme = useCallback(() => {
    setThemeState((current) =>
      current === 'system' ? 'light' : current === 'light' ? 'dark' : 'system'
    );
  }, []);

  const resolved: Exclude<Theme, 'system'> =
    theme === 'system' ? (prefersDark() ? 'dark' : 'light') : theme;

  return { theme, resolved, setTheme, cycleTheme };
}
