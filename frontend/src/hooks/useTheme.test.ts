/**
 * Tests for the theme hook.
 *
 * Two things are easy to get wrong here and both are user-visible: the
 * "system" state must survive a reload, and every localStorage access
 * must tolerate throwing — private windows and blocked site-data raise
 * on access rather than returning null, which would take the whole app
 * down at mount.
 */

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTheme } from './useTheme';

const KEY = 'vectormind:theme';

function mockMatchMedia(prefersDark: boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn().mockImplementation((query: string) => ({
      matches: query.includes('dark') ? prefersDark : false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
  );
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove('dark');
  mockMatchMedia(false);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useTheme', () => {
  it('defaults to following the system', () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('system');
  });

  it('restores a stored preference', () => {
    localStorage.setItem(KEY, 'dark');
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('dark');
  });

  it('ignores a corrupted stored value', () => {
    localStorage.setItem(KEY, 'chartreuse');
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('system');
  });

  it('adds the dark class when dark is chosen', () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('dark'));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('removes the dark class when light is chosen', () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('dark'));
    act(() => result.current.setTheme('light'));
    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('follows the OS when set to system', () => {
    mockMatchMedia(true);
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('system'));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(result.current.resolved).toBe('dark');
  });

  it('persists the choice', () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('light'));
    expect(localStorage.getItem(KEY)).toBe('light');
  });

  it('cycles system → light → dark → system', () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('system');
    act(() => result.current.cycleTheme());
    expect(result.current.theme).toBe('light');
    act(() => result.current.cycleTheme());
    expect(result.current.theme).toBe('dark');
    act(() => result.current.cycleTheme());
    expect(result.current.theme).toBe('system');
  });

  it('survives localStorage throwing on read', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('Access denied', 'SecurityError');
    });
    expect(() => renderHook(() => useTheme())).not.toThrow();
  });

  it('survives localStorage throwing on write', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Quota exceeded', 'QuotaExceededError');
    });
    const { result } = renderHook(() => useTheme());
    expect(() => act(() => result.current.setTheme('dark'))).not.toThrow();
    // The theme must still apply for this session even unsaved.
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});
