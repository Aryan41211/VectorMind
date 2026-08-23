/**
 * Three-state theme control: follow system, force light, force dark.
 *
 * A segmented control rather than a single cycling button, because a
 * cycling button gives no indication of what the next press does, and
 * "system" is invisible in it — the state most visitors are in.
 */

import type { Theme } from '../hooks/useTheme';
import { MonitorIcon, MoonIcon, SunIcon } from './Icon';

interface ThemeToggleProps {
  theme: Theme;
  onChange: (theme: Theme) => void;
}

const OPTIONS: { value: Theme; label: string; Icon: typeof SunIcon }[] = [
  { value: 'system', label: 'Match system theme', Icon: MonitorIcon },
  { value: 'light', label: 'Light theme', Icon: SunIcon },
  { value: 'dark', label: 'Dark theme', Icon: MoonIcon },
];

export function ThemeToggle({ theme, onChange }: ThemeToggleProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="inline-flex items-center gap-0.5 p-0.5 rounded-lg border border-subtle"
      style={{ background: 'var(--surface-sunken)' }}
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => onChange(value)}
            className="p-1.5 rounded-md transition-colors"
            style={{
              background: active ? 'var(--surface-raised)' : 'transparent',
              color: active ? 'var(--text-primary)' : 'var(--text-tertiary)',
              boxShadow: active ? 'var(--shadow-card)' : 'none',
            }}
          >
            <Icon className="w-4 h-4" />
          </button>
        );
      })}
    </div>
  );
}
