/**
 * Inline icon set.
 *
 * Hand-rolled rather than pulled from a library: the app needs nine
 * glyphs, and a dependency for that is ~50KB to ship and one more thing
 * to keep current. Every icon is 24x24, 1.5px stroke, `currentColor`, so
 * they inherit text colour and stay consistent across both themes.
 *
 * Icons are decorative here — every one sits beside a text label or
 * inside a button with an accessible name — so they are hidden from
 * assistive technology.
 */

import type { CSSProperties } from 'react';

type IconProps = {
  className?: string;
  style?: CSSProperties;
};

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
  focusable: false,
};

export function SearchIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </svg>
  );
}

export function ImageIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="m21 15-5-5L5 21" />
    </svg>
  );
}

export function UploadIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 9l5-5 5 5" />
      <path d="M12 4v12" />
    </svg>
  );
}

export function CloseIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

export function SunIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

export function MoonIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79Z" />
    </svg>
  );
}

export function MonitorIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  );
}

export function AlertIcon({ className = 'w-5 h-5', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v5M12 16h.01" />
    </svg>
  );
}

export function ArrowRightIcon({ className = 'w-4 h-4', style }: IconProps) {
  return (
    <svg {...base} className={className} style={style}>
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}
