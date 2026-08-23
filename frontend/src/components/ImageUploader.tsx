/**
 * Image upload by drop, click, or paste.
 *
 * Validation happens here rather than at the API: rejecting a 40MB TIFF
 * before it is uploaded is faster and clearer than a 413 after the wait.
 * The backend still validates independently — this is a courtesy, not a
 * security boundary.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from 'react';
import { ImageIcon, UploadIcon } from './Icon';

interface ImageUploaderProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
}

/** Flickr30k photographs are well under this; it exists to catch mistakes. */
const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

export function ImageUploader({ onUpload, isLoading }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  // Object URLs leak until revoked, and a visitor may try many images.
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = useCallback(
    (file: File) => {
      if (!ACCEPTED.includes(file.type)) {
        setError('That file is not a JPEG, PNG, WebP, or GIF.');
        return;
      }
      if (file.size > MAX_BYTES) {
        setError(
          `That image is ${(file.size / 1024 / 1024).toFixed(1)}MB. The limit is 10MB.`
        );
        return;
      }
      setError(null);
      setPreview((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(file);
      });
      onUpload(file);
    },
    [onUpload]
  );

  // Pasting a screenshot is the fastest path in, and costs three lines.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (isLoading) return;
      const file = Array.from(event.clipboardData?.files ?? [])[0];
      if (file) handleFile(file);
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [handleFile, isLoading]);

  // Depth counting: dragleave fires when crossing onto a child element,
  // so a naive handler flickers the highlight across the drop zone.
  const handleDragEnter = (event: DragEvent) => {
    event.preventDefault();
    dragDepth.current += 1;
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent) => {
    event.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) setIsDragging(false);
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    dragDepth.current = 0;
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
    // Reset so selecting the same file twice fires change again.
    event.target.value = '';
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragEnter={handleDragEnter}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        disabled={isLoading}
        aria-label="Upload an image to search by"
        className="relative w-full rounded-xl border-2 border-dashed transition-colors p-10 text-center disabled:cursor-not-allowed"
        style={{
          borderColor: isDragging ? 'var(--accent)' : 'var(--border-strong)',
          background: isDragging ? 'var(--accent-subtle)' : 'var(--surface-raised)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          onChange={handleChange}
          disabled={isLoading}
          className="hidden"
        />

        {preview ? (
          <img
            src={preview}
            alt="The image you uploaded"
            className="max-h-56 mx-auto rounded-lg shadow-card"
          />
        ) : (
          <div className="flex flex-col items-center gap-3">
            <span
              className="inline-flex items-center justify-center w-12 h-12 rounded-full"
              style={{ background: 'var(--surface-sunken)' }}
            >
              {isDragging ? (
                <UploadIcon className="w-6 h-6 text-accent" />
              ) : (
                <ImageIcon className="w-6 h-6 text-tertiary" />
              )}
            </span>
            <div>
              <p className="text-sm font-medium text-primary">
                {isDragging ? 'Drop to search' : 'Drop an image, click, or paste'}
              </p>
              <p className="text-xs text-tertiary mt-1">
                JPEG, PNG, WebP or GIF · up to 10MB
              </p>
            </div>
          </div>
        )}

        {isLoading && (
          <div
            className="absolute inset-0 flex items-center justify-center rounded-xl backdrop-blur-[2px]"
            style={{ background: 'var(--surface-raised)', opacity: 0.8 }}
          >
            <span className="text-sm font-medium text-accent">Searching…</span>
          </div>
        )}
      </button>

      {error && (
        <p role="alert" className="mt-3 text-xs text-center" style={{ color: 'var(--danger)' }}>
          {error}
        </p>
      )}
    </div>
  );
}
