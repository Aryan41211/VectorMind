/**
 * Loading placeholders shaped like the results that will replace them.
 *
 * A spinner says "wait". A skeleton says "here is what is coming, and
 * where", and it reserves the layout so the page does not jump when
 * results arrive — which on a grid of ten images is a large shift.
 */

const PLACEHOLDER_COUNT = 8;

export function ResultSkeleton() {
  return (
    <section
      className="w-full max-w-6xl mx-auto mt-10"
      aria-busy="true"
      aria-label="Loading results"
    >
      <div className="flex items-center justify-between pb-4 mb-6 border-b border-subtle">
        <div className="skeleton h-4 w-40 rounded" />
        <div className="skeleton h-3 w-24 rounded" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {Array.from({ length: PLACEHOLDER_COUNT }, (_, index) => (
          <div
            key={index}
            className="skeleton aspect-square rounded-card"
            /* Stagger so the grid ripples rather than pulsing as one
               block, which reads as a page that is working. */
            style={{ animationDelay: `${index * 70}ms` }}
          />
        ))}
      </div>
      <span className="sr-only">Searching…</span>
    </section>
  );
}
