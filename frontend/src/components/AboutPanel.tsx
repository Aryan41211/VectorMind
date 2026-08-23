/**
 * What this demo is, and what it cannot do.
 *
 * Shown on the idle screen rather than hidden behind a link. A visitor
 * whose first query returns three unrelated photographs should already
 * know they are looking at a 24M-parameter model trained from scratch on
 * 31k images, not a production CLIP — otherwise the honest conclusion
 * from a mediocre result is "this is broken".
 *
 * Numbers here are the shipped model's measured figures. They are stated
 * against the correct random-chance baseline, which the project's own
 * documentation got wrong by roughly 30x for weeks: chance for R@10 over
 * 15,895 captions is 0.31%, not the 10% carried over from the 100-image
 * Phase 3.5 sanity check.
 */

const METRICS = [
  { label: 'Test R@10', value: '19.6%', note: '62× chance' },
  { label: 'Test R@1', value: '4.6%', note: '147× chance' },
  { label: 'Parameters', value: '24M', note: 'from scratch' },
  { label: 'Corpus', value: '3,179', note: 'Flickr30k photos' },
];

const LIMITATIONS = [
  {
    title: 'Fine-grained actions',
    body: 'Scene-level concepts land well; specific verbs ("kicking" vs "holding") often do not.',
  },
  {
    title: 'Attribute binding',
    body: 'A blue boat and a red boat are hard for it to tell apart — colour rarely binds to the right object.',
  },
  {
    title: 'Spatial relations',
    body: 'Multi-object arrangements ("a cat on a mat beside a dog") are barely above chance.',
  },
];

export function AboutPanel() {
  return (
    <section className="w-full max-w-3xl mx-auto mt-16 animate-in">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px rounded-card overflow-hidden border border-subtle">
        {METRICS.map(({ label, value, note }) => (
          <div
            key={label}
            className="p-4 text-center"
            style={{ background: 'var(--surface-raised)' }}
          >
            <div className="text-xl font-semibold text-primary tabular">
              {value}
            </div>
            <div className="text-[11px] text-secondary mt-1">{label}</div>
            <div className="text-[10px] text-tertiary mt-0.5">{note}</div>
          </div>
        ))}
      </div>

      <div className="mt-6 grid md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-sm font-semibold text-primary mb-2">
            What this is
          </h2>
          <p className="text-sm text-secondary leading-relaxed">
            A CLIP-style dual encoder — a small CNN and a small Transformer
            projected into one 256-dimensional space — trained entirely from
            scratch on Flickr30k. No pretrained vision-language weights, on a
            single 6GB laptop GPU.
          </p>
          <p className="text-sm text-secondary leading-relaxed mt-3">
            It is a study in whether the architecture can be made to work at a
            scale one person can actually train, not an attempt to match a
            model trained on 400 million pairs.
          </p>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-primary mb-2">
            Where it falls down
          </h2>
          <ul className="space-y-2.5">
            {LIMITATIONS.map(({ title, body }) => (
              <li key={title} className="text-sm">
                <span className="text-primary font-medium">{title}. </span>
                <span className="text-secondary">{body}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
