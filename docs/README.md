# VectorMind Documentation

Start with [../README.md](../README.md). This index says which document
answers which question, so you do not have to open all sixteen.

If any document here contradicts [../CLAUDE.md](../CLAUDE.md) or
[../ARCHITECTURE.md](../ARCHITECTURE.md) §1-8, those two win — flag the
contradiction rather than silently picking one.

---

## Read these first

| Question | Document |
|---|---|
| What is this and why does it exist? | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| What actually works right now? | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| **What is broken or misleading?** | **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** |
| Why was it built this way? | [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) |
| What went wrong along the way? | [DEBUGGING_STORY.md](DEBUGGING_STORY.md) |

`KNOWN_ISSUES.md` is deliberately in that list. The headline result —
25.64% test Recall@10, 82× chance — is real, but the embedding space
behind it still grades ANISOTROPIC rather than healthy. Read it before
quoting a number.

## Results and experiments

| Question | Document |
|---|---|
| What did each training run produce? | [TRAINING_LOG.md](TRAINING_LOG.md) |
| What was the experiment log? | [EXPERIMENTS.md](EXPERIMENTS.md) |
| What hardware could it fit? | [PHASE_0_REPORT.md](PHASE_0_REPORT.md) |
| Where did the data come from, under what licence? | [DATASETS.md](DATASETS.md) |

Raw metric files live in `../reports/`.

## Working on the code

| Question | Document |
|---|---|
| What are the non-negotiable rules? | [PROJECT_RULES.md](PROJECT_RULES.md) |
| How is a feature taken from idea to merge? | [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) |
| What style is expected? | [CODING_STANDARDS.md](CODING_STANDARDS.md) |
| What belongs in which directory? | [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) |
| Why this library and not that one? | [TECH_STACK.md](TECH_STACK.md) |

## Longer-running context

| Question | Document |
|---|---|
| Why was a past decision made? | [PROJECT_MEMORY.md](PROJECT_MEMORY.md) |
| What is explicitly out of scope? | [FUTURE_IDEAS.md](FUTURE_IDEAS.md) |
