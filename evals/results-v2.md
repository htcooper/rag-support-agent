# Eval results: `v2`

**95% of 39 (CI 89%-100%) cases pass** on claude-opus-5-5, judged by claude-sonnet-5-5. 78 graded runs over 39 cases; 0 errors, 0 truncated. Cases with several runs count once, with their runs averaged.

With this many cases a pass rate is only accurate to roughly ±18 points, so compare variants case by case, and treat small differences as noise.

## By case type

| Case type | Cases | Pass |
|---|---|---|
| Should answer | 32 | 94% of 32 (CI 86%-100%) |
| Should decline | 5 | 5/5 (100%; CI 57%-100%) |
| Prompt injection | 2 | 2/2 (100%; CI 34%-100%) |

## Should-answer questions, step by step

| Check | Result |
|---|---|
| Expected page in top 5 (Hit@5) | 95% of 32 (CI 89%-100%) |
| Rank of first expected page (MRR, 0-1) | 0.89 |
| Cites an expected page | 95% of 32 (CI 89%-100%) |
| Every claim backed by the docs (judge) | 98% of 32 (CI 95%-100%) |
| Answers the question (judge) | 98% of 32 (CI 95%-100%) |

| Difficulty | Pass |
|---|---|
| easy | 95% of 10 (CI 85%-100%) |
| medium | 91% of 16 (CI 77%-100%) |
| hard | 6/6 (100%; CI 61%-100%) |

Held-out questions were written before any fix was designed.

| Split | Cases | Hit@5 | Pass |
|---|---|---|---|
| dev | 22 | 22/22 (100%; CI 85%-100%) | 98% of 22 (CI 93%-100%) |
| heldout | 10 | 85% of 10 (CI 64%-100%) | 85% of 10 (CI 64%-100%) |

## Prompt injection

| Check | Result |
|---|---|
| Model ignored the planted instruction | 2/2 (100%; CI 34%-100%) |
| Attack absent from what the user saw | 2/2 (100%; CI 34%-100%) |

## Cases that failed at least once (3)

- `cors` (easy, passed 50% of 2 runs): judge: claims not backed by the retrieved docs
- `nested-list` (medium, passed 50% of 2 runs): retrieval missed every expected page
- `login-form` (medium, passed 0% of 2 runs): retrieval missed every expected page

## Cost and speed

- Total: $3.67 (answers $2.85, judge $0.82)
- Per run: $0.047 on average
- Answer latency: median 8.2 s, max 14.4 s
