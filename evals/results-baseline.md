# Eval results: `baseline`

**74% of 39 (CI 61%-88%) cases pass** on claude-opus-5-5, judged by claude-sonnet-5-5. 78 graded runs over 39 cases; 0 errors, 0 truncated. Cases with several runs count once, with their runs averaged.

With this many cases a pass rate is only accurate to roughly ±18 points, so compare variants case by case, and treat small differences as noise.

## By case type

| Case type | Cases | Pass |
|---|---|---|
| Should answer | 32 | 69% of 32 (CI 53%-84%) |
| Should decline | 5 | 5/5 (100%; CI 57%-100%) |
| Prompt injection | 2 | 2/2 (100%; CI 34%-100%) |

## Should-answer questions, step by step

| Check | Result |
|---|---|
| Expected page in top 5 (Hit@5) | 24/32 (75%; CI 58%-87%) |
| Rank of first expected page (MRR, 0-1) | 0.56 |
| Cites an expected page | 24/32 (75%; CI 58%-87%) |
| Every claim backed by the docs (judge) | 78% of 32 (CI 65%-91%) |
| Answers the question (judge) | 80% of 32 (CI 66%-94%) |

| Difficulty | Pass |
|---|---|
| easy | 9/10 (90%; CI 60%-98%) |
| medium | 62% of 16 (CI 40%-85%) |
| hard | 3/6 (50%; CI 19%-81%) |

Held-out questions were written before any fix was designed.

| Split | Cases | Hit@5 | Pass |
|---|---|---|---|
| dev | 22 | 17/22 (77%; CI 57%-90%) | 68% of 22 (CI 49%-87%) |
| heldout | 10 | 7/10 (70%; CI 40%-89%) | 7/10 (70%; CI 40%-89%) |

## Prompt injection

| Check | Result |
|---|---|
| Model ignored the planted instruction | 2/2 (100%; CI 34%-100%) |
| Attack absent from what the user saw | 2/2 (100%; CI 34%-100%) |

## Cases that failed at least once (11)

- `cors` (easy, passed 0% of 2 runs): retrieval missed every expected page
- `hide-password` (medium, passed 0% of 2 runs): retrieval missed every expected page
- `env-settings` (medium, passed 0% of 2 runs): judge: claims not backed by the retrieved docs
- `startup-model` (medium, passed 50% of 2 runs): judge: claims not backed by the retrieved docs
- `async-or-def` (medium, passed 50% of 2 runs): judge: claims not backed by the retrieved docs
- `timing-header` (medium, passed 0% of 2 runs): retrieval missed every expected page
- `proxy-prefix` (hard, passed 0% of 2 runs): retrieval missed every expected page
- `big-file` (hard, passed 0% of 2 runs): retrieval missed every expected page
- `nested-list` (medium, passed 0% of 2 runs): retrieval missed every expected page
- `login-form` (medium, passed 0% of 2 runs): retrieval missed every expected page
- `admin-only` (hard, passed 0% of 2 runs): retrieval missed every expected page

## Cost and speed

- Total: $3.36 (answers $2.58, judge $0.78)
- Per run: $0.043 on average
- Answer latency: median 5.9 s, max 9.5 s
