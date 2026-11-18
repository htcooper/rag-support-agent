# Before and after: `baseline` vs `v1`

`v1` adds query rewriting: Claude Haiku 5.5 restates each question in FastAPI's vocabulary before the search. Everything else is the same. 39 cases, 2 runs each per variant. Pass rates average each case's runs; ranges are 95% confidence intervals over cases.

| Cases | n | Pass before | Pass after | Change (paired) |
|---|---|---|---|---|
| All cases | 39 | 74% (61%-88%) | 87% (79%-96%) | +13 pts (-1 to +27) |
| Should answer, dev | 22 | 68% (49%-87%) | 89% (78%-100%) | +20 pts (-2 to +42) |
| Should answer, held-out | 10 | 70% (40%-100%) | 75% (53%-97%) | +5 pts (-22 to +32) |
| Should decline | 5 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |
| Prompt injection | 2 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |

## Retrieval (should-answer questions)

| Expected page in top 5 | Before | After | Change (paired) |
|---|---|---|---|
| 32 questions | 75% (60%-90%) | 95% (89%-100%) | +20 pts (+6 to +34) |

## Run-to-run noise

Same pass/fail on every run: 37/39 cases in `baseline`, 33/39 in `v1`. The rest flipped between runs of identical code, so a single run of one case is weak evidence.

## Cases that changed

- better: `admin-only` (0% -> 100%)
- better: `big-file` (0% -> 100%)
- better: `cors` (0% -> 100%)
- better: `env-settings` (0% -> 100%)
- better: `hide-password` (0% -> 100%)
- better: `nested-list` (0% -> 50%)
- better: `proxy-prefix` (0% -> 50%)
- better: `startup-model` (50% -> 100%)
- better: `timing-header` (0% -> 100%)
- worse: `db-per-request` (100% -> 50%)
- worse: `docs-example` (100% -> 50%)
- worse: `positive-id` (100% -> 50%)
- worse: `upload` (100% -> 0%)

## Cost and speed per answer

- Cost, excluding the judge: $0.0331 -> $0.0345
- Median latency: 5.9 s -> 6.8 s
