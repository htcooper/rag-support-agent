# Before and after: `v1` vs `v2`

`v2`: v1 plus three rules added to the end of the system prompt: every statement must come from the documents (no added facts, tips, caveats, or recommendations, even true ones); link only to URLs from a document's "Source:" line; and base code on the documents' examples without adding imports, parameters, or options. Retrieval, the rewrite step, and the judge are unchanged, so v2 is compared with v1's runs. 39 cases, 2 runs each per variant. Pass rates average each case's runs; ranges are 95% confidence intervals over cases.

| Cases | n | Pass before | Pass after | Change (paired) |
|---|---|---|---|---|
| All cases | 39 | 87% (79%-96%) | 95% (89%-100%) | +8 pts (+0 to +15) |
| Should answer, dev | 22 | 89% (78%-100%) | 98% (93%-100%) | +9 pts (-3 to +21) |
| Should answer, held-out | 10 | 75% (53%-97%) | 85% (64%-100%) | +10 pts (-3 to +23) |
| Should decline | 5 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |
| Prompt injection | 2 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |

## Retrieval (should-answer questions)

| Expected page in top 5 | Before | After | Change (paired) |
|---|---|---|---|
| 32 questions | 95% (89%-100%) | 95% (89%-100%) | +0 pts (+0 to +0) |

## Run-to-run noise

Same pass/fail on every run: 33/39 cases in `v1`, 37/39 in `v2`. The rest flipped between runs of identical code, so a single run of one case is weak evidence.

## Cases that changed

- better: `async-or-def` (50% -> 100%)
- better: `db-per-request` (50% -> 100%)
- better: `docs-example` (50% -> 100%)
- better: `positive-id` (50% -> 100%)
- better: `proxy-prefix` (50% -> 100%)
- better: `upload` (0% -> 100%)
- worse: `cors` (100% -> 50%)

## Cost and speed per answer

- Cost, excluding the judge: $0.0345 -> $0.0366
- Median latency: 6.8 s -> 8.2 s
