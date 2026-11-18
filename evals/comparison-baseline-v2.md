# Before and after: `baseline` vs `v2`

`v2`: v1 plus three rules added to the end of the system prompt: every statement must come from the documents (no added facts, tips, caveats, or recommendations, even true ones); link only to URLs from a document's "Source:" line; and base code on the documents' examples without adding imports, parameters, or options. Retrieval, the rewrite step, and the judge are unchanged, so v2 is compared with v1's runs. 39 cases, 2 runs each per variant. Pass rates average each case's runs; ranges are 95% confidence intervals over cases.

| Cases | n | Pass before | Pass after | Change (paired) |
|---|---|---|---|---|
| All cases | 39 | 74% (61%-88%) | 95% (89%-100%) | +21 pts (+9 to +32) |
| Should answer, dev | 22 | 68% (49%-87%) | 98% (93%-100%) | +30 pts (+12 to +47) |
| Should answer, held-out | 10 | 70% (40%-100%) | 85% (64%-100%) | +15 pts (-6 to +36) |
| Should decline | 5 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |
| Prompt injection | 2 | 100% (100%-100%) | 100% (100%-100%) | +0 pts (+0 to +0) |

## Retrieval (should-answer questions)

| Expected page in top 5 | Before | After | Change (paired) |
|---|---|---|---|
| 32 questions | 75% (60%-90%) | 95% (89%-100%) | +20 pts (+6 to +34) |

## Run-to-run noise

Same pass/fail on every run: 37/39 cases in `baseline`, 37/39 in `v2`. The rest flipped between runs of identical code, so a single run of one case is weak evidence.

## Cases that changed

- better: `admin-only` (0% -> 100%)
- better: `async-or-def` (50% -> 100%)
- better: `big-file` (0% -> 100%)
- better: `cors` (0% -> 50%)
- better: `env-settings` (0% -> 100%)
- better: `hide-password` (0% -> 100%)
- better: `nested-list` (0% -> 50%)
- better: `proxy-prefix` (0% -> 100%)
- better: `startup-model` (50% -> 100%)
- better: `timing-header` (0% -> 100%)

## Cost and speed per answer

- Cost, excluding the judge: $0.0331 -> $0.0366
- Median latency: 5.9 s -> 8.2 s
