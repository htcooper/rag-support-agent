# v2: a stricter answer prompt

v1 plus three rules added to the end of the system prompt: every statement must come from the documents (no added facts, tips, caveats, or recommendations, even true ones); link only to URLs from a document's "Source:" line; and base code on the documents' examples without adding imports, parameters, or options. Retrieval, the rewrite step, and the judge are unchanged, so v2 is compared with v1's runs.

Why: in 6 of v1's 8 imperfect cases the right page was retrieved, and the answer added a detail no retrieved excerpt states.

Prediction, written before the run, on the 32 should-answer questions: "every claim supported" from 86% to about 95%, and pass from 87% (all 39 cases) to about 92%.

Counts as worse if "answers the question" falls below 95% (98% in v1), or if any should-answer question is declined.
