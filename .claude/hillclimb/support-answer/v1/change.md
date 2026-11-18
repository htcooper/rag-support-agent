# v1: query rewriting before search

Before searching, Claude Haiku 5.5 restates the question in FastAPI's vocabulary, and the search uses the question plus the rewrite. Claude Opus 5.5 still receives only the original question, and the rewriter's instructions name no FastAPI features.

Why: 8 of the baseline's 10 failures were retrieval misses where the question used the user's words ("my React app gets blocked") and the docs used feature names ("CORSMiddleware").

Prediction, written before the run: dev Hit@5 from 17/22 to at least 20/22, and dev pass rate from 15/22 to about 18/22.
