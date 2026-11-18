# Superseded results: judge without the source line

These baseline and v1 runs (2 runs per case each) were graded by a judge that saw each document's title and text but not its `Source: <url>` line, which Claude did see. The judge therefore marked correct mentions of a document's URL as "not supported" (for example, both v1 runs of `live-updates`).

The judge was fixed to receive the same document blocks as Claude, a regression test now checks that every field reaches it, and both variants were re-run from scratch. Results from before and after a grader change aren't comparable, so these files are kept only as a record and aren't used in any reported number.

Retrieval scores don't depend on the judge, and the retrieval comparison in `evals/retrieval.md` is unaffected.
