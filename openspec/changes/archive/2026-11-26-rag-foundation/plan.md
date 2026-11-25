## Task
7.4 Baseline run with real models: tune abstain threshold once, record threshold and results in the report and in eval-plan.md Baseline column

## Scope
- **The runs are done** (on your M1, 2026-11-25, commit `476445a`). This loop **records** them; it re-runs nothing. Models: `ollama:llama3.1:8b` (chat), `ollama:nomic-embed-text` (embeddings), `ollama:qwen2.5:7b` (judge).
- **The decision:** keep `DEFAULT_THRESHOLD = 0.3`. The single allowed tuning (0.2) was run and **didn't help**: mean correctness was 0.733 at both thresholds. Every PDF abstention on an answerable question had the right source retrieved (`hit=True`), and the other failures are citation-only replies (`"[c1]"`, `"[40] [c1]"`) or a mangled sentinel. The threshold isn't the lever, so 0.3 stays.
- **`eval-plan.md`:**
  - **Metrics table:** fill the Baseline column from run A (threshold 0.3): correctness DOCX 0.767 / PDF 0.700 (FAIL), faithfulness 0.897 / 0.875, abstention 1.0 / 1.0, hit-rate 0.967 / 1.0, judge agreement 1.0 / 0.9, latency p95 61.0 s / 58.0 s.
  - **New "Baseline run (task 7.4)" section:** the models, the host (M1, arm64), both runs side by side (A at 0.3, B at 0.2), the decision to keep 0.3, the failure diagnosis above as a table of causes with question ids, and the verdict: **the pass rule isn't met (correctness < 0.80 on both connectors in run A)**, with the fix routed to a follow-up issue on answer generation.
- **Evidence committed:** the three shareable summaries plus the smoke output go into a new tracked folder, `evals/baselines/rag-foundation/` (`reports/` is gitignored): `m1-sample.summary.json`, `m1-7.4-t030.summary.json`, `m1-7.4-t020.summary.json`, `m1-smoke.txt`. They're synthetic-corpus results with no per-question text (the smoke line shows one synthetic doc id and answer). The full 60 KB reports stay out of git; I'll quote the diagnosis numbers from them in `eval-plan.md`.
- **`design.md` D7:** one line recording the calibration result: 0.3 kept, and why.
- **`backend/src/insight_rag/graph.py`:** comment only. Replace the `ponytail: placeholder; calibrated once … in the 7.4 baseline` note with the calibrated outcome. `DEFAULT_THRESHOLD` stays `0.3`, so **no behaviour change**. Backend guide §5.3 treats threshold *changes* as answer-quality changes; this isn't one.
- **`tasks.md`:** tick 7.4, and update 9.2's note from "pending M1" to the M1 result (health, upload 201, cited answer, all PASS).
- **Branch:** `chore/rag-foundation-7.4-baseline` from `origin/main`, then a PR, because `main` is protected. The current checkout (`feat/23-saved-settings`) belongs to other work and isn't touched.
- Guides read: `docs/guides/backend-development-guide.md` (§5.3: threshold or prompt changes need evals; here the threshold value is unchanged). No frontend or connector code is touched.

## Not in scope
- **Fixing correctness:** prompt or format changes, retrying citation-only replies, temperature, a larger model. These are answer-quality changes, so they go to a **new GitHub issue** (`answer-quality`, standard track). I'll draft it for your review, separately.
- Running `/rag-verify` (next step; it will report **FAIL** on correctness, which is the honest outcome).
- Re-basing the rename's 5.1 on this baseline: that's PR #15's change. It gets a note there, not here.

## Files
| File | Change |
| --- | --- |
| `openspec/changes/rag-foundation/eval-plan.md` | fill the Baseline column; add a "Baseline run (task 7.4)" section |
| `openspec/changes/rag-foundation/design.md` | D7: calibration outcome (one line) |
| `openspec/changes/rag-foundation/tasks.md` | tick 7.4; update 9.2's note |
| `backend/src/insight_rag/graph.py` | comment only, next to `DEFAULT_THRESHOLD` |
| `evals/baselines/rag-foundation/*.summary.json`, `m1-smoke.txt` | new: the evidence (4 files) |
| `evals/README.md` | one line: `baselines/` holds committed evidence; `reports/` stays scratch |

## Tests
- No new tests: this records results. Its checks:
  - every number written into `eval-plan.md` is read **from the summary JSONs by a script** and compared, so nothing is retyped wrong;
  - the diagnosis counts (abstained with `hit=True`, citation-only replies) are recomputed from the full reports by the same script;
  - the graph change is a comment only: `git diff` shows only `#` lines and the value is still `0.3`.

## Checks to run
- a number-check script: summaries and full reports vs the values in `eval-plan.md`
- `cd backend && uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
- `openspec validate rag-foundation` and `python scripts/check_links.py`
- push the branch, open a PR, and CI must be green (`main` is protected)

## Risks / Questions
- **Ticking 7.4 while the baseline fails.** The task is "run, tune once, record", and all three are done. The *change's* pass rule is what fails, and `rag-verify` will report that. I think that's the honest split; say so if you'd rather leave 7.4 open until correctness passes.
- **Run-to-run noise:** about ±0.03–0.07 between runs at the same settings (different questions fail in A and B). I'll record that as a finding, because it affects every future comparison, including the rename's ±0.05 rule.
