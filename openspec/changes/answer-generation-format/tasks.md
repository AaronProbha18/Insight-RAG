## 1. Generation

- [ ] 1.1 Temperature 0: `models.chat_model()` passes `temperature=0` to `init_chat_model`; `judge_model()` is unchanged. Test: building the chat model with an `ollama:` string gives `.temperature == 0` (confirms the UNVERIFIED pass-through in design.md); the judge model has no temperature override.
- [ ] 1.2 Answer-first `PROMPT` with one synthetic worked example (`NOT_IN_SOURCES` rule unchanged). Test: a scripted LLM records the prompt it receives; assert the answer-first instruction and the example are present.
- [ ] 1.3 `_classify` and the one corrective retry in `graph.generate` (citation-only and no-valid-citation retry; `NOT_IN_SOURCES` abstains without a retry; at most two calls). Test, with a scripted LLM, one per spec scenario: citation-only then valid -> answer; citation-only twice -> abstain after 2 calls; text without a valid citation then valid -> answer; `NOT_IN_SOURCES` -> abstain after 1 call; `"[40] [c1]"` counts as citation-only.

## 2. Eval report

- [ ] 2.1 `support_rank` and `support_score` in `QuestionScore` and `score_question` (first retrieved chunk matching a golden `(document, locator)`, 1-based; null when not retrieved or unanswerable); written per question in the full report, with the summary unchanged. Test: retrieved at position 2 -> rank 2 and its score; not retrieved -> null/null; unanswerable -> null/null.

## 3. Eval and docs

- [ ] 3.1 Docs: `docs/architecture/overview.md` Ask flow step 4 (answer-first format, one corrective retry, temperature 0) and the backend development guide section 5 if it describes the generate step. Test: `python scripts/check_links.py` passes.
- [ ] 3.2 Eval run 1 and run 2 on the reference M1 per `eval-plan.md` (owner runs them; needs the M1 and Ollama). Commit both summaries to `evals/baselines/answer-generation-format/`, and record in `eval-plan.md` a results table against 7.4, the failure-cause table with `support_rank`/`support_score`, and the run-to-run spread. Test: the pass rule in `eval-plan.md` is met in both runs, or the task is reported as FAIL with the evidence.
