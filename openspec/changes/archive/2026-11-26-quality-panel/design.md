## Context

PR #19 (merged to `main`, now in this branch) makes the eval runner write
`<report>.summary.json` next to the report. Its fields: `started_at`,
`git_commit`, `host`, `models` (`llm`, `embeddings`, `judge`),
`threshold`, `top_k`, `sample`, `result` (`PASS` / `FAIL` /
`NO_GATE_RESULT`), `exit_code`, `reason` (exit 2 only),
`judge_agreement`, `duration_s`, `report` (a file path), and `connectors`
mapping name -> `result`, `failed` (gated metric names below 0.80),
`metrics` (rounded), `latency_ms`, `questions`, `golden_version`. Gated
metrics are `correctness`, `faithfulness`, `abstention`; the bar is
`GATE = 0.80` (`evals/scoring.py`). The runner's default report is
`evals/reports/latest.json`. Model ids at runtime come from
`PLUMBLINE_LLM` / `PLUMBLINE_EMBEDDINGS`; the threshold from
`PLUMBLINE_ABSTAIN_THRESHOLD` (default `DEFAULT_THRESHOLD`).

## Decisions

### Read the summary file, allowlist the output

- **Choice**: `quality.load_quality(path, running) -> dict` reads the
  JSON and builds the response from named fields only. Missing,
  unreadable or malformed file -> `not_measured`. `host`, `git_commit`,
  `report`, `reason`, `golden_version`, `latency_ms`, `questions`,
  `top_k`, `duration_s`, `judge_agreement` are never returned.
- **Alternatives**: return the file as-is; read the full report.
- **Why**: AC "summary fields only (no per-question answers, file paths
  or internal ids)"; the summary already holds a file path and host info.
- **ADR**: none (local to this change)

### PASS/FAIL comes from the runner, not recomputed

- **Choice**: use each connector's `result` and `failed` as written. The
  panel shows the three gated metrics and names those in `failed`.
- **Alternatives**: recompute against 0.80 in the API.
- **Why**: one source of truth for the gate (`scoring.gate`).
- **ADR**: none

### When is it "not measured"

- **Choice**: `not_measured` if the file is missing/invalid, `result ==
  "NO_GATE_RESULT"` / `exit_code == 2`, or `sample` is not null.
- **Why**: exit 2 is in the AC; product owner decided sampled runs are not
  gate results.
- **ADR**: none

### Configuration mismatch

- **Choice**: compare `models.llm`, `models.embeddings` and `threshold`
  with the running `PLUMBLINE_LLM`, `PLUMBLINE_EMBEDDINGS` and effective
  threshold; any difference -> `config_matches: false` plus which of the
  three differ. The judge is not compared (it doesn't affect answers).
- **Why**: AC names exactly these three.
- **ADR**: none

### Source path

- **Choice**: `PLUMBLINE_EVAL_SUMMARY`, default
  `evals/reports/latest.summary.json` resolved from the repo root as the
  runner does. Read on each request (small file, no cache).
- **Alternatives**: fixed path (always empty in Docker).
- **Why**: product owner choice; Docker users mount the file.
- **ADR**: none

### UI: button + dialog

- **Choice**: "Quality: PASS | FAIL | Not measured" button in the upload
  bar (state as text, plus an icon), opening a Base UI `Dialog` like the
  Documents dialog (#12). Per connector: name, PASS/FAIL as a word and an
  icon, the three metrics, failed ones named "below 0.80"; run date and
  model ids once. Mismatch: a warning naming what differs. Not measured:
  the text "Not measured yet", no numbers, and the command
  `uv run plumbline-eval`.
- **Why**: product owner choice; PASS/FAIL not by colour alone.
- **ADR**: none

## Interfaces

```python
# GET /quality -> 200 always
{
  "state": "measured" | "not_measured",
  # only when measured:
  "result": "PASS" | "FAIL",
  "run_at": "2026-11-25T09:00:00+00:00",
  "models": {"chat": "...", "embeddings": "...", "judge": "..."},
  "threshold": 0.3,
  "config_matches": true,
  "config_differences": [],            # subset of ["chat", "embeddings", "threshold"]
  "connectors": {
    "pdf": {"result": "PASS", "failed": [],
            "metrics": {"correctness": 0.9, "faithfulness": 0.88, "abstention": 0.95}}
  },
  # only when not measured:
  "eval_command": "uv run plumbline-eval"
}
```

```ts
// frontend/src/quality/api.ts
export async function getQuality(): Promise<Quality>   // throws on non-200 / network
```

## Data

None. Reads one JSON file.

## Risks / Open Questions

- PR #19 must be in the branch before implementation: the stack
  (#15 -> #17 -> #18 -> this) is based on pre-#19 `main`, and #19's runner
  still uses the old `insight_rag` names, so the rebase has conflicts in
  `evals/runner.py` to resolve with the rename.
- A summary from a newer/older runner may lack fields; any missing
  required field -> `not_measured` (tested).
- Static-markup tests can't open the dialog: keyboard and both themes are
  checked by hand.
