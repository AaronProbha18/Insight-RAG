# evals

Evaluation data. The code that runs evals lives in `backend/src/plumbline/evals`.

## Fixture corpus

`corpus-src/` holds four short documents about **Lumen Ridge Outfitters**,
a fictional outdoor-gear company. All of it was written for this project:
the company, the roles and the facts are invented, with no real people,
organisations or data. That keeps the corpus free to redistribute under
the repository's MIT license.

The same source is rendered into both formats, so each connector is
tested on identical facts:

| Folder | Contents |
| --- | --- |
| `corpus-src/*.md` | source of truth: edit these |
| `connectors/pdf/corpus/*.pdf` | generated with reportlab; `<!-- pagebreak -->` fixes page numbers |
| `connectors/docx/corpus/*.docx` | generated with python-docx; Markdown headings become Heading 1-3 styles |

The generated files are committed, so evals and connector tests don't
need the generator. After editing a source file, regenerate them from the
repo root:

```bash
uv run --no-project --with reportlab --with python-docx python evals/generate_corpus.py
```

The output is byte-for-byte reproducible on a given platform: reportlab
runs in invariant mode, and DOCX zip timestamps and core properties are
pinned. Unchanged sources therefore produce no git diff.

Golden questions cite locators such as `p. 2` or `Leave > Annual Leave`.
If an edit moves a fact to another page or heading, update the golden
sets too.

## Running evals

Requires Postgres (`docker compose up -d db`) and these environment
variables: `PLUMBLINE_LLM`, `PLUMBLINE_EMBEDDINGS`,
`PLUMBLINE_EMBEDDING_DIM`, `PLUMBLINE_JUDGE_LLM`, and optionally
`PLUMBLINE_ABSTAIN_THRESHOLD` and `DATABASE_URL`.

```bash
cd backend
uv run plumbline-eval                       # every connector with a golden set
uv run plumbline-eval --connectors pdf --sample 8   # quick iteration, not a gate result
```

Each connector runs in its own throwaway database, which is created,
migrated and dropped automatically. The judge is checked against
`judge/labelled.yaml` before anything else; until those labels are
human-reviewed (`reviewed: true`), the run stops with no gate result.

| Exit code | Meaning |
| --- | --- |
| 0 | every connector PASSes (correctness, faithfulness, abstention >= 0.80) |
| 1 | at least one connector FAILs; the summary names the metric |
| 2 | no PASS/FAIL is reported: judge not validated, or a pre-rename variable is set |

Committed evidence for a recorded baseline lives in `baselines/<change>/`
(summaries and smoke output only); `reports/` is scratch and gitignored.

The report (`reports/latest.json`, gitignored) records the model ids, the
threshold, the golden-set version, every metric, latency, and each
question's result.

Next to it, `reports/latest.summary.json` is the **shareable** result: models,
threshold, judge agreement, PASS/FAIL and the four metrics per connector,
latency, the answerable and unanswerable question counts, total duration,
git commit and host. It has no per-question text,
and it is written even when the judge gate stops the run (exit 2). In an
interactive terminal a progress bar shows judge validation, then ingest and
questions per connector with an ETA; `--no-progress` turns it off (it is off
automatically when output is piped).

The chat UI's Quality button reads this summary through `GET /quality`. Set
`PLUMBLINE_EVAL_SUMMARY` to read a different file (default:
`reports/latest.summary.json`). A sampled run (`--sample`) or a run that stops
at the judge gate (exit 2) shows "Not measured yet", so only full runs count.
In Docker the image does not contain `evals/`: mount the summary file into the
container and set `PLUMBLINE_EVAL_SUMMARY` to its path inside the container.

## Feedback candidates

People rate answers in the UI. `plumbline-feedback-export` turns every
rating into a candidate golden question for review:

```bash
cd backend
uv run plumbline-feedback-export            # writes evals/feedback/candidates.yaml
uv run plumbline-feedback-export --out PATH
```

| Feedback | `answerable` | `supporting` |
| --- | --- | --- |
| thumbs up on an answer | `true` | the cited file name (no extension) and locator |
| thumbs up on an abstention | `false` | none |
| thumbs down, reason "should have answered" | `true` | none (a reviewer adds it) |
| thumbs down, any other reason or none | `null`: the reviewer decides | the cited file name and locator |

Review flow: open the file, and for each entry you keep, write the
`reference_answer` and `key_fact`, decide `answerable` where it is `null`,
check `supporting`, delete the `feedback:` block, and paste the entry into
the right connector's `golden.yaml`. Nothing is added to a golden set
automatically, and the golden-set validator rejects an entry until it is
complete.

The file holds real questions and answers typed by users. It is
git-ignored: review it, don't commit it.
