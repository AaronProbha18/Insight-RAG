"""`plumbline-eval`: per-connector trust gate over golden sets (design D9).

Exit codes: 0 all connectors PASS, 1 any FAIL, 2 judge not validated (no gate result).
"""

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql

from plumbline import models
from plumbline.config import LegacyConfigError, check_legacy_env
from plumbline.connectors.registry import load_connectors
from plumbline.db import database_url
from plumbline.evals.golden import load_golden, load_labelled
from plumbline.evals.judge import JudgeNotValidated, validate_judge
from plumbline.evals.scoring import aggregate, gate, score_question
from plumbline.graph import DEFAULT_THRESHOLD, build_graph
from plumbline.ingestion import ingest
from plumbline.retrieval import retrieve

BACKEND = Path(__file__).parents[3]
EVALS_DIR = BACKEND.parent / "evals"


@contextmanager
def _isolated_db(name: str):
    """A migrated throwaway database for one connector; always dropped."""
    admin_url = database_url()
    db = f"plumbline_eval_{name}_{secrets.token_hex(4)}"
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
        try:
            url = psycopg.conninfo.make_conninfo(admin_url, dbname=db)
            # migrations/env.py reads DATABASE_URL, so point it at the throwaway db.
            os.environ["DATABASE_URL"] = url
            try:
                command.upgrade(Config(BACKEND / "alembic.ini"), "head")
            finally:
                os.environ["DATABASE_URL"] = admin_url
            with psycopg.connect(url, autocommit=True) as conn:
                yield conn
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(db))
            )


def _sample(items, n):
    """First n items in file order, keeping the answerable/unanswerable ratio."""
    if not n:
        return items
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    n_un = round(n * len(unanswerable) / len(items))
    return answerable[: n - n_un] + unanswerable[:n_un]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[int(q * (len(ordered) - 1))], 1) if ordered else 0.0


def _eval_connector(
    name, connector, llm, judge, embeddings, evals_dir, sample, threshold, top_k
):
    golden_path = evals_dir / "connectors" / name / "golden.yaml"
    items = _sample(load_golden(golden_path), sample)
    corpus = evals_dir / "connectors" / name / "corpus"
    scores, questions, latencies = [], [], []
    with _isolated_db(name) as conn:
        doc_stems = {}
        for f in sorted(corpus.iterdir()):  # top level only: corrupt/ is not corpus
            if f.is_file() and f.suffix.lower() in connector.extensions:
                result = ingest(
                    conn, f.read_bytes(), f.name, {name: connector}, embeddings
                )
                doc_stems[result.document_id] = f.stem
        graph = build_graph(
            lambda q, k: retrieve(conn, embeddings, q, k), llm, threshold, top_k
        )
        for item in items:
            start = time.perf_counter()
            result = graph.invoke({"question": item.question})
            latencies.append((time.perf_counter() - start) * 1000)
            score = score_question(item, result, judge, doc_stems)
            scores.append(score)
            questions.append({**asdict(score), "latency_ms": round(latencies[-1], 1)})
    metrics = aggregate(scores)
    failed = gate(metrics)
    return {
        "golden_version": hashlib.sha256(golden_path.read_bytes()).hexdigest()[:12],
        "metrics": metrics,
        "result": "FAIL" if failed else "PASS",
        "failed": failed,
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "questions": questions,
    }


def run(
    connectors: dict,
    llm,
    judge,
    embeddings,
    model_ids: dict[str, str],
    report_path: Path,
    evals_dir: Path = EVALS_DIR,
    sample: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
) -> int:
    try:
        validation = validate_judge(
            judge,
            load_labelled(evals_dir / "judge" / "labelled.yaml"),
            model_ids["judge"],
        )
    except JudgeNotValidated as exc:
        print(f"NO GATE RESULT: {exc}")
        return 2

    results = {
        name: _eval_connector(
            name, c, llm, judge, embeddings, evals_dir, sample, threshold, top_k
        )
        for name, c in connectors.items()
    }
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "models": model_ids,
        "threshold": threshold,
        "top_k": top_k,
        "sample": sample,
        "judge_agreement": validation.agreement,
        "connectors": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for name, r in results.items():
        detail = f" (below 0.80: {', '.join(r['failed'])})" if r["failed"] else ""
        print(f"{name}: {r['result']}{detail} {r['metrics']}")
    if sample:
        print(f"SAMPLE of {sample} questions: not a gate result")
    print(f"report: {report_path}")
    return 1 if any(r["failed"] for r in results.values()) else 0


def main(argv: list[str] | None = None) -> int:
    try:
        check_legacy_env()
    except LegacyConfigError as exc:
        print(f"NO GATE RESULT: {exc}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(prog="plumbline-eval", description=__doc__)
    parser.add_argument(
        "--connectors", help="comma-separated names (default: all with a golden set)"
    )
    parser.add_argument(
        "--report", type=Path, default=EVALS_DIR / "reports" / "latest.json"
    )
    parser.add_argument(
        "--sample", type=int, help="ask only N questions, keeping the ratio"
    )
    args = parser.parse_args(argv)

    available = {
        name: c
        for name, c in load_connectors().items()
        if (EVALS_DIR / "connectors" / name / "golden.yaml").exists()
    }
    wanted = args.connectors.split(",") if args.connectors else list(available)
    unknown = sorted(set(wanted) - set(available))
    if unknown:
        parser.error(f"no connector with a golden set: {', '.join(unknown)}")

    return run(
        connectors={n: available[n] for n in wanted},
        llm=models.chat_model(),
        judge=models.judge_model(),
        embeddings=models.embeddings(),
        model_ids={
            "llm": os.environ["PLUMBLINE_LLM"],
            "embeddings": os.environ["PLUMBLINE_EMBEDDINGS"],
            "judge": os.environ["PLUMBLINE_JUDGE_LLM"],
        },
        report_path=args.report,
        sample=args.sample,
        threshold=float(
            os.environ.get("PLUMBLINE_ABSTAIN_THRESHOLD", DEFAULT_THRESHOLD)
        ),
    )
