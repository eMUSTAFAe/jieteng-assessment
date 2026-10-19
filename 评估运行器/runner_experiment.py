#!/usr/bin/env python3
"""Repeatable adapter evaluation with visible failures, retries and costs.

Scenario
--------
A team wants a clear adapter ranking.  Adapters fail in different ways:

* ``fast``        - the supplied baseline adapter: always correct, fast;
* ``transient``   - correct when it completes, but fails 35% of attempts with a
                    deterministic pseudo-random exception; retries can help;
* ``slow``        - correct, but sleeps 10 ms on short inputs and 80 ms on long
                    inputs; the 50 ms per-attempt timeout kills long inputs;
* ``truncating``  - silently truncates outputs to 8 characters; short inputs are
                    correct, long inputs are wrong without any error.

The runner records every attempt, keeps final failures visible, charges each
attempt and each timeout, and compares reporting policies:

1. exclude every failed run (the most misleading policy);
2. drop timeouts/errors but score wrong outputs (completed-only);
3. count every run (failure-inclusive);
4. failure-inclusive accuracy plus cost per correct answer.

The decision is not "which adapter is fastest" but "what may honestly be
ranked, what must be qualified, and what should not be ranked at all".

Run
---
    python runner_experiment.py
    python make_figures.py

Outputs are written to ``results/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
TIMEOUT_S = 0.050          # per-attempt timeout
MAX_RETRIES = 3            # attempts = MAX_RETRIES + 1
REPEATS = 5                # repeats per case
COST_PER_ATTEMPT = 1.0
TIMEOUT_PENALTY = 0.5      # extra cost for an attempt that times out
BOOTSTRAP_N = 2000
RANK_BUDGET = MAX_RETRIES  # primary ranking uses the full retry budget
RECOMMENDED_BUDGET = 2     # discussed in the journal, not hard-coded into the ranking

# 40 small cases.  Length is deliberately mixed: 26 cases are <= 8 characters
# and 14 are longer.  ``slow`` and ``truncating`` are therefore correct on the
# same 26 cases, which makes their accuracy identical while their failure modes
# are completely different.
CASES: list[tuple[str, str]] = [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST"),
    ("a", "A"),
    ("", ""),
    ("go", "GO"),
    ("run", "RUN"),
    ("model", "MODEL"),
    ("tiny", "TINY"),
    ("data", "DATA"),
    ("python", "PYTHON"),
    ("runner", "RUNNER"),
    ("adapter", "ADAPTER"),
    ("latency", "LATENCY"),
    ("timeout", "TIMEOUT"),
    ("retry", "RETRY"),
    ("failure", "FAILURE"),
    ("cost", "COST"),
    ("evaluation", "EVALUATION"),
    ("reproducible", "REPRODUCIBLE"),
    ("hello world", "HELLO WORLD"),
    ("tiny model", "TINY MODEL"),
    ("evaluation runner", "EVALUATION RUNNER"),
    ("adapter ranking", "ADAPTER RANKING"),
    ("keep failures visible", "KEEP FAILURES VISIBLE"),
    ("straße", "STRASSE"),
    ("mañana", "MAÑANA"),
    ("café", "CAFÉ"),
    ("naïve", "NAÏVE"),
    ("Ångström", "ÅNGSTRÖM"),
    ("a b c", "A B C"),
    ("  spaced  ", "  SPACED  "),
    ("12345", "12345"),
    ("mixedCase 42", "MIXEDCASE 42"),
    ("x" * 10, "X" * 10),
    ("y" * 12, "Y" * 12),
    ("z" * 16, "Z" * 16),
    ("ab" * 10, "AB" * 10),
    ("short", "SHORT"),
    ("longer phrase here", "LONGER PHRASE HERE"),
]
CASE_IDS = [f"C{i:02d}" for i in range(len(CASES))]

ADAPTER_NAMES = ["fast", "transient", "slow", "truncating"]
ADAPTER_DESCRIPTIONS = {
    "fast": "Supplied baseline: correct, fast",
    "transient": "Correct when it completes; 35% deterministic transient errors",
    "slow": "Correct; 10 ms short / 80 ms long; 50 ms timeout",
    "truncating": "Silently truncates to 8 chars; long inputs are wrong",
}


# ---------------------------------------------------------------------------
# Deterministic pseudo-randomness
# ---------------------------------------------------------------------------


def _u01(adapter: str, case_id: str, repeat: int, attempt: int) -> float:
    """Stable uniform draw in [0, 1), independent of Python's hash seed."""
    payload = f"{SEED}|{adapter}|{case_id}|{repeat}|{attempt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def adapter_call(adapter: str, text: str, case_id: str, repeat: int, attempt: int) -> str:
    if adapter == "fast":
        return text.upper()
    if adapter == "transient":
        if _u01(adapter, case_id, repeat, attempt) < 0.35:
            raise RuntimeError("deterministic transient failure")
        return text.upper()
    if adapter == "slow":
        time.sleep(0.010 if len(text) <= 8 else 0.080)
        return text.upper()
    if adapter == "truncating":
        return text.upper()[:8]
    raise ValueError(f"unknown adapter {adapter}")


# ---------------------------------------------------------------------------
# Attempt and run records
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    adapter: str
    case_id: str
    input_text: str
    expected: str
    repeat: int
    attempt: int
    output: str | None
    correct: bool
    outcome: str          # correct | wrong | timeout | error
    detail: str           # exception type/message, or empty
    latency_ms: float
    cost: float


def run_attempt(
    adapter: str,
    case_id: str,
    text: str,
    expected: str,
    repeat: int,
    attempt: int,
    timeout_s: float = TIMEOUT_S,
) -> Attempt:
    """Run one attempt with a wall-clock timeout.

    Python cannot forcibly kill a thread.  The join timeout is therefore a
    reporting boundary, not a hard cancellation: an abandoned adapter call may
    finish in the background.  The timeout penalty in the cost model reflects
    that the capacity was still occupied.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = adapter_call(adapter, text, case_id, repeat, attempt)
        except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    start = time.perf_counter()
    thread.start()
    thread.join(timeout_s)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if thread.is_alive():
        return Attempt(
            adapter=adapter,
            case_id=case_id,
            input_text=text,
            expected=expected,
            repeat=repeat,
            attempt=attempt,
            output=None,
            correct=False,
            outcome="timeout",
            detail=f"exceeded {timeout_s * 1000:.0f} ms",
            latency_ms=latency_ms,
            cost=COST_PER_ATTEMPT + TIMEOUT_PENALTY,
        )
    if "error" in box:
        exc = box["error"]
        return Attempt(
            adapter=adapter,
            case_id=case_id,
            input_text=text,
            expected=expected,
            repeat=repeat,
            attempt=attempt,
            output=None,
            correct=False,
            outcome="error",
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            cost=COST_PER_ATTEMPT,
        )

    output = box.get("value")
    correct = output == expected
    return Attempt(
        adapter=adapter,
        case_id=case_id,
        input_text=text,
        expected=expected,
        repeat=repeat,
        attempt=attempt,
        output=output,
        correct=correct,
        outcome="correct" if correct else "wrong",
        detail="",
        latency_ms=latency_ms,
        cost=COST_PER_ATTEMPT,
    )


def run_base(
    adapter: str,
    case_id: str,
    text: str,
    expected: str,
    repeat: int,
    max_retries: int = MAX_RETRIES,
) -> list[Attempt]:
    """Run one base case, stopping on success or on a silently wrong output.

    Retry policy: retry only timeouts and exceptions.  A wrong output is not
    retried, because a deterministic wrong answer would just burn budget and
    could hide a quality problem behind a lucky retry.
    """
    attempts: list[Attempt] = []
    for attempt in range(max_retries + 1):
        outcome = run_attempt(adapter, case_id, text, expected, repeat, attempt)
        attempts.append(outcome)
        if outcome.correct:
            break
        if outcome.outcome == "wrong":
            break
    return attempts


def budget_row(
    attempts: list[Attempt], budget: int
) -> dict[str, Any]:
    """Derive the outcome at retry budget ``budget`` from the attempt record."""
    used = min(len(attempts), budget + 1)
    prefix = attempts[:used]
    correct = any(a.correct for a in prefix)
    last = prefix[-1]
    completed = last.outcome in {"correct", "wrong"}
    return {
        "adapter": last.adapter,
        "case_id": last.case_id,
        "repeat": last.repeat,
        "budget": budget,
        "attempts_used": used,
        "correct": correct,
        "outcome": "correct" if correct else last.outcome,
        "completed": completed,
        "timeouts_in_prefix": sum(1 for a in prefix if a.outcome == "timeout"),
        "cost": sum(a.cost for a in prefix),
        "latency_ms": sum(a.latency_ms for a in prefix),
        "input_text": last.input_text,
        "expected": last.expected,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize(budget_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (adapter, budget), group in budget_df.groupby(["adapter", "budget"]):
        runs = len(group)
        correct = int(group["correct"].sum())
        completed = int(group["completed"].sum())
        wrong = int((group["outcome"] == "wrong").sum())
        timeouts = int((group["outcome"] == "timeout").sum())
        errors = int((group["outcome"] == "error").sum())
        total_cost = float(group["cost"].sum())
        rows.append(
            {
                "adapter": adapter,
                "budget": int(budget),
                "runs": runs,
                "correct_runs": correct,
                "completed_runs": completed,
                "wrong_runs": wrong,
                "timeout_runs": timeouts,
                "error_runs": errors,
                "accuracy_all": correct / runs,
                "accuracy_completed": correct / completed if completed else None,
                "accuracy_correct_only": 1.0 if correct else None,
                "retained_fraction": correct / runs,
                "mean_attempts": float(group["attempts_used"].mean()),
                "total_cost": total_cost,
                "cost_per_run": total_cost / runs,
                "cost_per_correct": total_cost / correct if correct else None,
                "latency_p50_all_ms": pct(group["latency_ms"].tolist(), 50),
                "latency_p95_all_ms": pct(group["latency_ms"].tolist(), 95),
                "latency_p50_completed_ms": pct(
                    group.loc[group["completed"], "latency_ms"].tolist(), 50
                ),
                "latency_p95_completed_ms": pct(
                    group.loc[group["completed"], "latency_ms"].tolist(), 95
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget", "adapter"]).reset_index(drop=True)


def rank_with_ties(
    frame: pd.DataFrame, metric: str, ascending: bool
) -> pd.Series:
    """Competition ranking (1,2,2,4) so ties are visible."""
    values = frame[metric].to_numpy(dtype=float)
    ranks = np.empty(len(values), dtype=int)
    order = np.argsort(values if ascending else -values, kind="mergesort")
    sorted_values = values[order]
    current_rank = 1
    for position, idx in enumerate(order):
        if position > 0 and not np.isclose(
            sorted_values[position], sorted_values[position - 1], rtol=1e-12, atol=1e-12
        ):
            current_rank = position + 1
        ranks[idx] = current_rank
    return pd.Series(ranks, index=frame.index)


def ranking_table(summary: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Four reporting lenses at a fixed retry budget."""
    frame = summary[summary["budget"] == budget].copy().reset_index(drop=True)
    frame["rank_exclude_all_failures"] = rank_with_ties(
        frame, "accuracy_correct_only", ascending=False
    )
    frame["rank_completed_only"] = rank_with_ties(
        frame, "accuracy_completed", ascending=False
    )
    frame["rank_failure_inclusive"] = rank_with_ties(
        frame, "accuracy_all", ascending=False
    )
    frame["rank_cost_per_correct"] = rank_with_ties(
        frame, "cost_per_correct", ascending=True
    )
    return frame[
        [
            "adapter",
            "runs",
            "correct_runs",
            "completed_runs",
            "accuracy_all",
            "accuracy_completed",
            "retained_fraction",
            "cost_per_correct",
            "latency_p95_all_ms",
            "latency_p95_completed_ms",
            "rank_exclude_all_failures",
            "rank_completed_only",
            "rank_failure_inclusive",
            "rank_cost_per_correct",
        ]
    ]


# ---------------------------------------------------------------------------
# Bootstrap: case-level clustered, with repeated runs kept together
# ---------------------------------------------------------------------------


def bootstrap_metrics(budget_df: pd.DataFrame, budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = budget_df[budget_df["budget"] == budget].copy()
    case_ids = sorted(frame["case_id"].unique())
    case_index = {c: i for i, c in enumerate(case_ids)}
    adapters = sorted(frame["adapter"].unique())
    n_cases = len(case_ids)

    per_adapter: dict[str, dict[str, np.ndarray]] = {}
    for adapter in adapters:
        sub = frame[frame["adapter"] == adapter]
        correct = np.zeros(n_cases)
        runs = np.zeros(n_cases)
        cost = np.zeros(n_cases)
        for row in sub.itertuples(index=False):
            i = case_index[row.case_id]
            correct[i] += float(row.correct)
            runs[i] += 1.0
            cost[i] += float(row.cost)
        per_adapter[adapter] = {"correct": correct, "runs": runs, "cost": cost}

    rng = np.random.default_rng(SEED)
    acc_samples = {a: np.empty(BOOTSTRAP_N) for a in adapters}
    cost_samples = {a: np.empty(BOOTSTRAP_N) for a in adapters}
    ranks = {a: np.zeros(len(adapters), dtype=int) for a in adapters}

    for b in range(BOOTSTRAP_N):
        sample = rng.integers(0, n_cases, size=n_cases)
        accs: dict[str, float] = {}
        costs: dict[str, float] = {}
        for adapter in adapters:
            arrays = per_adapter[adapter]
            c = arrays["correct"][sample].sum()
            r = arrays["runs"][sample].sum()
            k = arrays["cost"][sample].sum()
            accs[adapter] = c / r if r else np.nan
            costs[adapter] = k / c if c else np.inf
            acc_samples[adapter][b] = accs[adapter]
            cost_samples[adapter][b] = costs[adapter]
        order = sorted(adapters, key=lambda a: (-accs[a], costs[a]))
        for position, adapter in enumerate(order, start=1):
            ranks[adapter][position - 1] += 1

    rows = []
    rank_rows = []
    for adapter in adapters:
        acc = acc_samples[adapter]
        cost = cost_samples[adapter]
        rows.append(
            {
                "adapter": adapter,
                "accuracy_mean": float(np.nanmean(acc)),
                "accuracy_ci_low": float(np.nanpercentile(acc, 2.5)),
                "accuracy_ci_high": float(np.nanpercentile(acc, 97.5)),
                "cost_per_correct_mean": float(np.nanmean(cost)),
                "cost_ci_low": float(np.nanpercentile(cost, 2.5)),
                "cost_ci_high": float(np.nanpercentile(cost, 97.5)),
            }
        )
        rank_rows.append(
            {
                "adapter": adapter,
                **{
                    f"rank_{i + 1}_prob": float(ranks[adapter][i] / BOOTSTRAP_N)
                    for i in range(len(adapters))
                },
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------


def cost_sensitivity(budget_df: pd.DataFrame, budget: int) -> pd.DataFrame:
    frame = budget_df[budget_df["budget"] == budget].copy()
    rows = []
    for penalty in (0.0, 0.5, 1.5):
        frame["_cost"] = frame["attempts_used"] + penalty * frame["timeouts_in_prefix"]
        for adapter, group in frame.groupby("adapter"):
            correct = int(group["correct"].sum())
            rows.append(
                {
                    "timeout_penalty": penalty,
                    "adapter": adapter,
                    "total_cost": float(group["_cost"].sum()),
                    "cost_per_correct": float(group["_cost"].sum() / correct)
                    if correct
                    else None,
                    "accuracy_all": correct / len(group),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Repeatability check
# ---------------------------------------------------------------------------


def repeatability_check() -> dict[str, Any]:
    """Run a small subset twice and compare every deterministic outcome field."""
    def one_run() -> list[tuple]:
        rows = []
        for adapter in ADAPTER_NAMES:
            for i, (text, expected) in enumerate(CASES[:5]):
                for repeat in range(2):
                    attempts = run_base(adapter, CASE_IDS[i], text, expected, repeat, max_retries=2)
                    for a in attempts:
                        rows.append(
                            (
                                a.adapter,
                                a.case_id,
                                a.repeat,
                                a.attempt,
                                a.correct,
                                a.outcome,
                                a.detail,
                            )
                        )
        return rows

    first = one_run()
    second = one_run()
    same = first == second
    return {
        "passed": same,
        "records": len(first),
        "checksum": hashlib.sha256(repr(first).encode("utf-8")).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def print_table(title: str, frame: pd.DataFrame, float_format: str = "{:.3f}") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda x: "" if pd.isna(x) else float_format.format(x)
            )
    print(formatted.to_string(index=False))


def main() -> None:
    tee = Tee(RESULTS / "report.txt")
    sys.stdout = tee
    try:
        print("=" * 78)
        print("EVALUATION RUNNER: ADAPTER RANKING WITH VISIBLE FAILURES")
        print("=" * 78)
        print(f"Python   : {platform.python_version()}")
        print(f"numpy    : {np.__version__}")
        print(f"pandas   : {pd.__version__}")
        print(f"seed     : {SEED}")
        print(f"cases    : {len(CASES)} ({sum(1 for t, _ in CASES if len(t) <= 8)} short, "
              f"{sum(1 for t, _ in CASES if len(t) > 8)} long)")
        print(f"repeats  : {REPEATS}")
        print(f"timeout  : {TIMEOUT_S * 1000:.0f} ms per attempt")
        print(f"retries  : 0..{MAX_RETRIES}")
        print(f"cost     : {COST_PER_ATTEMPT:g} per attempt + "
              f"{TIMEOUT_PENALTY:g} per timeout")
        print("\nAdapters")
        print("--------")
        for name in ADAPTER_NAMES:
            print(f"  {name:<11} {ADAPTER_DESCRIPTIONS[name]}")

        check = repeatability_check()
        print("\nRepeatability check")
        print("-------------------")
        print(f"  passed: {check['passed']}  records: {check['records']}  "
              f"checksum: {check['checksum']}")

        # ---- Run the experiment ---------------------------------------------
        per_example: list[Attempt] = []
        base_rows: list[dict[str, Any]] = []
        budget_rows: list[dict[str, Any]] = []
        start = time.perf_counter()
        for adapter in ADAPTER_NAMES:
            for i, (text, expected) in enumerate(CASES):
                for repeat in range(REPEATS):
                    attempts = run_base(
                        adapter, CASE_IDS[i], text, expected, repeat, MAX_RETRIES
                    )
                    per_example.extend(attempts)
                    final_correct = any(a.correct for a in attempts)
                    base_rows.append(
                        {
                            "adapter": adapter,
                            "case_id": CASE_IDS[i],
                            "repeat": repeat,
                            "input_text": text,
                            "expected": expected,
                            "attempts_used": len(attempts),
                            "first_attempt_correct": attempts[0].correct,
                            "final_correct": final_correct,
                            "final_outcome": "correct"
                            if final_correct
                            else attempts[-1].outcome,
                            "final_detail": "" if final_correct else attempts[-1].detail,
                            "attempts_to_success": next(
                                (a.attempt + 1 for a in attempts if a.correct), None
                            ),
                            "total_cost": sum(a.cost for a in attempts),
                            "total_latency_ms": sum(a.latency_ms for a in attempts),
                        }
                    )
                    for budget in range(MAX_RETRIES + 1):
                        budget_rows.append(budget_row(attempts, budget))
        elapsed = time.perf_counter() - start

        per_example_df = pd.DataFrame([asdict(a) for a in per_example])
        base_df = pd.DataFrame(base_rows)
        budget_df = pd.DataFrame(budget_rows)
        summary_df = summarize(budget_df)

        print(f"\nExecuted {len(per_example_df)} attempts in {elapsed:.1f} s wall-clock.")
        print(f"Base runs: {len(base_df)}; per-example records: {len(per_example_df)}.")

        # ---- Headline: the exclusion bias -----------------------------------
        headline = summary_df[summary_df["budget"] == RANK_BUDGET].copy()
        headline = headline[
            [
                "adapter",
                "runs",
                "correct_runs",
                "completed_runs",
                "wrong_runs",
                "timeout_runs",
                "error_runs",
                "accuracy_all",
                "accuracy_completed",
                "retained_fraction",
                "mean_attempts",
                "cost_per_run",
                "cost_per_correct",
                "latency_p95_all_ms",
                "latency_p95_completed_ms",
            ]
        ].sort_values("accuracy_all", ascending=False).reset_index(drop=True)
        print_table(
            f"Adapter summary at retry budget {RANK_BUDGET} "
            f"(all failures counted)",
            headline,
        )
        print(
            "\nRead this table carefully: `accuracy_completed` drops timeouts and\n"
            "errors; `retained_fraction` is the share of runs that still count.\n"
            "The slow adapter looks perfect on completed runs but only 65% of its\n"
            "runs completed.  The truncating adapter never errors, so it keeps all\n"
            "of its runs, but its silent wrong answers are scored as wrong."
        )

        # ---- Retry curves ----------------------------------------------------
        retry_view = summary_df[
            [
                "adapter",
                "budget",
                "accuracy_all",
                "accuracy_completed",
                "mean_attempts",
                "cost_per_correct",
                "timeout_runs",
                "wrong_runs",
                "error_runs",
            ]
        ].sort_values(["adapter", "budget"])
        print_table("Retry budget tradeoff (all adapters)", retry_view)

        # ---- Ranking lenses --------------------------------------------------
        ranks = ranking_table(summary_df, RANK_BUDGET)
        print_table(
            f"Reporting lenses at retry budget {RANK_BUDGET}",
            ranks,
            float_format="{:.3f}",
        )
        print(
            "\nThe first lens excludes every failed run: all adapters reach 100%\n"
            "and the ranking becomes uninformative.  The second drops timeouts and\n"
            "errors but scores wrong outputs; the slow adapter still looks perfect.\n"
            "The third counts every run.  The fourth is a cost lens; it must not be\n"
            "used alone, because a cheap silent-wrong adapter can look competitive."
        )

        # ---- Bootstrap -------------------------------------------------------
        boot_metrics, boot_ranks = bootstrap_metrics(budget_df, RANK_BUDGET)
        print_table(
            f"Case-level clustered bootstrap ({BOOTSTRAP_N} resamples, "
            f"budget {RANK_BUDGET})",
            boot_metrics,
        )
        print_table("Bootstrap rank probabilities", boot_ranks)
        top_acc = boot_metrics.sort_values("accuracy_mean", ascending=False)
        print(
            "\nPractical reading: the fast/transient pair is not separated by the\n"
            "bootstrap interval, and slow/truncating are exactly tied on accuracy\n"
            "by construction (both are correct on the same 26 short cases).  A\n"
            "single strict ranking is therefore not justified; the report uses\n"
            "tiers and qualifies the lower pair by failure mode and cost."
        )
        _ = top_acc

        # ---- Cost sensitivity ------------------------------------------------
        cost_sens = cost_sensitivity(budget_df, RANK_BUDGET)
        print_table(
            "Cost sensitivity: timeout penalty (attempt cost fixed at 1)",
            cost_sens,
        )

        # ---- Failure cases ---------------------------------------------------
        failures = base_df[~base_df["final_correct"]].copy()
        failures = failures.sort_values(["adapter", "case_id", "repeat"])
        print(f"\nFinal failure cases kept visible: {len(failures)} of {len(base_df)} "
              f"({len(failures) / len(base_df):.1%})")
        if len(failures):
            print_table(
                "First 12 final failures",
                failures[
                    [
                        "adapter",
                        "case_id",
                        "repeat",
                        "input_text",
                        "expected",
                        "attempts_used",
                        "final_outcome",
                        "final_detail",
                        "total_cost",
                    ]
                ].head(12),
            )

        # ---- Persist ---------------------------------------------------------
        per_example_df.to_csv(RESULTS / "per_example.csv", index=False)
        base_df.to_csv(RESULTS / "run_outcomes.csv", index=False)
        budget_df.to_csv(RESULTS / "budget_outcomes.csv", index=False)
        summary_df.to_csv(RESULTS / "adapter_summary.csv", index=False)
        ranks.to_csv(RESULTS / "ranking_policies.csv", index=False)
        boot_metrics.to_csv(RESULTS / "bootstrap_metrics.csv", index=False)
        boot_ranks.to_csv(RESULTS / "bootstrap_ranks.csv", index=False)
        cost_sens.to_csv(RESULTS / "cost_sensitivity.csv", index=False)
        failures.to_csv(RESULTS / "failure_cases.csv", index=False)

        metadata = {
            "seed": SEED,
            "cases": len(CASES),
            "short_cases": sum(1 for t, _ in CASES if len(t) <= 8),
            "long_cases": sum(1 for t, _ in CASES if len(t) > 8),
            "repeats": REPEATS,
            "timeout_ms": TIMEOUT_S * 1000,
            "max_retries": MAX_RETRIES,
            "cost_per_attempt": COST_PER_ATTEMPT,
            "timeout_penalty": TIMEOUT_PENALTY,
            "bootstrap_n": BOOTSTRAP_N,
            "repeatability": check,
            "attempts": len(per_example_df),
            "base_runs": len(base_df),
        }
        with (RESULTS / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        print("\nSaved:")
        for name in sorted(p.name for p in RESULTS.iterdir() if p.is_file()):
            print(f"  results/{name}")
        print("\nDone.")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
