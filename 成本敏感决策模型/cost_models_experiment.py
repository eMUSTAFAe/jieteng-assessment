#!/usr/bin/env python3
"""Cost-sensitive decision models: logistic regression vs. a decision tree.

Scenario
--------
The task supplies a cost table that charges a false negative (FN) four times as
much as a false positive (FP).  We use the German Credit data set (UCI, 1000
rows) and define the positive class as "bad credit risk" (the applicant would
default).  Under that definition:

* FN = a bad applicant is approved  -> business cost 4
* FP = a good applicant is rejected -> business cost 1

The business cost table does not fully account for the harm borne by a
creditworthy applicant who is wrongly rejected, nor for the manual-review
workload imposed on staff who investigate flagged applications.  The analysis
therefore (a) reports FP counts explicitly and (b) tests how the recommendation
moves when an explicit extra FP cost is added.

Protocol
--------
* Stratified 60/20/20 train/dev/test split, fixed seed (42).
* Models: logistic regression (scaled numeric + one-hot categorical features)
  and a depth-limited decision tree.  Hyperparameters are pre-specified and are
  not tuned on the test set.
* Thresholds are selected on dev only, under the supplied 1:4 cost table.
* The test set is used only for reporting.  It is never used to choose a model,
  a threshold or a hyperparameter.
* Robustness: the whole split/fit/select/evaluate procedure is repeated over 11
  stratified splits to show split sensitivity.
* Sensitivity: thresholds are re-selected on dev under alternative FN:FP ratios
  and under an added FP cost representing the omitted stakeholder harm.

Run
---
    python cost_models_experiment.py

Outputs
-------
    results/report.txt
    results/main_results.json
    results/main_table.csv
    results/confusion_matrices.csv
    results/dev_cost_curves.csv
    results/multiseed_results.csv
    results/cost_ratio_sensitivity.csv
    results/fp_harm_sensitivity.csv

The supplied 4-row baseline is kept unchanged.  A sanity check at the start of
this script reproduces its cost calculations; it is a logic check, not empirical
evidence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "german.data"
RESULTS_DIR = ROOT / "results"

SEED = 42
MULTISEED_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
C_FP = 1.0
C_FN = 4.0
THRESHOLD_GRID = np.linspace(0.001, 0.999, 999)
BOOTSTRAP_N = 2000
FP_GUARD_TOLERANCE = 0.10          # allow up to +10% dev cost ...
FP_GUARD_MIN_REDUCTION = 0.25      # ... to cut FP count by at least 25%
FIXED_THRESHOLD = 0.50

COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_german_credit(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the UCI German Credit data (space-separated, no header).

    The file encodes the target as 1 = good, 2 = bad.  We predict "bad risk"
    as the positive class, so y = 1 means the applicant would default.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  curl -o data/german.data "
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "statlog/german/german.data"
        )
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    if df.shape != (1000, len(COLUMNS)):
        raise ValueError(f"Unexpected shape {df.shape}; expected (1000, 21)")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in German Credit data")
    y = (df["class"] == 2).astype(int).to_numpy()
    if not 0.25 < y.mean() < 0.35:
        raise ValueError(f"Unexpected positive rate {y.mean():.3f}")
    X = df.drop(columns=["class"])
    return X, y


def split_data(
    X: pd.DataFrame, y: np.ndarray, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 train/dev/test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    categorical = [c for c in COLUMNS[:-1] if c in {
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    }]
    numeric = [c for c in COLUMNS[:-1] if c not in categorical]
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        sparse_threshold=0.0,
    )


def make_models(seed: int) -> dict[str, Pipeline]:
    """Two pre-specified models; no test-set hyperparameter tuning."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "DecisionTree": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=20,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Cost and threshold selection (dev only)
# ---------------------------------------------------------------------------


def cost_curve(
    y: np.ndarray,
    probabilities: np.ndarray,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
    grid: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    """Expected cost per applicant for every threshold in the grid."""
    predicted_positive = probabilities[None, :] >= grid[:, None]
    fp = ((predicted_positive == 1) & (y[None, :] == 0)).sum(axis=1)
    fn = ((predicted_positive == 0) & (y[None, :] == 1)).sum(axis=1)
    cost = (c_fp * fp + c_fn * fn) / len(y)
    return pd.DataFrame(
        {
            "threshold": grid,
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "cost": cost.astype(float),
        }
    )


def select_cost_optimal(curve: pd.DataFrame) -> dict[str, float | int]:
    """Pick the lowest-cost threshold; break ties toward the higher threshold.

    A higher threshold predicts fewer positives, so among equally costly
    thresholds it is the one with fewer false positives.
    """
    minimum = curve["cost"].min()
    ties = curve[np.isclose(curve["cost"], minimum, rtol=0.0, atol=1e-12)]
    row = ties.iloc[-1]  # grid is ascending
    return {
        "threshold": float(row["threshold"]),
        "cost": float(row["cost"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
    }


def select_fp_guarded(
    curve: pd.DataFrame,
    tolerance: float = FP_GUARD_TOLERANCE,
) -> dict[str, float | int]:
    """Highest threshold whose dev cost is within `tolerance` of the minimum.

    This is a pre-registered decision rule: accept at most a 10% increase in
    expected cost in exchange for as few false positives as possible.
    """
    minimum = curve["cost"].min()
    allowed = curve[curve["cost"] <= minimum * (1.0 + tolerance) + 1e-12]
    row = allowed.iloc[-1]  # highest threshold within tolerance
    return {
        "threshold": float(row["threshold"]),
        "cost": float(row["cost"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
    }


def fp_guard_is_worthwhile(
    optimal: dict[str, float | int],
    guarded: dict[str, float | int],
    min_reduction: float = FP_GUARD_MIN_REDUCTION,
) -> bool:
    """Pre-registered: use the guard only if it cuts dev FP by >= 25%."""
    if optimal["fp"] == 0:
        return False
    return guarded["fp"] <= (1.0 - min_reduction) * optimal["fp"]


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "n": int(n),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "expected_cost": float((c_fp * fp + c_fn * fn) / n),
        "fp_per_1000": float(1000.0 * fp / n),
        "fn_per_1000": float(1000.0 * fn / n),
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "predicted_positive_rate": float((tp + fp) / n),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def bootstrap_cost_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> list[float]:
    """Percentile bootstrap CI for expected cost, conditional on fixed predictions.

    It captures test-sample sampling uncertainty only; it does not capture
    model-training or split uncertainty (the multi-seed experiment covers that).
    """
    rng = np.random.default_rng(seed)
    predicted = (probabilities >= threshold).astype(int)
    n = len(y_true)
    costs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = predicted[idx]
        fp = int(np.sum((pb == 1) & (yb == 0)))
        fn = int(np.sum((pb == 0) & (yb == 1)))
        costs[i] = (c_fp * fp + c_fn * fn) / n
    low, high = np.percentile(costs, [2.5, 97.5])
    return [float(low), float(high)]


def paired_bootstrap_cost_diff(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    threshold_a: float,
    probs_b: np.ndarray,
    threshold_b: float,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired bootstrap for cost(model A) - cost(model B) on the same test rows."""
    rng = np.random.default_rng(seed)
    pred_a = (probs_a >= threshold_a).astype(int)
    pred_b = (probs_b >= threshold_b).astype(int)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        cost_a = (
            c_fp * np.sum((pa == 1) & (yb == 0))
            + c_fn * np.sum((pa == 0) & (yb == 1))
        ) / n
        cost_b = (
            c_fp * np.sum((pb == 1) & (yb == 0))
            + c_fn * np.sum((pb == 0) & (yb == 1))
        ) / n
        diffs[i] = cost_a - cost_b
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": [float(low), float(high)],
        "prob_a_worse": float(np.mean(diffs > 0)),
    }


# ---------------------------------------------------------------------------
# Baseline sanity check
# ---------------------------------------------------------------------------


def baseline_sanity_check() -> dict[float, float]:
    """Reproduce the supplied 4-row baseline. Logic check only, not evidence."""
    rows = [(0.1, 0), (0.4, 1), (0.6, 0), (0.9, 1)]

    def evaluate_baseline(threshold: float, fp: float = 1.0, fn: float = 4.0) -> float:
        errors = [
            fp if p >= threshold and label == 0
            else fn if p < threshold and label == 1
            else 0
            for p, label in rows
        ]
        return sum(errors) / len(errors)

    got = {t: evaluate_baseline(t) for t in (0.2, 0.5, 0.8)}
    expected = {0.2: 0.25, 0.5: 1.25, 0.8: 1.0}
    if got != expected:
        raise AssertionError(f"Baseline logic check failed: {got} != {expected}")
    return got


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    model_name: str
    seed: int
    dev_curve: pd.DataFrame
    optimal: dict[str, float | int]
    guarded: dict[str, float | int]
    guard_worthwhile: bool
    final_threshold: float
    final_dev_cost: float
    final_dev_fp: int
    final_dev_fn: int
    dev_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_metrics_at_opt: dict[str, Any]
    test_metrics_at_guard: dict[str, Any]
    test_metrics_at_fixed: dict[str, Any]
    test_ci_final: list[float]
    test_ci_opt: list[float]


def fit_and_select(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    seed: int,
) -> FitResult:
    """Fit on train, select thresholds on dev. No test data is touched here."""
    model.fit(X_train, y_train)
    dev_probs = model.predict_proba(X_dev)[:, 1]
    curve = cost_curve(y_dev, dev_probs)
    optimal = select_cost_optimal(curve)
    guarded = select_fp_guarded(curve)
    worthwhile = fp_guard_is_worthwhile(optimal, guarded)
    final = guarded if worthwhile else optimal
    final_metrics = evaluate(y_dev, dev_probs, float(final["threshold"]))
    return FitResult(
        model_name=model_name,
        seed=seed,
        dev_curve=curve,
        optimal=optimal,
        guarded=guarded,
        guard_worthwhile=worthwhile,
        final_threshold=float(final["threshold"]),
        final_dev_cost=float(final["cost"]),
        final_dev_fp=int(final["fp"]),
        final_dev_fn=int(final["fn"]),
        dev_metrics=final_metrics,
        test_metrics={},          # filled by caller after test evaluation
        test_metrics_at_opt={},
        test_metrics_at_guard={},
        test_metrics_at_fixed={},
        test_ci_final=[],
        test_ci_opt=[],
    )


def evaluate_held_out(
    result: FitResult,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> FitResult:
    """One reporting pass over the held-out test set."""
    test_probs = model.predict_proba(X_test)[:, 1]
    result.test_metrics = evaluate(y_test, test_probs, result.final_threshold)
    result.test_metrics_at_opt = evaluate(
        y_test, test_probs, float(result.optimal["threshold"])
    )
    result.test_metrics_at_guard = evaluate(
        y_test, test_probs, float(result.guarded["threshold"])
    )
    result.test_metrics_at_fixed = evaluate(y_test, test_probs, FIXED_THRESHOLD)
    result.test_ci_final = bootstrap_cost_ci(
        y_test, test_probs, result.final_threshold
    )
    result.test_ci_opt = bootstrap_cost_ci(
        y_test, test_probs, float(result.optimal["threshold"])
    )
    return result


def run_main_experiment(
    X: pd.DataFrame, y: np.ndarray, seed: int = SEED
) -> tuple[dict[str, FitResult], dict[str, Pipeline], dict[str, np.ndarray], np.ndarray]:
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, seed)
    models = make_models(seed)
    results: dict[str, FitResult] = {}
    test_probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        fitted = fit_and_select(name, model, X_train, y_train, X_dev, y_dev, seed)
        fitted = evaluate_held_out(fitted, model, X_test, y_test)
        results[name] = fitted
        test_probs[name] = model.predict_proba(X_test)[:, 1]
    return results, models, test_probs, y_test


def choose_recommendation(results: dict[str, FitResult]) -> tuple[str, FitResult]:
    """Pick the model with lower dev cost at its cost-optimal threshold.

    Model selection uses the pure cost-sensitive criterion on dev.  The FP guard
    is applied afterwards to the selected model's threshold; it is a
    threshold-level value judgment, not a model-selection trick.

    The tie-break (within 0.001 expected cost) favours logistic regression,
    because its smoother probabilities make threshold transfer more stable.
    """
    ranked = sorted(
        results.items(),
        key=lambda item: (
            round(item[1].optimal["cost"], 3),
            0 if item[0] == "LogisticRegression" else 1,
        ),
    )
    return ranked[0]


def run_multiseed(
    X: pd.DataFrame, y: np.ndarray, seeds: list[int]
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, seed)
        models = make_models(seed)
        seed_rows = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            dev_probs = model.predict_proba(X_dev)[:, 1]
            curve = cost_curve(y_dev, dev_probs)
            optimal = select_cost_optimal(curve)
            guarded = select_fp_guarded(curve)
            worthwhile = fp_guard_is_worthwhile(optimal, guarded)
            final = guarded if worthwhile else optimal
            test_probs = model.predict_proba(X_test)[:, 1]
            test_metrics = evaluate(y_test, test_probs, float(final["threshold"]))
            seed_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "dev_cost_opt": optimal["cost"],
                    "threshold_opt": optimal["threshold"],
                    "dev_fp_opt": optimal["fp"],
                    "guard_worthwhile": worthwhile,
                    "threshold_final": final["threshold"],
                    "dev_cost_final": final["cost"],
                    "dev_fp_final": final["fp"],
                    "test_cost": test_metrics["expected_cost"],
                    "test_fp": test_metrics["fp"],
                    "test_fn": test_metrics["fn"],
                    "test_fp_per_1000": test_metrics["fp_per_1000"],
                }
            )
        # Model that would be chosen on dev for this split (pure cost criterion)
        best = min(
            seed_rows,
            key=lambda r: (r["dev_cost_opt"], r["model"] != "LogisticRegression"),
        )
        for row in seed_rows:
            row["chosen_on_dev"] = row["model"] == best["model"]
        rows.extend(seed_rows)
    return pd.DataFrame(rows)


def run_cost_ratio_sensitivity(
    X: pd.DataFrame,
    y: np.ndarray,
    ratios: list[float],
    seed: int = SEED,
) -> pd.DataFrame:
    """Re-select thresholds on dev for alternative FN:FP ratios; report test.

    Models are fitted once per split; only the decision rule changes.  The test
    set is still used for reporting only: every threshold comes from dev.
    """
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, seed)
    models = make_models(seed)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        dev_probs = model.predict_proba(X_dev)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        for ratio in ratios:
            curve = cost_curve(y_dev, dev_probs, c_fp=1.0, c_fn=ratio)
            selected = select_cost_optimal(curve)
            metrics = evaluate(
                y_test, test_probs, float(selected["threshold"]), c_fp=1.0, c_fn=ratio
            )
            rows.append(
                {
                    "cost_ratio_fn_to_fp": ratio,
                    "model": name,
                    "threshold": selected["threshold"],
                    "dev_cost": selected["cost"],
                    "dev_fp": selected["fp"],
                    "dev_fn": selected["fn"],
                    "test_cost": metrics["expected_cost"],
                    "test_fp": metrics["fp"],
                    "test_fn": metrics["fn"],
                    "test_fp_per_1000": metrics["fp_per_1000"],
                    "test_fn_per_1000": metrics["fn_per_1000"],
                }
            )
    return pd.DataFrame(rows)


def run_fp_harm_sensitivity(
    X: pd.DataFrame,
    y: np.ndarray,
    added_fp_costs: list[float],
    seed: int = SEED,
) -> pd.DataFrame:
    """Internalise an unmeasured FP harm: effective C_FP = 1 + s.

    For each s, the threshold is re-selected on dev under
    C_FP = 1 + s, C_FN = 4.  We report the resulting business cost
    (C_FP = 1) and the effective social cost (C_FP = 1 + s).
    """
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, seed)
    models = make_models(seed)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        dev_probs = model.predict_proba(X_dev)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        for added in added_fp_costs:
            effective_fp = 1.0 + added
            curve = cost_curve(y_dev, dev_probs, c_fp=effective_fp, c_fn=C_FN)
            selected = select_cost_optimal(curve)
            business = evaluate(
                y_test, test_probs, float(selected["threshold"]),
                c_fp=1.0, c_fn=C_FN,
            )
            social = evaluate(
                y_test, test_probs, float(selected["threshold"]),
                c_fp=effective_fp, c_fn=C_FN,
            )
            rows.append(
                {
                    "added_fp_cost_s": added,
                    "effective_fp_cost": effective_fp,
                    "model": name,
                    "threshold": selected["threshold"],
                    "dev_effective_cost": selected["cost"],
                    "test_business_cost": business["expected_cost"],
                    "test_social_cost": social["expected_cost"],
                    "test_fp": business["fp"],
                    "test_fn": business["fn"],
                    "test_fp_per_1000": business["fp_per_1000"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Tee:
    """Write the report to both stdout and results/report.txt."""

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


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    tee = Tee(RESULTS_DIR / "report.txt")
    sys.stdout = tee
    try:
        print("=" * 78)
        print("COST-SENSITIVE DECISION MODELS")
        print("Logistic regression vs. decision tree on German Credit (UCI)")
        print("=" * 78)
        print(f"Python      : {platform.python_version()}")
        print(f"scikit-learn: {sklearn.__version__}")
        print(f"numpy       : {np.__version__}")
        print(f"pandas      : {pd.__version__}")
        print(f"seed        : {SEED}")
        print(f"cost table  : C_FP={C_FP:g}, C_FN={C_FN:g}")

        print("\nBaseline logic check (supplied 4-row example):")
        print(f"  {baseline_sanity_check()}")
        print("  This is a logic check only; it is not empirical evidence.")

        X, y = load_german_credit(DATA_PATH)
        print("\nDataset")
        print("-------")
        print(f"  rows: {len(X)}, features: {X.shape[1]}, positive rate: {y.mean():.3f}")
        print(f"  positive class: bad credit risk (would default)")
        print(f"  source: {DATA_PATH.relative_to(ROOT)}")
        print(f"  sha256: {file_sha256(DATA_PATH)}")

        # ---- Main experiment -------------------------------------------------
        results, models, test_probs, y_test = run_main_experiment(X, y)
        lr = results["LogisticRegression"]
        dt = results["DecisionTree"]

        print("\nDev-set threshold selection (seed 42)")
        print("-------------------------------------")
        dev_rows = []
        for name, r in results.items():
            dev_rows.append(
                {
                    "model": name,
                    "rule": "cost-optimal",
                    "threshold": r.optimal["threshold"],
                    "dev_cost": r.optimal["cost"],
                    "dev_fp": r.optimal["fp"],
                    "dev_fn": r.optimal["fn"],
                }
            )
            dev_rows.append(
                {
                    "model": name,
                    "rule": "FP-guarded",
                    "threshold": r.guarded["threshold"],
                    "dev_cost": r.guarded["cost"],
                    "dev_fp": r.guarded["fp"],
                    "dev_fn": r.guarded["fn"],
                }
            )
        dev_table = pd.DataFrame(dev_rows)
        print_table("Threshold rules", dev_table)
        print(
            f"\nTheoretical Bayes threshold if probabilities were calibrated: "
            f"{C_FP / (C_FP + C_FN):.3f}"
        )
        print(
            "Pre-registered FP guard: highest threshold with dev cost <= "
            f"{FP_GUARD_TOLERANCE:.0%} above the minimum, used only if it cuts "
            f"dev FP by >= {FP_GUARD_MIN_REDUCTION:.0%}."
        )
        for name, r in results.items():
            print(
                f"  {name}: guard worthwhile = {r.guard_worthwhile}; "
                f"final threshold = {r.final_threshold:.3f}"
            )

        rec_name, rec = choose_recommendation(results)
        print(
            f"\nModel selected on dev: {rec_name} "
            f"(cost-optimal: dev cost {rec.optimal['cost']:.4f} "
            f"at threshold {rec.optimal['threshold']:.3f}; "
            f"final rule: dev cost {rec.final_dev_cost:.4f} "
            f"at threshold {rec.final_threshold:.3f})"
        )
        print(
            f"Test cost at the recommended rule: {rec.test_metrics['expected_cost']:.4f}"
        )

        # ---- Main held-out table --------------------------------------------
        main_rows = []
        for name, r in results.items():
            for rule, threshold, metrics in (
                ("cost-optimal (dev)", float(r.optimal["threshold"]), r.test_metrics_at_opt),
                ("FP-guarded (dev)", float(r.guarded["threshold"]), r.test_metrics_at_guard),
                ("final rule (dev)", float(r.final_threshold), r.test_metrics),
                ("fixed 0.50", FIXED_THRESHOLD, r.test_metrics_at_fixed),
            ):
                main_rows.append(
                    {
                        "model": name,
                        "threshold_rule": rule,
                        "threshold": threshold,
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tn": metrics["tn"],
                        "expected_cost": metrics["expected_cost"],
                        "fp_per_1000": metrics["fp_per_1000"],
                        "fn_per_1000": metrics["fn_per_1000"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "roc_auc": metrics["roc_auc"],
                    }
                )
        main_table = pd.DataFrame(main_rows)
        print_table(
            "Held-out test results (seed 42; test used for reporting only)",
            main_table,
        )

        # ---- Bootstrap CIs ---------------------------------------------------
        print("\nTest-set bootstrap 95% CIs (2000 resamples)")
        print("--------------------------------------------")
        for name, r in results.items():
            print(
                f"  {name} @ final threshold {r.final_threshold:.3f}: "
                f"cost {r.test_metrics['expected_cost']:.4f} "
                f"CI [{r.test_ci_final[0]:.4f}, {r.test_ci_final[1]:.4f}]"
            )
            print(
                f"  {name} @ cost-optimal {r.optimal['threshold']:.3f}: "
                f"cost {r.test_metrics_at_opt['expected_cost']:.4f} "
                f"CI [{r.test_ci_opt[0]:.4f}, {r.test_ci_opt[1]:.4f}]"
            )
        # The paired difference uses each model's final threshold.
        diff = paired_bootstrap_cost_diff(
            y_test,
            test_probs["LogisticRegression"],
            lr.final_threshold,
            test_probs["DecisionTree"],
            dt.final_threshold,
        )
        print(
            f"  Paired cost difference (LR - DT) at each model's final threshold: "
            f"{diff['mean_diff']:.4f}, "
            f"95% CI [{diff['ci95'][0]:.4f}, {diff['ci95'][1]:.4f}], "
            f"P(LR worse) = {diff['prob_a_worse']:.3f}"
        )

        # ---- Cost-ratio sensitivity -----------------------------------------
        ratios = [1.0, 2.0, 4.0, 8.0, 16.0]
        ratio_table = run_cost_ratio_sensitivity(X, y, ratios)
        print_table(
            "Sensitivity to the assumed FN:FP cost ratio (threshold re-selected on dev)",
            ratio_table,
        )

        # ---- FP-harm sensitivity --------------------------------------------
        added_costs = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        fp_harm_table = run_fp_harm_sensitivity(X, y, added_costs)
        print_table(
            "Sensitivity to an unmeasured FP harm (effective C_FP = 1 + s)",
            fp_harm_table,
        )

        # ---- Multi-seed robustness ------------------------------------------
        multiseed = run_multiseed(X, y, MULTISEED_SEEDS)
        summary = (
            multiseed.groupby("model")
            .agg(
                n_splits=("seed", "nunique"),
                mean_test_cost=("test_cost", "mean"),
                sd_test_cost=("test_cost", "std"),
                min_test_cost=("test_cost", "min"),
                max_test_cost=("test_cost", "max"),
                mean_threshold=("threshold_final", "mean"),
                min_threshold=("threshold_final", "min"),
                max_threshold=("threshold_final", "max"),
                mean_test_fp=("test_fp", "mean"),
                mean_test_fn=("test_fn", "mean"),
            )
            .reset_index()
        )
        print_table(
            f"Split sensitivity across {len(MULTISEED_SEEDS)} stratified splits",
            summary,
        )
        chosen_counts = (
            multiseed[multiseed["chosen_on_dev"]]
            .groupby("model")["seed"]
            .nunique()
            .to_dict()
        )
        print(f"  Models chosen on dev across splits: {chosen_counts}")
        if "LogisticRegression" in multiseed["model"].values:
            lr_wins = multiseed[
                (multiseed["model"] == "LogisticRegression")
            ].set_index("seed")["test_cost"]
            dt_wins = multiseed[
                (multiseed["model"] == "DecisionTree")
            ].set_index("seed")["test_cost"]
            common = lr_wins.index.intersection(dt_wins.index)
            lr_better = int((lr_wins.loc[common] < dt_wins.loc[common]).sum())
            print(
                f"  LR had lower test cost in {lr_better}/{len(common)} splits; "
                f"mean difference {float((lr_wins.loc[common] - dt_wins.loc[common]).mean()):.4f}"
            )

        # ---- Persist ---------------------------------------------------------
        main_table.to_csv(RESULTS_DIR / "main_table.csv", index=False)
        dev_table.to_csv(RESULTS_DIR / "dev_thresholds.csv", index=False)
        ratio_table.to_csv(RESULTS_DIR / "cost_ratio_sensitivity.csv", index=False)
        fp_harm_table.to_csv(RESULTS_DIR / "fp_harm_sensitivity.csv", index=False)
        multiseed.to_csv(RESULTS_DIR / "multiseed_results.csv", index=False)
        pd.concat(
            [
                r.dev_curve.assign(model=name)
                for name, r in results.items()
            ],
            ignore_index=True,
        ).to_csv(RESULTS_DIR / "dev_cost_curves.csv", index=False)

        confusion_rows = []
        for name, r in results.items():
            for rule, metrics in (
                ("final", r.test_metrics),
                ("cost_optimal", r.test_metrics_at_opt),
                ("fp_guarded", r.test_metrics_at_guard),
                ("fixed_0.50", r.test_metrics_at_fixed),
            ):
                row = {"model": name, "rule": rule}
                row.update(metrics)
                confusion_rows.append(row)
        pd.DataFrame(confusion_rows).to_csv(
            RESULTS_DIR / "confusion_matrices.csv", index=False
        )

        summary_json = {
            "dataset": {
                "name": "German Credit (UCI Statlog)",
                "path": str(DATA_PATH.relative_to(ROOT)),
                "sha256": file_sha256(DATA_PATH),
                "n_rows": int(len(X)),
                "n_features": int(X.shape[1]),
                "positive_class": "bad credit risk (would default)",
                "positive_rate": float(y.mean()),
            },
            "protocol": {
                "split": "stratified 60/20/20 train/dev/test",
                "main_seed": SEED,
                "multiseed_seeds": MULTISEED_SEEDS,
                "cost_table": {"c_fp": C_FP, "c_fn": C_FN},
                "threshold_grid": {
                    "min": float(THRESHOLD_GRID[0]),
                    "max": float(THRESHOLD_GRID[-1]),
                    "n": int(len(THRESHOLD_GRID)),
                },
                "fp_guard_tolerance": FP_GUARD_TOLERANCE,
                "fp_guard_min_reduction": FP_GUARD_MIN_REDUCTION,
                "fixed_threshold": FIXED_THRESHOLD,
                "model_hyperparameters": {
                    "LogisticRegression": {"max_iter": 5000, "solver": "lbfgs"},
                    "DecisionTree": {"max_depth": 4, "min_samples_leaf": 20},
                },
            },
            "main_split": {
                name: {
                    "dev_cost_optimal": r.optimal,
                    "dev_fp_guarded": r.guarded,
                    "guard_worthwhile": r.guard_worthwhile,
                    "final_threshold": r.final_threshold,
                    "final_dev_cost": r.final_dev_cost,
                    "test_final": r.test_metrics,
                    "test_cost_optimal": r.test_metrics_at_opt,
                    "test_fixed_0.50": r.test_metrics_at_fixed,
                    "test_ci_final": r.test_ci_final,
                    "test_ci_optimal": r.test_ci_opt,
                }
                for name, r in results.items()
            },
            "pairwise_test_cost_diff_LR_minus_DT": diff,
            "recommendation": {
                "model": rec_name,
                "threshold": rec.final_threshold,
                "dev_cost": rec.final_dev_cost,
                "dev_cost_optimal": rec.optimal["cost"],
                "dev_threshold_optimal": rec.optimal["threshold"],
                "dev_threshold_guarded": rec.guarded["threshold"],
                "fp_guard_applied": rec.guard_worthwhile,
                "test_cost": rec.test_metrics["expected_cost"],
                "test_confusion": {
                    "tp": rec.test_metrics["tp"],
                    "fp": rec.test_metrics["fp"],
                    "fn": rec.test_metrics["fn"],
                    "tn": rec.test_metrics["tn"],
                },
                "selection_rule": (
                    "Model chosen by lower dev expected cost at the cost-optimal "
                    "threshold.  The selected model's threshold is then set by the "
                    "pre-registered FP guard: use the highest threshold within 10% "
                    "of the minimum dev cost if it cuts dev FP by >=25%.  Test is "
                    "not used for selection."
                ),
            },
        }
        with (RESULTS_DIR / "main_results.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_json, handle, indent=2)

        print("\nSaved:")
        for name in sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_file()):
            print(f"  results/{name}")
        print("\nDone.")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
