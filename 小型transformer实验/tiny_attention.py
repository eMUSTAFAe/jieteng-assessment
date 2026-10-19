#!/usr/bin/env python3
"""Reference scaled dot-product attention with a causal mask.

The supplied ``transformer-baseline.py`` implements attention with Python loops
and a ``-1e9`` causal mask.  This module completes the same component with a
vectorised NumPy implementation and tests it against the supplied baseline:

1. output matches the baseline for ``causal=False`` and ``causal=True``;
2. with ``causal=True``, changing a future key/value does not change an
   earlier output;
3. with ``causal=False``, the same perturbation does change it.

Run:

    python tiny_attention.py

Writes ``results/attention_tests.json``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINE_PATH = ROOT / "transformer-baseline.py"


def _load_baseline() -> Any:
    spec = importlib.util.spec_from_file_location(
        "transformer_baseline", BASELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load transformer-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """Scaled dot-product attention.

    q: (T_q, d), k: (T_k, d), v: (T_k, d_v).
    causal=True masks key j for query i when j > i.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = q.shape[-1]
    scores = q @ k.T / math.sqrt(d)
    if causal:
        t_q, t_k = scores.shape
        future = np.triu(np.ones((t_q, t_k), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    return weights @ v


def run_tests(seed: int = 0, write: bool = True) -> dict[str, Any]:
    baseline = _load_baseline()
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(5, 6))
    k = rng.normal(size=(5, 6))
    v = rng.normal(size=(5, 3))

    base_nc = np.asarray(
        baseline.attention(q.tolist(), k.tolist(), v.tolist(), causal=False)
    )
    base_c = np.asarray(
        baseline.attention(q.tolist(), k.tolist(), v.tolist(), causal=True)
    )
    ours_nc = attention(q, k, v, causal=False)
    ours_c = attention(q, k, v, causal=True)
    diff_nc = float(np.max(np.abs(base_nc - ours_nc)))
    diff_c = float(np.max(np.abs(base_c - ours_c)))

    # Perturb a future key/value (position 4).  Queries 0..3 must not see it
    # under a causal mask, but must see it without a mask.
    k2 = k.copy()
    v2 = v.copy()
    k2[4] += 10.0
    v2[4] += 10.0
    causal_pert = attention(q, k2, v2, causal=True)
    noncausal_pert = attention(q, k2, v2, causal=False)
    causal_unchanged = bool(
        np.allclose(ours_c[:4], causal_pert[:4], atol=1e-12, rtol=0.0)
    )
    noncausal_changed = bool(
        not np.allclose(ours_nc[:4], noncausal_pert[:4], atol=1e-12, rtol=0.0)
    )

    result = {
        "seed": seed,
        "max_abs_diff_non_causal": diff_nc,
        "max_abs_diff_causal": diff_c,
        "matches_baseline_non_causal": bool(diff_nc < 1e-9),
        "matches_baseline_causal": bool(diff_c < 1e-9),
        "future_perturbation_ignored_when_causal": causal_unchanged,
        "future_perturbation_changes_output_when_non_causal": noncausal_changed,
        "passed": bool(
            diff_nc < 1e-9
            and diff_c < 1e-9
            and causal_unchanged
            and noncausal_changed
        ),
    }
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "attention_tests.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
