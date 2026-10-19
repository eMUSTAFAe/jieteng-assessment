#!/usr/bin/env python3
"""Measure the recommended combined policy.

The main comparison isolates components: bounded concurrency, batching and
priority.  The release recommendation combines them.  This script measures that
combination directly:

    priority_batched = 8 workers, queue 64, batch 8 / 10 ms wait, priority

It runs the same three scenarios and repeats as the main experiment and writes
``results/combined_policy.csv``.  It is deliberately separate so the main
component comparison stays five configurations.

Run:

    python combined_policy.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import service_experiment as se

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG_NAME = "priority_batched"


def main() -> None:
    config = se.CONFIGS[CONFIG_NAME]
    rows: list[dict] = []
    server_rows: list[dict] = []

    print(f"Measuring {CONFIG_NAME}: workers={config.max_workers}, "
          f"queue={config.max_queue}, batch={config.batch_size}, "
          f"priority={config.priority}")

    for scenario in se.SCENARIO_ORDER:
        for repeat in range(se.REPEATS):
            with se.RemoteService(CONFIG_NAME) as remote:
                records, elapsed = se.run_load(
                    CONFIG_NAME, remote.url, scenario, repeat, None
                )
                time.sleep(0.1)
            server_records = remote.records
            for record in records:
                record["repeat"] = repeat
            for server_record in server_records:
                server_record["config"] = CONFIG_NAME
                server_record["scenario"] = scenario
                server_record["repeat"] = repeat
            server_rows.extend(server_records)
            rows.append(
                se.summarize_run(
                    records, elapsed, CONFIG_NAME, scenario, repeat, server_records
                )
            )
            print(f"  {scenario} repeat {repeat}: "
                  f"success={rows[-1]['successes']}/{rows[-1]['n']}, "
                  f"goodput={rows[-1]['goodput_rps']:.1f} rps, "
                  f"p95={rows[-1]['latency_p95_all_ms']:.1f} ms")

    runs = pd.DataFrame(rows)
    numeric = [
        c for c in runs.columns
        if c not in {"config", "scenario", "repeat"}
        and pd.api.types.is_numeric_dtype(runs[c])
    ]
    mean = runs.groupby(["config", "scenario"], sort=False)[numeric].mean().reset_index()
    sd = runs.groupby(["config", "scenario"], sort=False)[numeric].std().reset_index()
    sd = sd.rename(columns={c: f"{c}_sd" for c in numeric})
    summary = mean.merge(sd, on=["config", "scenario"], how="left")

    runs.to_csv(RESULTS / "combined_policy_runs.csv", index=False)
    summary.to_csv(RESULTS / "combined_policy.csv", index=False)
    pd.DataFrame(server_rows).to_csv(
        RESULTS / "combined_policy_server_requests.csv", index=False
    )

    display = [
        "scenario", "n", "successes", "error_rate", "throughput_rps",
        "goodput_rps", "latency_p50_all_ms", "latency_p95_all_ms",
        "latency_p99_all_ms", "latency_p95_ok_ms", "status_429", "status_504",
        "server_mean_queue_ms", "server_p95_queue_ms", "mean_batch_size",
        "interactive_p95_all_ms", "batch_p95_all_ms",
        "interactive_error_rate", "batch_error_rate",
    ]
    print("\nCombined policy summary (mean of repeats)")
    print("-----------------------------------------")
    print(summary[display].to_string(index=False))

    # The combined policy uses the same handler as the other bounded configs;
    # run the validation cases once to confirm the 400 path is unchanged.
    with se.RemoteService(CONFIG_NAME) as remote:
        validation = se.validation_test(CONFIG_NAME, remote.url)
    pd.DataFrame(validation).to_csv(
        RESULTS / "combined_policy_validation.csv", index=False
    )
    print("\nValidation statuses:",
          {row["case"]: row["status"] for row in validation})

    metadata = {
        "config": config.__dict__,
        "scenarios": se.SCENARIOS,
        "repeats": se.REPEATS,
    }
    (RESULTS / "combined_policy_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSaved results/combined_policy.csv, combined_policy_runs.csv, "
          "combined_policy_validation.csv.")


if __name__ == "__main__":
    main()
