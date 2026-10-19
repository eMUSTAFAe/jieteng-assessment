#!/usr/bin/env python3
"""Bounded inference service: admission control, batching, priority, timeout.

The supplied baseline is a single-threaded HTTP server with no validation, no
queue bound, no timeout and no cancellation.  This experiment adds:

* request validation (JSON object, string ``text``, length limit, known class);
* bounded concurrency and a bounded admission queue (429 when full);
* a priority policy that protects interactive requests from batch traffic;
* optional dynamic batching (throughput at the cost of waiting);
* a server-side queue deadline (504) and cooperative cancellation (499);
* per-request records for latency, queue time, service time and batch size.

Configurations compared
-----------------------
``baseline``   supplied serial server
``bounded``    4 workers, queue 32, FIFO, no batching
``batched``    4 workers, queue 32, batch 8 / 10 ms wait
``wide_fast``  16 workers, queue 128, FIFO, no batching (faster, less predictable)
``priority``   4 workers, queue 32, interactive-first (recommended)

Load scenarios
--------------
``nominal``    40 requests, concurrency 4, all interactive
``overload``   200 requests, concurrency 64, all interactive
``mixed``      200 requests, concurrency 64, 70% interactive / 30% batch

Run
---
    python service_experiment.py
    python make_figures.py

Outputs are written to ``results/``.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import multiprocessing
import platform
import queue
import socket
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
MAX_TEXT_LEN = 2000
CLIENT_TIMEOUT_S = 2.5
WARMUP_REQUESTS = 5
WARMUP_WAIT_S = 0.30
REPEATS = 3
PER_ITEM_MS = 30.0  # same simulated model cost for baseline and bounded configs


@dataclass
class ServiceConfig:
    name: str
    max_workers: int = 4
    max_queue: int = 32
    per_item_ms: float = 30.0
    queue_timeout_ms: float = 200.0
    batch_size: int = 1
    batch_wait_ms: float = 0.0
    batch_overhead_ms: float = 5.0
    batch_efficiency: float = 0.25
    priority: bool = False
    client_wait_s: float = 3.0


CONFIGS: dict[str, ServiceConfig] = {
    "bounded": ServiceConfig(
        name="bounded", max_workers=4, max_queue=32, per_item_ms=30.0,
        queue_timeout_ms=200.0, priority=False,
    ),
    "batched": ServiceConfig(
        name="batched", max_workers=4, max_queue=32, per_item_ms=30.0,
        queue_timeout_ms=300.0, batch_size=8, batch_wait_ms=10.0, priority=False,
    ),
    "wide_fast": ServiceConfig(
        name="wide_fast", max_workers=16, max_queue=128, per_item_ms=30.0,
        queue_timeout_ms=500.0, priority=False,
    ),
    "priority": ServiceConfig(
        name="priority", max_workers=4, max_queue=32, per_item_ms=30.0,
        queue_timeout_ms=200.0, priority=True,
    ),
    # The recommended combination, measured separately by combined_policy.py so
    # the main component comparison keeps its five configurations.
    "priority_batched": ServiceConfig(
        name="priority_batched", max_workers=8, max_queue=64, per_item_ms=30.0,
        queue_timeout_ms=300.0, batch_size=8, batch_wait_ms=10.0, priority=True,
    ),
}
CONFIG_ORDER = ["baseline", "bounded", "batched", "wide_fast", "priority"]

SCENARIOS: dict[str, dict[str, Any]] = {
    "nominal": {"n": 40, "concurrency": 4, "batch_fraction": 0.0},
    "overload": {"n": 200, "concurrency": 64, "batch_fraction": 0.0},
    "mixed": {"n": 200, "concurrency": 64, "batch_fraction": 0.30},
}
SCENARIO_ORDER = ["nominal", "overload", "mixed"]


# ---------------------------------------------------------------------------
# Request tasks and engine
# ---------------------------------------------------------------------------


@dataclass
class RequestTask:
    request_id: str
    text: str
    klass: str
    arrival: float
    deadline: float
    event: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"          # queued | ok | timeout | cancelled
    prediction: int | None = None
    queue_ms: float | None = None
    service_ms: float | None = None
    batch_size: int | None = None
    worker_id: int | None = None
    response_status: int | None = None
    cancelled: bool = False
    started: bool = False


class ServingEngine:
    """Bounded worker pool with a priority admission queue and optional batching."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=config.max_queue)
        self.stop_event = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._tasks: dict[str, RequestTask] = {}
        self._tasks_lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self._records_lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for worker_id in range(self.config.max_workers):
            thread = threading.Thread(
                target=self._worker,
                args=(worker_id,),
                daemon=True,
                name=f"{self.config.name}-worker-{worker_id}",
            )
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        # Workers poll with a 100 ms timeout, so stop_event is enough; injecting
        # None sentinels into a PriorityQueue would compare None with None.
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)

    def submit(self, task: RequestTask) -> bool:
        priority = 0
        if self.config.priority and task.klass == "batch":
            priority = 1
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        try:
            self.queue.put_nowait((priority, seq, task))
        except queue.Full:
            return False
        with self._tasks_lock:
            self._tasks[task.request_id] = task
        return True

    def cancel(self, request_id: str) -> bool:
        with self._tasks_lock:
            task = self._tasks.get(request_id)
        if task is None or task.event.is_set() or task.started:
            return False
        task.cancelled = True
        task.status = "cancelled"
        task.event.set()
        return True

    def _worker(self, worker_id: int) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break

            batch: list[tuple[int, int, RequestTask]] = [item]
            if self.config.batch_size > 1:
                deadline = time.monotonic() + self.config.batch_wait_ms / 1000.0
                while len(batch) < self.config.batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self.queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if nxt is None:
                        self.queue.task_done()
                        self.stop_event.set()
                        break
                    batch.append(nxt)

            self._process_batch(batch, worker_id)

    def _process_batch(
        self, batch: list[tuple[int, int, RequestTask]], worker_id: int
    ) -> None:
        now = time.monotonic()
        active: list[RequestTask] = []
        for _, _, task in batch:
            if task.cancelled:
                task.status = "cancelled"
                task.event.set()
                self._record(task, "cancelled", worker_id, now)
                self.queue.task_done()
                continue
            if now > task.deadline:
                task.status = "timeout"
                task.event.set()
                self._record(task, "timeout", worker_id, now)
                self.queue.task_done()
                continue
            active.append(task)

        if not active:
            return

        # Re-check the queue deadline after the batch wait.
        now = time.monotonic()
        still_active: list[RequestTask] = []
        for task in active:
            if task.cancelled:
                task.status = "cancelled"
                task.event.set()
                self._record(task, "cancelled", worker_id, now)
                self.queue.task_done()
            elif now > task.deadline:
                task.status = "timeout"
                task.event.set()
                self._record(task, "timeout", worker_id, now)
                self.queue.task_done()
            else:
                still_active.append(task)
        active = still_active
        if not active:
            return

        batch_size = len(active)
        if batch_size == 1:
            service_ms = self.config.per_item_ms
        else:
            service_ms = (
                self.config.batch_overhead_ms
                + self.config.per_item_ms * batch_size * self.config.batch_efficiency
            )
        for task in active:
            task.started = True
        service_start = time.monotonic()
        time.sleep(service_ms / 1000.0)
        service_end = time.monotonic()

        for task in active:
            task.batch_size = batch_size
            task.worker_id = worker_id
            task.service_ms = (service_end - service_start) * 1000.0
            task.queue_ms = (service_start - task.arrival) * 1000.0
            if task.cancelled:
                task.status = "cancelled"
            else:
                task.status = "ok"
                task.prediction = len(task.text)
            task.event.set()
            self._record(task, task.status, worker_id, service_start)
            self.queue.task_done()

    def _record(
        self, task: RequestTask, status: str, worker_id: int, service_start: float
    ) -> None:
        record = {
            "request_id": task.request_id,
            "class": task.klass,
            "arrival": task.arrival,
            "queue_ms": (service_start - task.arrival) * 1000.0,
            "service_ms": task.service_ms,
            "batch_size": task.batch_size,
            "worker_id": worker_id,
            "status": status,
        }
        with self._records_lock:
            self.records.append(record)

    def snapshot_records(self) -> list[dict[str, Any]]:
        with self._records_lock:
            return list(self.records)


# ---------------------------------------------------------------------------
# HTTP handler for the bounded engine
# ---------------------------------------------------------------------------


class BoundHandler(BaseHTTPRequestHandler):
    engine: ServingEngine
    config: ServiceConfig

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # silence
        pass

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None, "invalid Content-Length"
        if length <= 0:
            return None, "empty body"
        if length > 1_000_000:
            return None, "body too large"
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
        if not isinstance(payload, dict):
            return None, "body must be a JSON object"
        return payload, None

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload, error = self._read_json()
        if error is not None:
            self._send_json(400, {"error": error})
            return
        assert payload is not None

        if path == "/cancel":
            request_id = payload.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                self._send_json(400, {"error": "request_id must be a non-empty string"})
                return
            cancelled = self.engine.cancel(request_id)
            self._send_json(200, {"request_id": request_id, "cancelled": cancelled})
            return

        if path != "/predict":
            self._send_json(404, {"error": "unknown path"})
            return

        text = payload.get("text")
        if not isinstance(text, str):
            self._send_json(400, {"error": "text must be a string"})
            return
        if len(text) > MAX_TEXT_LEN:
            self._send_json(
                400, {"error": f"text exceeds {MAX_TEXT_LEN} characters"}
            )
            return
        klass = payload.get("class", "interactive")
        if klass not in ("interactive", "batch"):
            self._send_json(400, {"error": "class must be interactive or batch"})
            return
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = uuid.uuid4().hex

        arrival = time.monotonic()
        task = RequestTask(
            request_id=request_id,
            text=text,
            klass=klass,
            arrival=arrival,
            deadline=arrival + self.config.queue_timeout_ms / 1000.0,
        )
        if not self.engine.submit(task):
            self._send_json(
                429,
                {
                    "request_id": request_id,
                    "error": "admission queue full",
                    "class": klass,
                },
            )
            return

        wait_s = max(
            0.001,
            min(self.config.client_wait_s, task.deadline - time.monotonic()),
        )
        finished = task.event.wait(wait_s)
        if not finished:
            task.cancelled = True
            task.status = "timeout"
            self._send_json(
                504,
                {
                    "request_id": request_id,
                    "error": "client wait exceeded",
                    "class": klass,
                },
            )
            return

        if task.status == "ok":
            status = 200
        elif task.status == "timeout":
            status = 504
        elif task.status == "cancelled":
            status = 499
        else:
            status = 500
        task.response_status = status
        self._send_json(
            status,
            {
                "request_id": request_id,
                "prediction": task.prediction,
                "class": klass,
                "batch_size": task.batch_size,
                "queue_ms": task.queue_ms,
                "service_ms": task.service_ms,
                "worker_id": task.worker_id,
                "status": task.status,
            },
        )


def make_handler(config: ServiceConfig, engine: ServingEngine) -> type[BoundHandler]:
    name = f"Handler_{config.name}_{uuid.uuid4().hex[:6]}"
    return type(name, (BoundHandler,), {"engine": engine, "config": config})


class BoundedHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with a listen backlog large enough for the load.

    The stdlib default is 5.  With 64 concurrent clients that backlog is a
    hidden queue in front of the service and would dominate the latency.
    """

    request_queue_size = 128
    daemon_threads = True


# ---------------------------------------------------------------------------
# Server lifecycles
# ---------------------------------------------------------------------------


@contextmanager
def running_service(
    config: ServiceConfig,
) -> Iterator[tuple[str, ServingEngine]]:
    engine = ServingEngine(config)
    engine.start()
    handler = make_handler(config, engine)
    server = BoundedHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", engine
    finally:
        server.shutdown()
        server.server_close()
        engine.stop()
        thread.join(timeout=2.0)


def _load_baseline_module():
    path = ROOT / "inference-baseline.py"
    spec = importlib.util.spec_from_file_location("inference_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load inference-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server_process(
    config_name: str,
    config_dict: dict[str, Any] | None,
    port_queue: Any,
    stop_event: Any,
    result_path: str,
) -> None:
    """Run one HTTP server in a separate process.

    The server is isolated so the client's threads do not compete with the
    server's worker threads for the GIL.  The supplied baseline is wrapped only
    to add the same simulated 30 ms model cost as the bounded configurations;
    the original baseline file is unchanged.
    """
    engine: ServingEngine | None = None
    if config_name == "baseline":
        module = _load_baseline_module()
        module.Handler.log_message = lambda *args, **kwargs: None

        class FairBaselineHandler(module.Handler):
            def do_POST(self):  # noqa: N802
                time.sleep(PER_ITEM_MS / 1000.0)
                super().do_POST()

        server = HTTPServer(("127.0.0.1", 0), FairBaselineHandler)
    else:
        assert config_dict is not None
        config = ServiceConfig(**config_dict)
        engine = ServingEngine(config)
        engine.start()
        handler = make_handler(config, engine)
        server = BoundedHTTPServer(("127.0.0.1", 0), handler)

    server.handle_error = lambda *args, **kwargs: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port_queue.put(server.server_address[1])
    stop_event.wait()
    server.shutdown()
    server.server_close()
    if engine is not None:
        engine.stop()
        records = engine.snapshot_records()
    else:
        records = []
    Path(result_path).write_text(json.dumps(records), encoding="utf-8")
    thread.join(timeout=2.0)


class RemoteService:
    """Start one server process, expose its URL, collect records on exit."""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.context = multiprocessing.get_context("spawn")
        self.port_queue = self.context.Queue()
        self.stop_event = self.context.Event()
        with tempfile.NamedTemporaryFile(
            prefix=f"service-{config_name}-", suffix=".json", delete=False
        ) as handle:
            self.result_path = Path(handle.name)
        config_dict = (
            None if config_name == "baseline" else CONFIGS[config_name].__dict__
        )
        self.process = self.context.Process(
            target=server_process,
            args=(
                config_name,
                config_dict,
                self.port_queue,
                self.stop_event,
                str(self.result_path),
            ),
            daemon=True,
        )
        self.url = ""
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "RemoteService":
        self.process.start()
        try:
            port = self.port_queue.get(timeout=15)
        except Exception:
            self.process.terminate()
            self.process.join(timeout=2)
            raise RuntimeError(f"server {self.config_name} did not start")
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.stop_event.set()
        self.process.join(timeout=8)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.result_path.exists():
            try:
                self.records = json.loads(
                    self.result_path.read_text(encoding="utf-8")
                )
            except Exception:
                self.records = []
            try:
                self.result_path.unlink()
            except OSError:
                pass
        try:
            self.port_queue.close()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def http_post(
    url: str,
    payload: dict[str, Any],
    *,
    path: str = "/predict",
    raw: bytes | None = None,
    timeout_s: float = CLIENT_TIMEOUT_S,
) -> dict[str, Any]:
    parsed = urlparse(url)
    start = time.perf_counter()
    status = -1
    body = b""
    error = ""
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
        data = raw if raw is not None else json.dumps(payload).encode("utf-8")
        conn.request(
            "POST",
            path,
            body=data,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        response = conn.getresponse()
        status = response.status
        body = response.read()
        conn.close()
    except socket.timeout:
        status = 0
        error = "client_timeout"
    except Exception as exc:  # noqa: BLE001 - recorded per request
        status = -1
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000.0
    parsed_body: dict[str, Any] = {}
    if body:
        try:
            parsed_body = json.loads(body)
        except json.JSONDecodeError:
            parsed_body = {}
    return {
        "status": status,
        "latency_ms": latency_ms,
        "error": error,
        "response": parsed_body,
    }


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------


def make_payload(
    scenario: str, repeat: int, index: int, batch_fraction: float
) -> dict[str, Any]:
    request_id = f"{scenario}-r{repeat}-{index:04d}"
    text = f"{scenario}-{repeat}-{index:03d}-" + "x" * (5 + index % 17)
    klass = "batch" if (batch_fraction > 0 and index % 10 < int(batch_fraction * 10)) else "interactive"
    return {"request_id": request_id, "text": text, "class": klass}


def client_record(
    config: str, scenario: str, repeat: int, payload: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    response = result["response"]
    status = result["status"]
    expected_prediction = len(payload["text"])
    response_id = response.get("request_id")
    return {
        "config": config,
        "scenario": scenario,
        "repeat": repeat,
        "request_id": payload["request_id"],
        "class": payload["class"],
        "text_len": expected_prediction,
        "status": status,
        "error": result["error"],
        "latency_ms": result["latency_ms"],
        "prediction_ok": bool(status == 200 and response.get("prediction") == expected_prediction),
        "ordering_ok": (response_id == payload["request_id"]) if response_id is not None else None,
        "response_request_id": response_id,
        "batch_size": response.get("batch_size"),
        "queue_ms": response.get("queue_ms"),
        "service_ms": response.get("service_ms"),
        "worker_id": response.get("worker_id"),
    }


def run_load(
    config_name: str,
    url: str,
    scenario: str,
    repeat: int,
    engine: ServingEngine | None,
) -> tuple[list[dict[str, Any]], float]:
    spec = SCENARIOS[scenario]
    n = spec["n"]
    concurrency = spec["concurrency"]
    batch_fraction = spec["batch_fraction"]

    # Warm-up (discarded)
    for i in range(WARMUP_REQUESTS):
        payload = make_payload("warmup", repeat, i, 0.0)
        http_post(url, payload)
    time.sleep(WARMUP_WAIT_S)

    payloads = [make_payload(scenario, repeat, i, batch_fraction) for i in range(n)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(http_post, url, payload) for payload in payloads]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start

    records = [
        client_record(config_name, scenario, repeat, payload, result)
        for payload, result in zip(payloads, results)
    ]
    return records, elapsed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def safe_percentile(values: list[float], q: float) -> float | None:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.percentile(clean, q))


def summarize_run(
    records: list[dict[str, Any]],
    elapsed: float,
    config_name: str,
    scenario: str,
    repeat: int,
    server_records: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(records)
    statuses = pd.Series([r["status"] for r in records])
    success_mask = [r["status"] == 200 for r in records]
    success_count = int(sum(success_mask))
    latencies_all = [r["latency_ms"] for r in records]
    latencies_ok = [r["latency_ms"] for r in records if r["status"] == 200]
    interactive = [r for r in records if r["class"] == "interactive"]
    batch = [r for r in records if r["class"] == "batch"]

    status_counts = {
        f"status_{code}": int((statuses == code).sum())
        for code in (400, 429, 499, 500, 503, 504)
    }
    client_timeouts = int((statuses == 0).sum())
    client_errors = int((statuses == -1).sum())

    server_queue = [r["queue_ms"] for r in server_records if r.get("queue_ms") is not None]
    server_service = [r["service_ms"] for r in server_records if r.get("service_ms") is not None]
    batch_sizes = [r["batch_size"] for r in server_records if r.get("batch_size") is not None]

    return {
        "config": config_name,
        "scenario": scenario,
        "repeat": repeat,
        "n": n,
        "successes": success_count,
        "error_rate": 1.0 - success_count / n,
        "throughput_rps": n / elapsed,
        "goodput_rps": success_count / elapsed,
        "elapsed_s": elapsed,
        **status_counts,
        "client_timeouts": client_timeouts,
        "client_errors": client_errors,
        "latency_p50_all_ms": safe_percentile(latencies_all, 50),
        "latency_p95_all_ms": safe_percentile(latencies_all, 95),
        "latency_p99_all_ms": safe_percentile(latencies_all, 99),
        "latency_p50_ok_ms": safe_percentile(latencies_ok, 50),
        "latency_p95_ok_ms": safe_percentile(latencies_ok, 95),
        "interactive_p95_all_ms": safe_percentile([r["latency_ms"] for r in interactive], 95),
        "batch_p95_all_ms": safe_percentile([r["latency_ms"] for r in batch], 95),
        "interactive_error_rate": (
            1.0 - sum(1 for r in interactive if r["status"] == 200) / len(interactive)
            if interactive else None
        ),
        "batch_error_rate": (
            1.0 - sum(1 for r in batch if r["status"] == 200) / len(batch)
            if batch else None
        ),
        "server_mean_queue_ms": float(np.mean(server_queue)) if server_queue else None,
        "server_p95_queue_ms": safe_percentile(server_queue, 95),
        "server_mean_service_ms": float(np.mean(server_service)) if server_service else None,
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else None,
        "max_batch_size": int(np.max(batch_sizes)) if batch_sizes else None,
        "prediction_match_rate": float(np.mean([bool(r["prediction_ok"]) for r in records])),
        "ordering_match_rate": (
            float(np.mean([bool(r["ordering_ok"]) for r in records if r["ordering_ok"] is not None]))
            if any(r["ordering_ok"] is not None for r in records)
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Validation, cancellation, timeout, ordering tests
# ---------------------------------------------------------------------------


def validation_test(config_name: str, url: str) -> list[dict[str, Any]]:
    cases = [
        ("invalid_json", b"not json"),
        ("missing_text", json.dumps({"request_id": "v-missing"}).encode()),
        ("text_not_string", json.dumps({"text": 123}).encode()),
        ("text_too_long", json.dumps({"text": "x" * (MAX_TEXT_LEN + 1)}).encode()),
        ("unknown_class", json.dumps({"text": "hello", "class": "urgent"}).encode()),
    ]
    rows = []
    for name, raw in cases:
        result = http_post(url, {}, raw=raw, timeout_s=1.0)
        rows.append(
            {
                "config": config_name,
                "case": name,
                "status": result["status"],
                "error": result["error"],
                "body": json.dumps(result["response"])[:200],
            }
        )
    return rows


def timeout_test() -> dict[str, Any]:
    config = ServiceConfig(
        name="timeout_test", max_workers=1, max_queue=4, per_item_ms=300.0,
        queue_timeout_ms=450.0, client_wait_s=2.0,
    )
    with running_service(config) as (url, engine):
        first = {}
        thread = threading.Thread(
            target=lambda: first.update(http_post(url, {"request_id": "t1", "text": "first"})),
            daemon=True,
        )
        thread.start()
        time.sleep(0.05)
        second = http_post(url, {"request_id": "t2", "text": "second"}, timeout_s=2.0)
        thread.join(timeout=3.0)
        time.sleep(0.2)
        records = engine.snapshot_records()
    return {
        "first_status": first.get("status"),
        "second_status": second["status"],
        "second_error": second["error"],
        "server_records": records,
        "passed": second["status"] == 504,
    }


def cancellation_test() -> dict[str, Any]:
    config = ServiceConfig(
        name="cancel_test", max_workers=1, max_queue=10, per_item_ms=300.0,
        queue_timeout_ms=5000.0, client_wait_s=3.0,
    )
    with running_service(config) as (url, engine):
        results: dict[str, dict[str, Any]] = {}
        lock = threading.Lock()

        def send(payload: dict[str, Any]) -> None:
            result = http_post(url, payload, timeout_s=4.0)
            with lock:
                results[payload["request_id"]] = result

        threads = []
        for rid in ("c1", "c2"):
            t = threading.Thread(
                target=send,
                args=({"request_id": rid, "text": rid, "class": "interactive"},),
                daemon=True,
            )
            t.start()
            threads.append(t)
        time.sleep(0.05)
        for rid in ("k1", "k2"):
            t = threading.Thread(
                target=send,
                args=({"request_id": rid, "text": rid, "class": "batch"},),
                daemon=True,
            )
            t.start()
            threads.append(t)
        time.sleep(0.05)
        cancels = {
            rid: http_post(url, {"request_id": rid}, path="/cancel", timeout_s=1.0)
            for rid in ("k1", "k2")
        }
        for t in threads:
            t.join(timeout=5.0)
        time.sleep(0.5)
        records = engine.snapshot_records()

    cancel_ok = all(
        response["status"] == 200 and response["response"].get("cancelled") is True
        for response in cancels.values()
    )
    cancelled_statuses = [
        results.get(rid, {}).get("status") for rid in ("k1", "k2")
    ]
    return {
        "cancel_responses": {
            rid: {"status": response["status"], "body": response["response"]}
            for rid, response in cancels.items()
        },
        "cancelled_request_statuses": cancelled_statuses,
        "server_records": records,
        "passed": bool(cancel_ok and all(s == 499 for s in cancelled_statuses)),
    }


def in_service_cancellation_test() -> dict[str, Any]:
    """Cancelling a request that has already started must be refused.

    The worker marks the task ``started`` before the service sleep.  A cancel
    after that point returns ``cancelled: false``; the original request still
    completes with 200.  This is the honest boundary of cooperative
    cancellation: queued work can be dropped, in-service work cannot be
    interrupted safely.
    """
    config = ServiceConfig(
        name="cancel_in_service", max_workers=1, max_queue=4, per_item_ms=500.0,
        queue_timeout_ms=5000.0, client_wait_s=3.0,
    )
    with running_service(config) as (url, engine):
        result: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: result.update(
                http_post(
                    url,
                    {"request_id": "s1", "text": "running", "class": "interactive"},
                    timeout_s=4.0,
                )
            ),
            daemon=True,
        )
        thread.start()
        time.sleep(0.12)  # the worker has entered the 500 ms service sleep
        cancel = http_post(url, {"request_id": "s1"}, path="/cancel", timeout_s=1.0)
        thread.join(timeout=4.0)
        time.sleep(0.2)
        records = engine.snapshot_records()
    return {
        "cancel_response": {"status": cancel["status"], "body": cancel["response"]},
        "original_status": result.get("status"),
        "server_records": records,
        "passed": bool(
            cancel["status"] == 200
            and cancel["response"].get("cancelled") is False
            and result.get("status") == 200
        ),
    }


def ordering_test() -> dict[str, Any]:
    config = ServiceConfig(
        name="ordering_test", max_workers=4, max_queue=64, per_item_ms=20.0,
        queue_timeout_ms=500.0, batch_size=8, batch_wait_ms=10.0,
    )
    with running_service(config) as (url, engine):
        payloads = [
            {
                "request_id": f"o{i:03d}",
                "text": f"order-{i:03d}-" + "y" * (i % 9),
                "class": "interactive",
            }
            for i in range(32)
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda p: http_post(url, p), payloads))
        time.sleep(0.2)
        records = engine.snapshot_records()
    matches = []
    predictions = []
    for payload, result in zip(payloads, results):
        response = result["response"]
        matches.append(response.get("request_id") == payload["request_id"])
        predictions.append(response.get("prediction") == len(payload["text"]))
    return {
        "ordering_match_rate": float(np.mean(matches)),
        "prediction_match_rate": float(np.mean(predictions)),
        "batch_sizes": [r["batch_size"] for r in records],
        "passed": bool(all(matches) and all(predictions)),
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
        print("BOUNDED INFERENCE SERVICE")
        print("=" * 78)
        print(f"Python   : {platform.python_version()}")
        print(f"numpy    : {np.__version__}")
        print(f"pandas   : {pd.__version__}")
        print(f"client timeout: {CLIENT_TIMEOUT_S * 1000:.0f} ms")
        print(f"repeats  : {REPEATS}")
        print("\nConfigurations")
        print("--------------")
        for name in CONFIG_ORDER:
            if name == "baseline":
                print("  baseline    supplied serial HTTPServer; no validation/bounds/timeout")
            else:
                c = CONFIGS[name]
                print(
                    f"  {name:<11} workers={c.max_workers:<2} queue={c.max_queue:<3} "
                    f"batch={c.batch_size} wait={c.batch_wait_ms:.0f}ms "
                    f"queue_timeout={c.queue_timeout_ms:.0f}ms "
                    f"priority={c.priority}"
                )

        all_client: list[dict[str, Any]] = []
        all_server: list[dict[str, Any]] = []
        run_rows: list[dict[str, Any]] = []

        for scenario in SCENARIO_ORDER:
            for repeat in range(REPEATS):
                for config_name in CONFIG_ORDER:
                    with RemoteService(config_name) as remote:
                        records, elapsed = run_load(
                            config_name, remote.url, scenario, repeat, None
                        )
                        time.sleep(0.1)
                    server_records = remote.records
                    for record in records:
                        record["repeat"] = repeat
                    all_client.extend(records)
                    for server_record in server_records:
                        server_record["config"] = config_name
                        server_record["scenario"] = scenario
                        server_record["repeat"] = repeat
                    all_server.extend(server_records)
                    run_rows.append(
                        summarize_run(
                            records,
                            elapsed,
                            config_name,
                            scenario,
                            repeat,
                            server_records,
                        )
                    )

        client_df = pd.DataFrame(all_client)
        server_df = pd.DataFrame(all_server)
        runs_df = pd.DataFrame(run_rows)

        # Aggregate repeats: mean and sd
        group_cols = ["config", "scenario"]
        numeric_cols = [
            c for c in runs_df.columns
            if c not in group_cols + ["repeat"] and pd.api.types.is_numeric_dtype(runs_df[c])
        ]
        mean_df = runs_df.groupby(group_cols, sort=False)[numeric_cols].mean().reset_index()
        sd_df = runs_df.groupby(group_cols, sort=False)[numeric_cols].std().reset_index()
        sd_df = sd_df.rename(columns={c: f"{c}_sd" for c in numeric_cols})
        summary = mean_df.merge(sd_df, on=group_cols, how="left")

        display_cols = [
            "config", "n", "successes", "error_rate", "throughput_rps",
            "goodput_rps",
            "latency_p50_all_ms", "latency_p95_all_ms", "latency_p99_all_ms",
            "latency_p95_ok_ms", "status_429", "status_504",
            "client_timeouts", "client_errors",
            "server_mean_queue_ms", "server_p95_queue_ms",
            "mean_batch_size",
        ]
        for scenario in SCENARIO_ORDER:
            frame = summary[summary["scenario"] == scenario][display_cols].copy()
            frame = frame.set_index("config").reindex(CONFIG_ORDER).reset_index()
            print_table(f"Scenario: {scenario}", frame)
            print(
                f"  (n={SCENARIOS[scenario]['n']}, "
                f"concurrency={SCENARIOS[scenario]['concurrency']}, "
                f"batch_fraction={SCENARIOS[scenario]['batch_fraction']:.0%}; "
                f"mean of {REPEATS} repeats)"
            )

        # Mixed: per-class
        mixed = summary[summary["scenario"] == "mixed"][
            [
                "config", "interactive_p95_all_ms", "batch_p95_all_ms",
                "interactive_error_rate", "batch_error_rate",
            ]
        ].set_index("config").reindex(CONFIG_ORDER).reset_index()
        print_table("Mixed workload: interactive vs batch (mean)", mixed)

        # Validation
        validation_rows: list[dict[str, Any]] = []
        for config_name in CONFIG_ORDER:
            with RemoteService(config_name) as remote:
                validation_rows.extend(validation_test(config_name, remote.url))
        validation_df = pd.DataFrame(validation_rows)
        print_table("Validation: invalid requests", validation_df)

        # Targeted tests
        print("\nTargeted tests")
        print("--------------")
        timeout_result = timeout_test()
        cancel_result = cancellation_test()
        in_service_result = in_service_cancellation_test()
        order_result = ordering_test()
        print(f"  timeout test passed   : {timeout_result['passed']} "
              f"(first={timeout_result['first_status']}, "
              f"second={timeout_result['second_status']})")
        print(f"  cancellation passed   : {cancel_result['passed']} "
              f"(cancelled statuses={cancel_result['cancelled_request_statuses']})")
        print(f"  in-service cancel     : {in_service_result['passed']} "
              f"(cancel={in_service_result['cancel_response']['body']}, "
              f"original={in_service_result['original_status']})")
        print(f"  ordering passed       : {order_result['passed']} "
              f"(ordering={order_result['ordering_match_rate']:.0%}, "
              f"prediction={order_result['prediction_match_rate']:.0%})")

        targeted = {
            "timeout": timeout_result,
            "cancellation": cancel_result,
            "in_service_cancellation": in_service_result,
            "ordering": order_result,
        }

        # Persist
        client_df.to_csv(RESULTS / "client_requests.csv", index=False)
        server_df.to_csv(RESULTS / "server_requests.csv", index=False)
        runs_df.to_csv(RESULTS / "scenario_runs.csv", index=False)
        summary.to_csv(RESULTS / "scenario_summary.csv", index=False)
        mixed.to_csv(RESULTS / "mixed_by_class.csv", index=False)
        validation_df.to_csv(RESULTS / "validation_results.csv", index=False)
        with (RESULTS / "targeted_tests.json").open("w", encoding="utf-8") as handle:
            json.dump(targeted, handle, indent=2, default=str)

        metadata = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "client_timeout_s": CLIENT_TIMEOUT_S,
            "repeats": REPEATS,
            "configs": {
                name: (None if name == "baseline" else CONFIGS[name].__dict__)
                for name in CONFIG_ORDER
            },
            "scenarios": SCENARIOS,
        }
        with (RESULTS / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        print("\nSaved:")
        for name in sorted(p.name for p in RESULTS.iterdir() if p.is_file()):
            print(f"  results/{name}")
        print("\nDone.")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
