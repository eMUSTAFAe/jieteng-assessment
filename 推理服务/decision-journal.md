# Decision journal — inference service

> **Note on authorship and AI assistance.** This journal was drafted with AI
> assistance, which is disclosed in the README. It is written in the first person
> because it is my submission. I own the final reasoning and am responsible for
> defending it without AI during the on-site interview. Q4 answers with the
> decision I revised in this assessment; it is the decision I would discuss in
> the interview.

## 1. My priorities and values

Under limited serving capacity, I prioritise **predictable completion for
interactive users** and **explicit overload** over maximum raw throughput. That
means:

* validate requests and reject bad input with 400 instead of crashing or
  silently predicting on it;
* bound the queue and return 429/504 when the service cannot meet its deadline,
  rather than letting users wait indefinitely;
* protect interactive traffic from batch traffic when the two compete;
* report latency, goodput and errors together, never a single throughput number.

What I am willing to sacrifice is **batch latency and batch fairness under
overload**. Batch users are the right group to queue or reject: their work is
usually not interactive, it can be retried or batched later, and the
alternative — letting batch traffic delay an interactive user — is worse for the
experience the product promises. I would not sacrifice correctness or
validation for throughput.

The stakeholders are interactive users (they feel every tail latency and
timeout), batch users (they want completion, but can absorb waiting), and the
operations team (it needs capacity limits and explicit error counters). The
baseline serves none of them well under overload: 47% errors, a 2-second p95,
and 93 connection-level failures.

## 2. My tradeoff and stakeholder judgment

The three ways to handle overload are admission control, longer waits, and
degraded responses. I implemented and measured the first two; I did not
implement degraded responses.

* **Admission control (bounded/priority):** reject excess with 429, expire
  queued requests at 504. Under overload this gives p95 ≈ 229 ms, but only 14%
  of requests succeed and goodput is 79 rps. The service stays predictable at
  the cost of refusing most of the load.
* **Longer waits (wide):** accept everything into a 128-entry queue. With 16
  workers this gives 490 rps goodput and a 123 ms p95 at the measured load; but
  the tail is bounded only by the queue and the 500 ms deadline, and the
  configuration costs 4× the workers.
* **Throughput at fixed capacity (batched):** 4 workers, batch 8, 10 ms wait.
  This gives 436 rps goodput and a 141 ms p95 under the same overload — the best
  use of 4 workers. At nominal concurrency it is *worse* than bounded (65 ms vs
  34 ms p95) because batches barely form and the wait is pure cost.
* **Priority (priority):** interactive requests jump ahead of batch requests.
  In the mixed workload, interactive error is 76.7% vs batch 90.6%, and
  interactive p95 is 205 ms vs batch 228 ms. Batch users bear the overload. The
  effect is modest because 4 workers are too few; the policy is right, the
  capacity is not.
* **Combined (`priority_batched`):** 8 workers, queue 64, batch 8 / 10 ms,
  priority. It reaches **699 rps goodput and a 104 ms p95 under overload with
  zero errors** — better than every component configuration. At nominal it is
  slower (62 ms p95) because batches barely form; batching is a throughput tool.
* **Baseline:** 46.7% errors, 2051 ms p95, 25 rps goodput. It has no validation,
  no bound, no deadline and no cancellation; a missing `text` is accepted and a
  non-string `text` resets the connection.

I would release **`priority_batched`**: it is the only measured configuration
that combines high goodput (699 rps), a bounded tail (104 ms p95) and zero errors
under the measured overload. The component configs remain evidence for why each
piece is needed; the combined config is the release candidate. At low-load
interactive-only, use `bounded` (34 ms p95) or disable batching.

The stakeholder tradeoff is explicit: under overload, batch users get queued and
rejected first; interactive users keep the priority lane; operations gets
429/504 counters and queue metrics instead of the baseline's silent OS-backlog
waits. I would rather explain a 429 to a batch user than a 2-second timeout to an
interactive user. That is a value judgment, and the report makes it visible so a
team that disagrees can reverse it.

## 3. My judgment under uncertainty

The faster configuration is not automatically more predictable. In the measured
overload, `wide_fast` is both faster and lower-p95 than `bounded`, but that is
because it has **16 workers and a 128-entry queue**, not because it has a better
overload policy. Its tail is bounded only by the queue and the 500 ms deadline;
a burst beyond that capacity would surface 429/504. The experiment did not push
it past that point, so I cannot claim it stays predictable at a larger load.

The `batched` configuration shows the opposite tradeoff: it is worse at nominal
(p95 65 ms vs 34 ms) and much better under overload (141 ms vs 229 ms; goodput
436 vs 79). The right release decision depends on the workload mix and the
capacity budget, not on a single “fastest” number.

What I would report: the per-scenario mean of 3 repeats, the p95/p99 with errors
included, the success/429/504 breakdown, the priority class split, and the
worker/queue budget behind every number. What I would withhold: a single
“fastest configuration” claim without its worker count and queue size; a claim
that priority fixes capacity (it does not — 4 workers still shed 85%); a claim
that the wide configuration is safe beyond the measured load.

What would change my conclusion:

* a larger burst test beyond the wide configuration's 128-entry queue — if its
  tail stays bounded without errors, it deserves more weight;
* actual interactive users reporting that a 200 ms p95 is unacceptable, or
  batch users reporting that 429 retries are too expensive;
* the real cost of 16 workers versus 4;
* a hard SLO for interactive completion (e.g., p99 < 300 ms) — the bounded
  policy is designed for this, the wide policy is not;
* a real model with a different compute profile — batching efficiency and
  service time would change the throughput/latency tradeoff;
* evidence that batch requests are interactive-in-disguise, which would
  invalidate the priority assumption.

What remains unknown: the real arrival process (my load is closed-loop), the
real model cost, the production CPU, the cost of 429 retries to batch users, and
whether a larger (16-worker) priority + batching combination would be even
better. I measured the 8-worker combination; it dominates the components at this
load, but a burst beyond its 64-entry queue is untested.

## 4. My ownership and reflection

One optimization I **rejected**: releasing the wide/unbounded configuration as
the default because it had the best overload numbers (100% success, 490 rps,
123 ms p95). Those numbers are real at the measured load, but they come from
16 workers and a 128-entry queue, and the configuration has no priority
isolation. It treats an interactive request and a batch request identically; at
a load beyond its queue, both would see the same tail. I kept it as a capacity
reference, not as the release default.

One recommendation I **retained only after testing it**: the combined
`priority_batched` configuration. I did not want to recommend a combination that
had not been run, so I measured it separately: 699 rps goodput, 104 ms p95 and
zero errors under overload. That turned the release recommendation from an
inference into a measured candidate. I also recorded its nominal cost: 62 ms p95
vs 34 ms for `bounded`, so batching is not a low-latency default.

One configuration detail I **revised**: the first run used the stdlib default
listen backlog of 5. At 64 concurrent clients that backlog became a hidden
queue, producing connection errors and a 1.4–2 s p95 even for configurations
that were serving fine. I set the bounded servers' backlog to 128 and re-ran.
This changed the overload story from “everything is slow” to a real policy
comparison. I also revised the timeout test after the first version timed out
the first request: its 300 ms service exceeded the 100 ms deadline, so the test
was measuring a total budget, not queue waiting. The corrected test gives
first=200, second=504.

The judgment beyond the metric: the metric said the wide configuration was
fastest; the stakeholder analysis said it was a capacity purchase with no
priority isolation. The metric said the priority configuration protected
interactive users; the goodput number said 4 workers were too few. The combined
measurement resolved the argument: 8 workers + batching + priority serves the
measured overload with zero errors and a bounded tail. Reflection: overload
policy is about **who waits and who is rejected**, not just about throughput;
the denominator (successful requests) and the tail with errors included are the
honest measures; a faster configuration can be less predictable if its tail is
bounded by a large queue; and the first experiment you run often reveals a
measurement artifact (here, the listen backlog) that looks like a model result.

The workplace analogue I would bring to an interview is a faster configuration
that looked better in a happy-path benchmark but shifted the tail to users. The
decision I would defend is the same: keep the bounded queue and explicit overload
signals, and refuse to call a capacity purchase an optimization. In this
assessment the wide 16-worker configuration was fastest at the measured load,
but it depended on four times the workers and a 128-entry queue; the priority
lane, not raw throughput, was what protected interactive users. What I learned
is to ask who waits when the queue fills, not only how fast the average request
is.
