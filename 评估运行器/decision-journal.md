# Decision journal — evaluation runner

> **Note on authorship and AI assistance.** This journal was drafted with AI
> assistance, which is disclosed in the README. It is written in the first person
> because it is my submission. I own the final reasoning and am responsible for
> defending it without AI during the on-site interview. Q4 answers with the
> decision I revised in this assessment; it is the decision I would discuss in
> the interview.

## 1. My priorities and values

A useful evaluation should reveal **the experience users actually receive**, not
the best subset of it. That means I prioritise:

* **failure-inclusive accuracy** — every attempted run counts, including
  timeouts, exceptions and silent wrong answers;
* **failure modes over a single score** — a timeout and a silently wrong answer
  are not the same user experience, even when they produce the same accuracy;
* **cost including retries and timeouts** — the team paying for execution should
  see the price of the reliability it is buying;
* **tail latency with failures included** — a p95 computed only on completed
  runs hides the users who waited and got nothing.

What I am willing to sacrifice is the clean single-number ranking. The
success-only table is easier to read and more flattering to every adapter, but
it hides 35% of the `slow` adapter's runs and 35% of the `truncating` adapter's
correctness. I would rather publish a tiered, qualified result than a precise
but misleading order.

The stakeholders are the readers choosing an adapter (they need a defensible
recommendation), the users who experience failures (they bear the timeouts and
wrong answers), and the team paying execution costs (it bears retries and
timeouts). A reporting policy that benefits the reader by making the table clean
pushes the cost onto users and the team.

## 2. My tradeoff and stakeholder judgment

Excluding failed runs is the central temptation. On this experiment it changes
the story completely:

* `slow`: **100%** accuracy among completed runs, **65%** when every run is
  counted. It is correct whenever it finishes; it times out on all 14 long
  cases. The excluded runs are exactly the bad user experience.
* `truncating`: **65%** either way, but with **35% silently wrong** answers. It
  is cheap (1.54 per correct) and fast (p95 0.18 ms) because it fails without
  raising an error.
* `transient`: **100%** completed vs **99%** failure-inclusive at 3 retries. The
  2 exhausted runs are a small but real user-facing failure.
* `fast`: 100% everywhere; it is the control.

I would **include failures at their estimated cost**: count each failed run as
incorrect, charge every attempt and every timeout, and report latency both with
and without failures. This keeps the comparison transparent and comparable:
every adapter is measured on the same 200 runs, and the team can see what
reliability costs. Excluding failures is not neutral; it selects the runs that
make an adapter look good and then calls the result quality.

The stakeholder tradeoff is asymmetric. `fast` is best for every stakeholder, so
the real decision is about the fallbacks. `transient` shifts cost to the team
(retries, ~1.5 per correct) but gives users a 99% success rate and no silent
wrong answers. `slow` shifts cost to users (timeouts and 253 ms p95) and to the
team (4.23 per correct). `truncating` shifts cost to users in the worst way:
they receive wrong answers and may not know it. If I must choose who bears
overload, I would rather the team pay for a retry than the user receive a silent
wrong answer. That is the value judgment behind my ranking.

## 3. My judgment under uncertainty

The ranking is not statistically clean. Two facts drive my reporting:

1. **`fast` and `transient` overlap at 100%.** The bootstrap interval for
   `transient` is [97.5%, 100%], so I cannot claim `fast` is significantly more
   accurate. I still recommend `fast` as the default because it is also the
   cheapest (1.00 vs 1.50) and has the lowest tail latency, but the accuracy
   claim is qualified.
2. **`slow` and `truncating` are exactly tied on accuracy (65%, CI [50%, 80%])
   but fail differently.** `slow` times out visibly; `truncating` returns wrong
   answers silently. The rank order between them is not identified by accuracy;
   it is created by the cost tie-break (truncating 1.54 < slow 4.23). A strict
   "3rd vs 4th" ranking would therefore be an artifact of the cost model, not a
   finding.

What I would report: the tiered ranking (Tier 1 `fast`, Tier 2 `transient`,
not recommended `slow`/`truncating`), the failure-inclusive accuracy, the
failure-mode breakdown, the cost table, the latency-with-failures table, and the
bootstrap intervals. What I would withhold: a strict 1–4 order; any claim that
`truncating` is better than `slow` because it is cheaper; any claim that `fast`
is significantly more accurate than `transient`.

What would change my conclusion:

* a product requirement that **silent wrong answers are unacceptable** — then
  `truncating` drops regardless of cost;
* a tight interactive latency SLO — then `slow` and any retry-heavy
  configuration drop;
* a measured cost of a timeout that is much lower than 0.5 units — then `slow`'s
  cost disadvantage narrows, though it still fails 35% of runs;
* a larger case set with narrower intervals — the `fast`/`transient` overlap
  might resolve, and the `slow`/`truncating` accuracy tie might break;
* a lower transient failure rate — retries may become unnecessary, and
  `transient` could become the default.
* actual user feedback showing that a wrong answer and a timeout are valued
  differently — the failure-mode tradeoff would change.

What remains unknown: the real cost of a timeout, the real cost of a silent
wrong answer, the deployment input distribution, and whether the 35% failure
rates are representative. The report is a decision aid, not a claim about a
production adapter ecosystem.

## 4. My ownership and reflection

One reporting decision I **rejected**: ranking adapters by success-only
accuracy. It is the default in many evaluation harnesses — drop crashes and
timeouts, average the rest — and it made the `slow` adapter look perfect (100%)
even though it failed 35% of all runs. I rejected it after seeing the
completed-only and all-runs numbers side by side. Replacing it with
failure-inclusive accuracy changed the recommendation for `slow` from "top
tier" to "not recommended."

One retry decision I **revised**: I initially wanted to retry any failed run,
including wrong outputs. I rejected that because a deterministic wrong answer
does not improve on a retry; it burns budget and can create a false impression
that the adapter is reliable. The runner now retries only timeouts and
exceptions, and records every attempt. I retained a 3-retry budget for
`transient` because the cost per correct answer is essentially flat (1.53 at 0
retries, 1.50 at 3) and accuracy rises from 65.5% to 99%; for an interactive
path I would cut it to 2 retries (97%) to reduce the tail.

The judgment beyond the metric: the metric said `slow` was perfect on completed
runs; the stakeholder analysis said it was failing the users who need it most.
The metric said `truncating` was cheaper than `transient` per correct answer;
the failure-mode analysis said the cheapness came from failing silently. I chose
the interpretation that keeps the user's experience visible, and I disclosed the
numbers that support the other view. Reflection: the denominator of an
evaluation is a product decision; "excluding failures" is not a neutral
cleaning step; retries are a treatment for a specific failure mode, not a
general quality improvement; and a ranking that cannot be defended at the level
of failure modes should not be published as a strict order.

The workplace analogue I would bring to an interview is an evaluation where the
success-only dashboard looked clean while a meaningful share of runs were
timing out. The decision I would defend is the same: replace the success-only
metric with a failure-inclusive one, keep the failure log, and report the cost
of retries. In this assessment that changed `slow` from a 100% adapter to a
65% one. What I learned is that the denominator is a product decision; a clean
table can be the least honest artifact in the room.
