# Decision journal — small transformer experiment

> **Note on authorship and AI assistance.** This journal was drafted with AI
> assistance, which is disclosed in the README. It is written in the first person
> because it is my submission. I own the final reasoning and am responsible for
> defending it without AI during the on-site interview. Q4 answers with the
> decision I revised in this assessment; it is the decision I would discuss in
> the interview.

## 1. My priorities and values

I prioritise **learning value and interpretability** over a more impressive
headline. The brief offers a choice between an ambitious intervention that may
improve a benchmark but is hard to interpret and a simple intervention that
teaches more while producing a less impressive result. I chose the simple one:
causal masking versus no masking, with everything else held fixed.

The reason is that a benchmark number is only useful if I can explain what
caused it. The unmasked model's lower held-out loss (0.113 vs 1.487 nats) looks
like a win, but the target token is also the next input token; the unmasked
model can attend to it and copy. That is not a better language model; it is a
measurement artifact. The causal model's higher loss is the honest number, and
it is essentially the Bayes loss for this synthetic task (1.487 vs 1.501).

The stakeholders are the experiment collaborators (they need to understand and
reproduce the result), the reviewers (they need to audit the control), and future
users of the reported result (they should not deploy an invalid model or cite a
leakage artifact as progress). I am willing to sacrifice the more impressive
headline and the larger-model/architecture-search direction in exchange for a
result that isolates one mechanism and can be defended line by line.

## 2. My tradeoff and stakeholder judgment

The alternative experiment would be more ambitious: remove the mask, or increase
the model size, or search a few architectures. Any of those could produce a
lower loss and a more exciting headline. But the no-mask result is **confounded
by construction**: in next-token prediction, the input contains the target at
position `i + 1`, and bidirectional attention can see it. A larger model would
confound size and optimization with the masking question.

The clean intervention is the mask itself. Same data, same seed, same initial
weights, same optimizer and steps; the only difference is the attention mask.
That lets me attribute the effect to the mechanism. It produces a less
impressive headline — the causal model's loss is higher — but the headline is
valid. The unmasked model is still in the experiment, but as a **diagnostic**:
its 0.113 loss and 50% next-token attention measure the size of the leakage, not
the quality of a candidate model.

I find this tradeoff defensible because it serves the three stakeholders
differently and honestly:

* collaborators get a controlled comparison they can reproduce;
* reviewers get the mechanism check (attention to the next token) and the Bayes
  reference, so the interpretation is auditable;
* future users are told explicitly not to treat the unmasked number as a
  performance result.

The cost is that the "positive" result is small: the causal model matches Bayes
rather than beating it. That is the correct result for a synthetic task with a
known Bayes predictor. An experiment that reports a big win here would more
likely be measuring leakage than learning.

## 3. My judgment under uncertainty

The point estimates are strong and consistent across three seeds, but the
interpretation is where the uncertainty lives.

* The causal model's test loss is **1.4874 ± 0.0010** and its accuracy
  **71.88%**; the Bayes reference is **1.5014** and **71.25%**. The causal model
  is slightly better than Bayes on this finite test set; I report it as
  "approximately Bayes", not as a better-than-Bayes result.
* The unmasked model is **0.1132 ± 0.0018** and **98.00%**; its next-token
  attention is **0.500 ± 0.018** and its future attention is **0.093**. The
  attention diagnostic is the mechanism check: the causal model's future
  attention is exactly **0**.
* The sample predictions make it concrete: the unmasked model copies the target
  for positions 0–13, and fails at position 14 because the last target is not in
  the input; the causal model predicts the deterministic successor and errs on
  the 30% noise positions.

What I would report: the causal held-out loss and accuracy, the Bayes reference,
the unmasked metrics **as a leakage diagnostic**, the attention diagnostic, the
sample predictions, the 3-seed spread, and the limitations. What I would defer:
any claim about natural language, larger models or scaling; any claim that
causal masking "hurts" or "helps" performance in general — it makes the
next-token benchmark valid; any claim that the unmasked model is a candidate.

What would change my conclusion:

* if the unmasked model's next-token attention were near zero while its loss
  remained far below Bayes, there must be another leakage path, and the
  mechanism explanation would be incomplete;
* if a masked-input control at inference removed the unmasked model's advantage,
  that would confirm the leakage interpretation;
* if the causal model failed to approach Bayes, the task or training would be
  the problem, not the mask;
* if the data had no learnable structure, both models would be at the uniform
  loss and the comparison would be uninformative;
* if the effect vanished across more seeds or a real corpus, the result would be
  a synthetic curiosity rather than a useful mechanism demonstration.

What remains unknown: how this transfers to real language, larger models and
other architectures; whether attention is the only copy path (the diagnostic
reads the first block of each layer, not the full circuit); and whether a
different masking variant would teach something more. The experiment is
deliberately bounded, so these are deferred, not hidden.

## 4. My ownership and reflection

One interpretation I **revised**: my first reading of the per-seed table was
"the unmasked model is much better — 0.113 vs 1.487 loss and 98% vs 72%
accuracy." I revised it to "the unmasked benchmark is invalid due to target
leakage." The evidence was the construction (the target is the next input
token), the attention diagnostic (0.500 next-token attention vs exactly 0), and
the sample predictions (copying for 14 positions, failure at the last position
where there is no future token). I rejected reporting the lower loss as the
headline.

One design choice I **retained**: the simple mask intervention instead of a
larger model or an architecture search. The larger model might have produced a
more impressive number and a more confounded cause; the mask isolates the
mechanism. I also retained the 3-seed protocol and added the Bayes reference and
the attention diagnostic, because a single seed and a single loss number would
not have distinguished "learned the task" from "copied the answer."

The judgment beyond the metric: the metric said the unmasked model was far
better; the experimental construction said it was cheating; the mechanism check
said how. I chose the interpretation that keeps the benchmark honest, and I
reported the unmasked number as a diagnostic rather than deleting it. Reflection:
a lower benchmark number can be a measurement artifact; a controlled
intervention is worth more than an impressive but confounded one; the mechanism
check is as important as the headline metric; and the most valuable result here
is the one that tells future users **not** to trust the lower number. I would
rather publish a 1.487 that means something than a 0.113 that means nothing.

The workplace analogue I would bring to an interview is an impressive benchmark
number that disappears once the information leak is removed. The decision I
would defend is the same: report the controlled result, keep the impressive
number only as a diagnostic, and explain the mechanism. In this assessment the
unmasked model's 0.113 loss looked four times better than the causal model's
1.487, but its next-token attention was 0.500 and the causal model's was exactly
0. What I learned is that a lower metric can be a measurement artifact, and the
control is part of the result.
