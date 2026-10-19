# Decision journal — cost-sensitive decision models

> **Note on authorship and AI assistance.** This journal was drafted with AI
> assistance, which is disclosed in the README. It is written in the first person
> because it is my submission. I own the final reasoning and am responsible for
> defending it without AI during the on-site interview. Q4 answers with the
> decision I revised in this assessment; it is the decision I would discuss in
> the interview.

## 1. My priorities and values

I would prioritise avoiding **false negatives** — approving an applicant who then
defaults — but only while the false-positive burden stays visible and bounded.
The supplied table charges FN at 4 and FP at 1, so a model that misses bad risk
is expensive in the business's own accounting. A default is a concentrated loss
to the business owner, and in a real portfolio it can affect other customers and
staff.

A wrongful rejection is also a real harm. The applicant loses access, time, and
possibly money; the harm is spread across many people, which makes it easy for an
organisation to under-count. My values are:

* count errors in people as well as in cost units;
* do not let an average cost hide a large number of individual harms;
* make the tradeoff explicit enough that the people bearing it can contest it.

The affected stakeholders are the business owner (portfolio loss), creditworthy
applicants (wrongful denial), applicants who would default (the FN cases), and
the staff who investigate flags (workload). I prioritise FN avoidance for the
business case, but I refuse to treat FP as free.

## 2. My tradeoff and stakeholder judgment

Logistic regression and the decision tree distribute errors differently. At the
pure cost-optimal thresholds on dev, LR has **77 FP and 3 FN**; the tree has
**73 FP and 5 FN**. On the 200-person test set, LR has **75 FP / 7 FN** and the
tree **79 FP / 7 FN**; LR's expected cost is **0.515 vs. 0.535**, and its AUC is
**0.791 vs. 0.727**.

The tree is easier to explain to a loan officer — a short set of if/then rules —
which benefits the people investigating errors and the applicant who asks for a
reason. But its leaf probabilities are coarse, its ranking is worse, and its
cost-optimal threshold produces a higher FP burden on test. I recommend LR at
**0.251**: 47 FP / 13 FN on test. Against the tree at 0.200 (79 FP / 7 FN), that
is **32 fewer wrongful rejections for 6 more missed bad risks**; the expected
cost still favours LR (0.495 vs. 0.535). Against LR at its cost-optimal 0.136
(75 FP / 7 FN), the FP guard is **28 fewer FP for 6 more FN**. The business owner
bears the extra FN; applicants and reviewers bear the reduced FP.

I find this tradeoff defensible because the cost table already prices FN at 4x,
the guard is capped at +10% dev cost (+7.9% in practice), and I disclose the FP
count rather than averaging it away. If the business owner values FN more than
the table says, or if review capacity cannot handle 47 flags per 200 applicants,
the threshold should move down. I would show the cost curve and make that an
explicit business decision, not hide it in a model default. The tree remains a
useful explanatory model; it is not the decision model.

## 3. My judgment under uncertainty

The 1:4 cost table omits at least two consequences: the harm to a creditworthy
applicant who is wrongly rejected, and the workload on the people who investigate
flags. I would challenge the table before recommending a model.

Questions I would ask:

* Who produced the ratio, and from what data?
* Does “cost 1” include the applicant's time, stigma, and lost access, or only
  the business's lost margin?
* Is a flag an automatic rejection or a human review?
* What is the review capacity per 1,000 applications?
* Do FP rates differ across applicant groups?
* What is the appeal rate and the cost of an appeal?

I would not accept a single number as the social cost of a wrongful denial. I
ran two sensitivities.

**Cost ratio.** At 1:1 the dev-optimal LR threshold is 0.572 (test cost 0.260,
70 FP per 1,000); at 4:1 it is 0.136 (0.515, 375 FP per 1,000); at 16:1 it is
still 0.136 (0.935, 375 FP per 1,000). The threshold is a **cost assumption**,
not a property of the data.

**Added FP harm.** If each FP carries an extra `s` units, the dev-optimal LR
threshold moves from 0.136 at `s = 0` to **0.414 at `s = 1`**, and test FP falls
from 375 to **125 per 1,000** while business cost rises from 0.515 to **0.605**.
At `s = 8` the threshold is 0.646 and test FP is 60 per 1,000, at a business cost
of 0.900. A plausible unmeasured harm would justify a much more conservative
threshold than the supplied table.

I would present the recommendation as conditional, ask the business to supply a
defensible FP cost, and re-select the threshold on dev when it arrives. What
remains unknown: deployment prevalence, the true social cost of a wrongful
denial, subgroup effects, calibration drift, review capacity, and appeal costs.
If the unmeasured harm is near one unit, or if review capacity is below the flag
volume, my recommendation would move. That is not a weakness of the analysis; it
is the correct response to missing information.

## 4. My ownership and reflection

One recommendation I **revised**: I started with the pure cost-optimal
threshold. For LR it was **0.136** with dev cost **0.445**, but it flagged
**77 / 200 dev applicants (38.5%)** and would have produced **75 / 200 test
false positives (375 per 1,000)**. I revised to the FP-guarded **0.251**: dev
cost **0.480 (+7.9%)**, dev FP **52 (−32.5%)**, test FP **47**, test cost
**0.495**. I accepted a small dev cost increase because the FP price in the
table is not the whole harm.

I also **rejected the decision tree as the primary model** despite its
interpretability: its AUC was **0.727 vs. 0.791**, and at its cost-optimal
threshold it flagged **79 / 200 test applicants**. I kept the tree as a
secondary explanatory model. I rejected a fixed 0.50 threshold as “neutral”: for
LR it produces **29 FN on test (145 per 1,000)**, which the table prices at 4x,
so 0.50 is not a neutral default.

The judgment beyond the metric: the metric says 0.136; the stakeholder analysis
says 0.251; I chose 0.251 and disclosed both. Reflection: the cost table is a
value-laden artifact, not a natural constant; the threshold choice redistributes
errors between the business and applicants; the best headline metric is not
automatically the most defensible; and reporting FP counts and uncertainty is
part of the decision, not an afterthought. In a real deployment I would add human
review and an appeal path, and I would not auto-reject on this model alone.

The workplace analogue I would bring to an interview is the same pattern: a
metric that improves when a failed or unrepresentative subset is excluded, and
the decision to report the inclusive number. In this assessment the concrete
case is the FP guard: I rejected the cost-optimal 0.136 because it hid the
people who would be wrongly denied, and accepted 0.251 because the +7.9% dev
cost bought a 32.5% reduction in false positives. What I learned is that the
denominator and the stakeholder consequence are part of the model choice, not
post-hoc reporting.
