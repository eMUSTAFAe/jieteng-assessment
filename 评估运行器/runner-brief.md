# Evaluation runner

Build a repeatable evaluation runner and decide how to report failures, retry costs and uncertain rankings honestly.

Your role: Engineer preparing an evaluation for a product decision.

Decision scenario: A team wants a clear adapter ranking. Adapters fail in different ways, retries consume resources, and excluding failed runs can make results look better while hiding the experience users receive.

Affected stakeholders: Readers choosing an adapter, users experiencing failures and the team paying execution costs.

Scope and constraints: Use small supplied adapters and explicit per-example records. Keep original failures visible; no large benchmark campaign is required.

Decision to defend: What would you include, qualify or decline to rank, and how would your reporting priorities affect the decision?

This is an assessment scenario, not a claim about measured results. State any additional assumptions; you may choose a different reasonable option. Use your code and run results to support your own judgment, and explain what remains unknown.

Submitted work
Provide a bounded, reproducible implementation or experiment, run instructions, results and limitations that satisfy the requirements below. AI assistance is allowed when disclosed. A larger codebase or system does not by itself improve the quality of the submitted results.

Requirements:
- Implement repeatable runs across supplied adapters.
- Record per-example results, failures and cost.
- Handle timeouts and retries without hiding failed cases.

Personal judgment
Use this scenario and decisions from your actual work to explain your priorities, stakeholder tradeoffs, uncertainty and reflection in the four decision-journal questions. Several conclusions can be defensible; there is no required preferred opinion. Be ready to discuss your reasoning during an on-site interview.
- Your priorities and values: Choose what a useful evaluation should reveal to its readers. Explain which outcomes and values you prioritize.
- Your tradeoff and stakeholder judgment: Compare excluding failed runs with including them at an estimated cost. Explain transparency, comparability and stakeholder tradeoffs.
- Your judgment under uncertainty: Two adapters fail differently and the ranking is unstable. Explain what you would report or withhold and what would change your conclusion.
- Your ownership and reflection: Describe one reporting or retry decision you personally accepted, rejected or revised. Explain your reasoning and what you learned.

Run the supplied baseline with Python 3: python runner-baseline.py
Extend it within the stated task scope. Include source files, exact run commands, output tables or plots, failed cases and known limitations.

We recommend selecting 1–3 tasks; choosing more is allowed. Save answers as you work and return before the displayed deadline. The 48-hour (two-day) window starts at the invitation effective time and is a return/submission deadline, not expected working time.

AI may assist implementation. Use your own logic in the decision journal and be ready to discuss it without AI assistance during an on-site interview.
