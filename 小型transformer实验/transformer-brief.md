# Small transformer experiment

Run a small transformer experiment and decide which result is worth learning from and reporting within a bounded scope.

Hardware and model access: Training machine needed · CPU sufficient
Use a standard CPU machine for the tiny experiment. A GPU is optional if you choose a larger model; large training runs are not required.

Your role: Engineer planning a small research experiment.

Decision scenario: Your team can investigate one design change. An ambitious intervention may improve a benchmark but make its cause hard to interpret; a simple intervention may teach more while producing a less impressive headline.

Affected stakeholders: Experiment collaborators, reviewers and future users of the reported result.

Scope and constraints: Use a tiny reproducible experiment and one controlled comparison. Large training runs and extensive architecture search are outside the task scope.

Decision to defend: Would you prioritize interpretability, learning value or performance, and what would you responsibly conclude from limited experiment results?

This is an assessment scenario, not a claim about measured results. State any additional assumptions; you may choose a different reasonable option. Use your code and run results to support your own judgment, and explain what remains unknown.

Submitted work
Provide a bounded, reproducible implementation or experiment, run instructions, results and limitations that satisfy the requirements below. AI assistance is allowed when disclosed. A larger codebase or system does not by itself improve the quality of the submitted results.

Requirements:
- Complete a tiny attention or masking component.
- Compare one intervention against a fixed baseline.
- Report held-out loss, predictions and limitations.

Personal judgment
Use this scenario and decisions from your actual work to explain your priorities, stakeholder tradeoffs, uncertainty and reflection in the four decision-journal questions. Several conclusions can be defensible; there is no required preferred opinion. Be ready to discuss your reasoning during an on-site interview.
- Your priorities and values: Choose whether your experiment should prioritize simplicity, learning value or benchmark performance. Explain your own priorities.
- Your tradeoff and stakeholder judgment: Compare a clean interpretable intervention with a more promising but confounded one. Defend your experiment choice.
- Your judgment under uncertainty: Results are weak or contradictory near the deadline. Explain what you would report, defer or investigate and why.
- Your ownership and reflection: Describe one design or AI-suggested interpretation you accepted, rejected or revised. Explain what you learned about your own reasoning.

Run the supplied baseline with Python 3: python transformer-baseline.py
Extend it within the stated task scope. Include source files, exact run commands, output tables or plots, failed cases and known limitations.

We recommend selecting 1–3 tasks; choosing more is allowed. Save answers as you work and return before the displayed deadline. The 48-hour (two-day) window starts at the invitation effective time and is a return/submission deadline, not expected working time.

AI may assist implementation. Use your own logic in the decision journal and be ready to discuss it without AI assistance during an on-site interview.
