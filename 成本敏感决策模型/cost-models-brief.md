# Cost-sensitive decision models

Compare cost-sensitive models and defend a recommendation that accounts for who benefits and who bears errors.

Hardware and model access: Modeling machine needed · CPU sufficient
Use a standard CPU machine for logistic regression and decision-tree experiments. No GPU is required.

Your role: Engineer recommending a model and threshold.

Decision scenario: Two models make different mistakes. A business cost table favors one, but it may omit a consequence borne by a less visible stakeholder; a different threshold redistributes those mistakes.

Affected stakeholders: The business owner, people affected by predictions and those investigating errors.

Scope and constraints: Use small reproducible data splits and the supplied model comparison. Keep held-out evaluation separate from tuning, and identify assumed costs.

Decision to defend: Which model and threshold would you recommend, what consequence would you refuse to hide, and how would you defend the tradeoff?

This is an assessment scenario, not a claim about measured results. State any additional assumptions; you may choose a different reasonable option. Use your code and run results to support your own judgment, and explain what remains unknown.

Submitted work
Provide a bounded, reproducible implementation or experiment, run instructions, results and limitations that satisfy the requirements below. AI assistance is allowed when disclosed. A larger codebase or system does not by itself improve the quality of the submitted results.

Requirements:
- Compare logistic regression and a decision tree.
- Select thresholds on development data only.
- Report held-out confusion matrices and expected costs.

Personal judgment
Use this scenario and decisions from your actual work to explain your priorities, stakeholder tradeoffs, uncertainty and reflection in the four decision-journal questions. Several conclusions can be defensible; there is no required preferred opinion. Be ready to discuss your reasoning during an on-site interview.
- Your priorities and values: Choose which errors this system should prioritize avoiding. Explain your values and the affected stakeholders.
- Your tradeoff and stakeholder judgment: Compare two models that distribute benefits and harms differently. Explain why your preferred tradeoff is defensible.
- Your judgment under uncertainty: The business cost table omits a stakeholder consequence. Explain whether and how you would challenge it before recommending a model.
- Your ownership and reflection: Describe one threshold or model recommendation you accepted, rejected or revised. Explain the judgment beyond the reported metric.

Run the supplied baseline with Python 3: python cost-models-baseline.py
Extend it within the stated task scope. Include source files, exact run commands, output tables or plots, failed cases and known limitations.

We recommend selecting 1–3 tasks; choosing more is allowed. Save answers as you work and return before the displayed deadline. The 48-hour (two-day) window starts at the invitation effective time and is a return/submission deadline, not expected working time.

AI may assist implementation. Use your own logic in the decision journal and be ready to discuss it without AI assistance during an on-site interview.
