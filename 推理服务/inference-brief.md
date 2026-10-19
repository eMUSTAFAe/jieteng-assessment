# Inference service

Measure a bounded inference service and choose a serving policy that balances waiting, predictability and resource limits.

Hardware and model access: Inference machine needed · CPU sufficient
Use a local or remote machine for the small model and load experiments. CPU is sufficient for the supplied scope; a GPU is optional for models that need it.

Your role: Engineer responsible for a shared inference service.

Decision scenario: Demand can exceed capacity. Some users prefer quick best-effort responses; others need predictable completion. A throughput improvement can increase tail latency or make cancellation harder to handle.

Affected stakeholders: Interactive users, batch users and the operations team.

Scope and constraints: Use a small local service and reproducible load cases. No large GPU deployment or production capacity claim is required.

Decision to defend: How would you allocate limited capacity and choose a release policy, and who should bear the cost of overload?

This is an assessment scenario, not a claim about measured results. State any additional assumptions; you may choose a different reasonable option. Use your code and run results to support your own judgment, and explain what remains unknown.

Submitted work
Provide a bounded, reproducible implementation or experiment, run instructions, results and limitations that satisfy the requirements below. AI assistance is allowed when disclosed. A larger codebase or system does not by itself improve the quality of the submitted results.

Requirements:
- Add validation, bounded concurrency or batching.
- Measure latency, throughput and errors against baseline.
- Test overload, ordering, cancellation and timeouts.

Personal judgment
Use this scenario and decisions from your actual work to explain your priorities, stakeholder tradeoffs, uncertainty and reflection in the four decision-journal questions. Several conclusions can be defensible; there is no required preferred opinion. Be ready to discuss your reasoning during an on-site interview.
- Your priorities and values: Choose the user experience you prioritize under limited serving capacity. Explain your values and what you would sacrifice.
- Your tradeoff and stakeholder judgment: Compare admission control, longer waits and degraded responses. Defend who should bear the costs of your policy.
- Your judgment under uncertainty: A faster configuration is less predictable under load. Explain your release decision, uncertainty and run results or user feedback that would change your view.
- Your ownership and reflection: Describe one optimization or AI suggestion you rejected, changed or retained. Explain your reasoning and accountability to users.

Run the supplied baseline with Python 3: python inference-baseline.py
Extend it within the stated task scope. Include source files, exact run commands, output tables or plots, failed cases and known limitations.

We recommend selecting 1–3 tasks; choosing more is allowed. Save answers as you work and return before the displayed deadline. The 48-hour (two-day) window starts at the invitation effective time and is a return/submission deadline, not expected working time.

AI may assist implementation. Use your own logic in the decision journal and be ready to discuss it without AI assistance during an on-site interview.
