# Jieteng technical assessment — four tasks

Four bounded, reproducible engineering assessments. Each task includes:

* source code and pinned requirements;
* a `README.md` with results, exact run commands, assumptions and limitations;
* a `decision-journal.md` answering the four personal-judgment questions;
* result tables, figures and the supplied baseline/brief kept unchanged.

| Task | Core decision / result | Entry points |
|---|---|---|
| **Cost-sensitive decision models** | Recommend logistic regression at threshold **0.251**; held-out expected cost 0.495; report the 235 FP/1,000 and the human-review consequence instead of hiding it | [README](成本敏感决策模型/README.md) · [journal](成本敏感决策模型/decision-journal.md) |
| **Evaluation runner** | Publish a **tiered** ranking, not a strict 1–2–3–4: `fast` tier 1, `transient` tier 2, `slow`/`truncating` not recommended and not strictly ranked; keep all 142 final failures visible | [README](评估运行器/README.md) · [journal](评估运行器/decision-journal.md) |
| **Inference service** | Release **`priority_batched`** (8 workers, batch 8, priority): under overload 100% success, 0 errors, **699 rps goodput, 104 ms p95**; batch users bear the cost beyond capacity | [README](推理服务/README.md) · [journal](推理服务/decision-journal.md) |
| **Small transformer experiment** | Report the **causal-mask** model (test loss **1.487** ≈ Bayes **1.501**); the unmasked **0.113** is target leakage, not a performance win | [README](小型transformer实验/README.md) · [journal](小型transformer实验/decision-journal.md) |

## Quick start

```bash
cd 成本敏感决策模型
python -m pip install -r requirements.txt && python cost_models_experiment.py && python make_figures.py

cd ../评估运行器
python -m pip install -r requirements.txt && python runner_experiment.py && python make_figures.py

cd ../推理服务
python -m pip install -r requirements.txt && python service_experiment.py && python combined_policy.py && python make_figures.py

cd ../小型transformer实验
python -m pip install -r requirements.txt && python tiny_attention.py && python transformer_experiment.py && python make_figures.py
```

Each task's README ends with a **Requirement checklist** mapping the brief to the
delivered evidence.

## Repository layout

```
成本敏感决策模型/      cost-sensitive LR vs DT experiment
评估运行器/            repeatable adapter evaluation runner
推理服务/              bounded inference service + load experiments
小型transformer实验/   tiny attention component + causal-mask experiment
SUBMISSION_INDEX.md    Chinese index with run commands and interview focus
*.pdf                  supplied assessment materials, included for context
```

## Selected results

* Cost-sensitive: **LR @ 0.251**, test cost **0.495**, and an explicit FP count
  (235 per 1,000) rather than an average that hides it.
* Evaluation runner: `slow` is **100%** accurate on completed runs but **65%**
  when every run is counted; the report keeps the 70 timeouts and 70 silent wrong
  answers in `results/failure_cases.csv`.
* Inference service: the measured combined policy reaches **699 rps goodput,
  104 ms p95, 0% errors** under overload; nominal p95 is 62 ms because batching
  is a throughput tool, not a low-latency default.
* Transformer: unmasked loss **0.0020** on positions 0–13 (target present in the
  input) vs **1.6701** on the last position (no future token); causal is
  **1.4821 / 1.5604**. The leakage is localised exactly where the target is
  available.

## AI assistance disclosure

AI assistance (Claude Code) was used to implement the experiments, generate the
figures and draft the documentation. Each task's README and decision journal
discloses this. The candidate owns the final submission and is responsible for
defending the reasoning without AI assistance during an on-site interview.

A more detailed Chinese index is in [`SUBMISSION_INDEX.md`](SUBMISSION_INDEX.md).
