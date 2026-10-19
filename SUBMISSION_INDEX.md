# 捷誊技术笔试 — 完成索引

四个任务均按同一交付标准完成：**可复现实现 + 结果表/图 + README + 四问决策日志 + AI 披露**。每个任务目录内的 supplied baseline 和 brief 均保持未改动。

| # | 任务 | 核心结论 | 目录 |
|---|---|---|---|
| 1 | Cost-sensitive decision models | 推荐逻辑回归 @ 阈值 0.251；test 期望成本 0.495；显式报告 235 FP/1,000 与人工复核建议 | `成本敏感决策模型/` |
| 2 | Evaluation runner | 分层排名：`fast` 第一层、`transient` 第二层、`slow`/`truncating` 不推荐且不严格互排；失败全部保留 | `评估运行器/` |
| 3 | Inference service | 推荐 `priority_batched`（8w/batch8/priority）：overload 699 rps、p95 104ms、0 错误；超出容量时 batch 先被 shed | `推理服务/` |
| 4 | Small transformer experiment | 报告 causal mask 模型（test loss 1.487 ≈ Bayes 1.501）；unmasked 的 0.113 是 target leakage | `小型transformer实验/` |

## 每个任务的入口

| 任务 | 报告 | 决策日志 | 主脚本 | 结果图 |
|---|---|---|---|---|
| 成本敏感 | `成本敏感决策模型/README.md` | `成本敏感决策模型/decision-journal.md` | `cost_models_experiment.py` | `results/figures/fig1_dev_cost_curves.png` 等 |
| 评估运行器 | `评估运行器/README.md` | `评估运行器/decision-journal.md` | `runner_experiment.py` | `results/figures/fig1_exclusion_bias.png` 等 |
| 推理服务 | `推理服务/README.md` | `推理服务/decision-journal.md` | `service_experiment.py` + `combined_policy.py` | `results/figures/fig1_overload.png` 等 |
| Transformer | `小型transformer实验/README.md` | `小型transformer实验/decision-journal.md` | `transformer_experiment.py` + `tiny_attention.py` | `results/figures/fig1_training_curves.png` 等 |

## 运行

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

## 提交说明

- **AI 使用已按任务要求披露**：每个 README 的第 9/10 节和每份 decision journal 顶部都说明了 AI 辅助范围。
- **决策日志是完整版本**，包含四个问题的回答和基于本 assessment 的反思段落；没有待填占位符。
- **面试准备**：on-site 可能要求不借助 AI 解释决策。建议重点准备四组问题：
  1. 成本敏感：为什么 0.251 而不是 0.136，FP 后果如何披露；
  2. 评估运行器：为什么拒绝 success-only 排名，slow/truncating 为什么不做严格互排；
  3. 推理服务：为什么让 batch 用户承担 overload，wide 16-worker 为什么不是默认发布；
  4. Transformer：为什么 unmasked 的低 loss 不算性能优势，因果掩码如何成为有效基线。
- 各任务 README 末尾均有 **Requirement checklist**，可逐条对照 brief。
- 所有实验均可离线复现；数据与结果已保存在各任务目录内。
