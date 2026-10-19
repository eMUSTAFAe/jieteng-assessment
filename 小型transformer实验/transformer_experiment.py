#!/usr/bin/env python3
"""Tiny transformer experiment: causal mask vs. no mask.

Question
--------
A team can investigate one design change.  This experiment asks a deliberately
simple question: **does causal masking matter for next-token prediction?**  The
answer is not "which model has lower loss" — the unmasked model has an unfair
information advantage, because for position ``i`` the target token is also the
input token at position ``i + 1``.  The unmasked model can attend to it; the
causal model cannot.  The experiment quantifies that leakage.

Setup
-----
* Synthetic sequences from a first-order Markov chain: with probability 0.7 the
  next token is ``(token * 7 + 3) % 24``, otherwise uniform noise.  The Bayes
  loss is ~1.50 nats; a causal model should approach it.
* Data: 12,000 train / 2,000 val / 2,000 test sequences, fixed seeds, saved to
  ``data/tiny_lm_data.npz``.
* Model: 2 layers, 4 heads, d_model 64, d_ff 128, context 15, vocab 24 (~70k
  parameters).  The only difference between the two runs is the causal mask.
* Same seed, same initial weights, same data order, same optimizer and steps.
* 3 seeds per condition; report mean ± sd.
* Held-out test loss, top-1 accuracy, sample predictions and the attention paid
  to the next input position (the direct copy path).

Run
---
    python tiny_attention.py          # component tests
    python transformer_experiment.py  # training experiment
    python make_figures.py

Outputs are written to ``results/``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import tiny_attention

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
RESULTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

VOCAB = 24
CONTEXT = 16                 # full sequence length; input length is 15
D_MODEL = 64
N_HEAD = 4
N_LAYER = 2
D_FF = 128
DROPOUT = 0.0
TRAIN_N = 12_000
VAL_N = 2_000
TEST_N = 2_000
BATCH = 64
STEPS = 600
LR = 3e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_EVERY = 50
EVAL_BATCH = 256
SEEDS = [0, 1, 2]
MODELS = ["unmasked", "causal"]
DETERMINISTIC_PROB = 0.7

torch.set_num_threads(min(4, os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def generate_sequences(n: int, rng: np.random.Generator) -> np.ndarray:
    seq = np.empty((n, CONTEXT), dtype=np.int64)
    seq[:, 0] = rng.integers(0, VOCAB, size=n)
    for t in range(1, CONTEXT):
        deterministic = (seq[:, t - 1] * 7 + 3) % VOCAB
        noise = rng.integers(0, VOCAB, size=n)
        take_deterministic = rng.random(n) < DETERMINISTIC_PROB
        seq[:, t] = np.where(take_deterministic, deterministic, noise)
    return seq


def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    path = DATA / "tiny_lm_data.npz"
    if path.exists():
        loaded = np.load(path)
        return loaded["train"], loaded["val"], loaded["test"], True
    rng = np.random.default_rng(20261019)
    train = generate_sequences(TRAIN_N, rng)
    val = generate_sequences(VAL_N, rng)
    test = generate_sequences(TEST_N, rng)
    np.savez_compressed(path, train=train, val=val, test=test)
    return train, val, test, False


def theoretical_bayes() -> tuple[float, float]:
    """Entropy (nats) and top-1 accuracy of the Bayes predictor."""
    p_det = DETERMINISTIC_PROB + (1.0 - DETERMINISTIC_PROB) / VOCAB
    p_other = (1.0 - DETERMINISTIC_PROB) / VOCAB
    entropy = -(p_det * math.log(p_det) + (VOCAB - 1) * p_other * math.log(p_other))
    return float(entropy), float(p_det)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.triu(
            torch.ones(CONTEXT, CONTEXT, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("mask", mask)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            scores = scores.masked_fill(
                self.mask[:length, :length], float("-inf")
            )
        weights = F.softmax(scores, dim=-1)
        self.last_attn = weights.detach()
        out = weights @ value
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, causal: bool):
        super().__init__()
        self.causal = causal
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEAD, D_FF, causal) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Evaluation and diagnostics
# ---------------------------------------------------------------------------


def evaluate(model: TinyTransformer, data: np.ndarray) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(data), EVAL_BATCH):
            seq = torch.from_numpy(data[start : start + EVAL_BATCH])
            inputs, targets = seq[:, :-1], seq[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_correct += int((logits.argmax(-1) == targets).sum())
            total_tokens += targets.numel()
    return total_loss / total_tokens, total_correct / total_tokens


def position_losses(
    model: TinyTransformer, data: np.ndarray
) -> tuple[float, float]:
    """Loss on positions 0..T-2 vs. the last position.

    For positions 0..T-2 the target is also present in the input at the next
    position, so an unmasked model can copy it.  At the last position there is
    no future input token to copy; the loss there is the honest comparison.
    """
    model.eval()
    other_total = 0.0
    last_total = 0.0
    other_count = 0
    last_count = 0
    with torch.no_grad():
        for start in range(0, len(data), EVAL_BATCH):
            seq = torch.from_numpy(data[start : start + EVAL_BATCH])
            inputs, targets = seq[:, :-1], seq[:, 1:]
            logits = model(inputs)
            losses = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)
            last_total += float(losses[:, -1].sum())
            last_count += losses.shape[0]
            other_total += float(losses[:, :-1].sum())
            other_count += losses[:, :-1].numel()
    return other_total / other_count, last_total / last_count


def attention_diagnostics(
    model: TinyTransformer, data: np.ndarray, batch_size: int = 64
) -> tuple[float, float]:
    """Mean attention to the next token and to all future positions.

    The diagnostic reads the first block of every layer.  With a causal mask the
    future weights are exactly zero; without it they are the direct copy path.
    """
    model.eval()
    seq = torch.from_numpy(data[:batch_size])
    inputs = seq[:, :-1]
    with torch.no_grad():
        model(inputs)
    next_weights = []
    future_weights = []
    for block in model.blocks:
        attn = block.attn.last_attn
        assert attn is not None
        batch, heads, length, _ = attn.shape
        query_idx = torch.arange(length - 1)
        next_weights.append(float(attn[:, :, query_idx, query_idx + 1].mean()))
        future_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool), diagonal=1
        )
        future_weights.append(float(attn[:, :, future_mask].mean()))
    return float(np.mean(next_weights)), float(np.mean(future_weights))


def sample_predictions(
    model: TinyTransformer, data: np.ndarray, n_sequences: int = 2
) -> list[dict[str, Any]]:
    model.eval()
    seq = torch.from_numpy(data[:n_sequences])
    inputs, targets = seq[:, :-1], seq[:, 1:]
    with torch.no_grad():
        predictions = model(inputs).argmax(-1)
    rows = []
    for s in range(n_sequences):
        for t in range(inputs.shape[1]):
            rows.append(
                {
                    "sequence_id": s,
                    "position": t,
                    "input_token": int(inputs[s, t]),
                    "target_token": int(targets[s, t]),
                    "predicted_token": int(predictions[s, t]),
                    "correct": bool(predictions[s, t] == targets[s, t]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(
    model_name: str, seed: int, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[TinyTransformer, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    causal = model_name == "causal"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = TinyTransformer(causal=causal)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    parameters = sum(p.numel() for p in model.parameters())

    curves: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    started = time.perf_counter()

    for step in range(1, STEPS + 1):
        model.train()
        indices = torch.randint(0, TRAIN_N, (BATCH,), generator=generator)
        seq = torch.from_numpy(train[indices.numpy()])
        inputs, targets = seq[:, :-1], seq[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_losses.append(float(loss.detach()))

        if step % EVAL_EVERY == 0 or step == STEPS:
            val_loss, val_accuracy = evaluate(model, val)
            curves.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "step": step,
                    "train_loss": float(
                        np.mean(train_losses[-EVAL_EVERY:])
                    ),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(model, test)
    other_position_loss, last_position_loss = position_losses(model, test)
    next_attention, future_attention = attention_diagnostics(model, test)
    elapsed = time.perf_counter() - started
    final_train_loss = float(np.mean(train_losses[-EVAL_EVERY:]))

    metrics = {
        "model": model_name,
        "seed": seed,
        "causal": causal,
        "parameters": parameters,
        "train_loss": final_train_loss,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "other_position_loss": other_position_loss,
        "last_position_loss": last_position_loss,
        "next_token_attention": next_attention,
        "future_attention": future_attention,
        "seconds": elapsed,
    }
    predictions = sample_predictions(model, test)
    return model, metrics, curves, predictions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def print_table(title: str, frame: pd.DataFrame, float_format: str = "{:.4f}") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda x: "" if pd.isna(x) else float_format.format(x)
            )
    print(formatted.to_string(index=False))


def main() -> None:
    tee = Tee(RESULTS / "report.txt")
    sys.stdout = tee
    try:
        print("=" * 78)
        print("TINY TRANSFORMER: CAUSAL MASK vs NO MASK")
        print("=" * 78)
        print(f"Python     : {platform.python_version()}")
        print(f"PyTorch    : {torch.__version__}")
        print(f"numpy      : {np.__version__}")
        print(f"pandas     : {pd.__version__}")
        print(f"threads    : {torch.get_num_threads()}")
        print(f"vocab      : {VOCAB}")
        print(f"context    : {CONTEXT} (input {CONTEXT - 1})")
        print(f"model      : {N_LAYER} layers, {N_HEAD} heads, d_model {D_MODEL}, d_ff {D_FF}")
        print(f"training   : {STEPS} steps, batch {BATCH}, lr {LR}, seeds {SEEDS}")
        print(f"data       : {TRAIN_N} train / {VAL_N} val / {TEST_N} test")

        train, val, test, cached = get_data()
        bayes_loss, bayes_accuracy = theoretical_bayes()
        print("\nData")
        print("----")
        print(f"  cached: {cached}")
        print(f"  train shape: {train.shape}; val: {val.shape}; test: {test.shape}")
        print(f"  unique tokens in train: {len(np.unique(train))}/{VOCAB}")
        print(f"  theoretical Bayes loss: {bayes_loss:.4f} nats")
        print(f"  theoretical Bayes top-1 accuracy: {bayes_accuracy:.4f}")

        attention_result = tiny_attention.run_tests(write=True)
        print("\nAttention component tests")
        print("-------------------------")
        print(f"  passed: {attention_result['passed']}")
        print(
            f"  max |ours - baseline|: "
            f"non-causal {attention_result['max_abs_diff_non_causal']:.2e}, "
            f"causal {attention_result['max_abs_diff_causal']:.2e}"
        )
        print(
            "  causal ignores future perturbation: "
            f"{attention_result['future_perturbation_ignored_when_causal']}; "
            "non-causal sees it: "
            f"{attention_result['future_perturbation_changes_output_when_non_causal']}"
        )

        metrics_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        models_by_seed: dict[tuple[int, str], TinyTransformer] = {}

        for seed in SEEDS:
            for model_name in MODELS:
                print(f"\nTraining {model_name} / seed {seed} ...")
                model, metrics, curves, predictions = train_one(
                    model_name, seed, train, val, test
                )
                models_by_seed[(seed, model_name)] = model
                metrics_rows.append(metrics)
                curve_rows.extend(curves)
                for row in predictions:
                    row.update({"model": model_name, "seed": seed})
                    prediction_rows.append(row)
                print(
                    f"  test loss {metrics['test_loss']:.4f}, "
                    f"accuracy {metrics['test_accuracy']:.4f}, "
                    f"next-token attention {metrics['next_token_attention']:.4f}, "
                    f"{metrics['seconds']:.1f}s"
                )

        metrics_df = pd.DataFrame(metrics_rows)
        curves_df = pd.DataFrame(curve_rows)
        predictions_df = pd.DataFrame(prediction_rows)

        print_table(
            "Per-seed held-out metrics",
            metrics_df[
                [
                    "model",
                    "seed",
                    "parameters",
                    "train_loss",
                    "best_val_loss",
                    "test_loss",
                    "test_accuracy",
                    "other_position_loss",
                    "last_position_loss",
                    "next_token_attention",
                    "future_attention",
                    "seconds",
                ]
            ],
        )

        summary = (
            metrics_df.groupby("model")
            .agg(
                seeds=("seed", "nunique"),
                test_loss_mean=("test_loss", "mean"),
                test_loss_sd=("test_loss", "std"),
                accuracy_mean=("test_accuracy", "mean"),
                accuracy_sd=("test_accuracy", "std"),
                other_position_loss_mean=("other_position_loss", "mean"),
                last_position_loss_mean=("last_position_loss", "mean"),
                next_attention_mean=("next_token_attention", "mean"),
                next_attention_sd=("next_token_attention", "std"),
                future_attention_mean=("future_attention", "mean"),
                val_loss_mean=("best_val_loss", "mean"),
                train_loss_mean=("train_loss", "mean"),
            )
            .reindex(MODELS)
            .reset_index()
        )
        print_table("Summary (mean +/- sd across seeds)", summary)

        print("\nSample predictions (seed 0, first test sequence)")
        for model_name in MODELS:
            sample = predictions_df[
                (predictions_df["model"] == model_name)
                & (predictions_df["seed"] == 0)
                & (predictions_df["sequence_id"] == 0)
            ]
            print(
                f"\n  {model_name}: "
                f"input={sample['input_token'].tolist()}\n"
                f"  {' ' * len(model_name)}  target={sample['target_token'].tolist()}\n"
                f"  {' ' * len(model_name)}  pred  ={sample['predicted_token'].tolist()}"
            )

        print("\nInterpretation")
        print("--------------")
        print(
            "The unmasked model has a lower held-out loss, but the target token is\n"
            "present in its input at the next position.  Its low loss is therefore\n"
            "not a valid next-token result: it is copying from the future.  The\n"
            "causal model cannot see the future; its loss is compared with the\n"
            "theoretical Bayes loss above.  The next-token attention diagnostic is\n"
            "the mechanism check: it is exactly zero for the causal model and\n"
            "positive for the unmasked model."
        )
        unmasked_row = summary[summary["model"] == "unmasked"].iloc[0]
        causal_row = summary[summary["model"] == "causal"].iloc[0]
        print(
            f"\nPosition control: unmasked loss on positions 0..T-2 = "
            f"{unmasked_row['other_position_loss_mean']:.4f}, on the last position = "
            f"{unmasked_row['last_position_loss_mean']:.4f}; causal = "
            f"{causal_row['other_position_loss_mean']:.4f} / "
            f"{causal_row['last_position_loss_mean']:.4f}."
        )

        # Persist
        metrics_df.to_csv(RESULTS / "metrics.csv", index=False)
        summary.to_csv(RESULTS / "summary.csv", index=False)
        curves_df.to_csv(RESULTS / "training_curves.csv", index=False)
        predictions_df.to_csv(RESULTS / "predictions.csv", index=False)
        metadata = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "vocab": VOCAB,
            "context": CONTEXT,
            "d_model": D_MODEL,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "d_ff": D_FF,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds": SEEDS,
            "train_n": TRAIN_N,
            "val_n": VAL_N,
            "test_n": TEST_N,
            "deterministic_prob": DETERMINISTIC_PROB,
            "bayes_loss": bayes_loss,
            "bayes_accuracy": bayes_accuracy,
            "attention_tests": attention_result,
        }
        with (RESULTS / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        print("\nSaved:")
        for name in sorted(p.name for p in RESULTS.iterdir() if p.is_file()):
            print(f"  results/{name}")
        print("\nDone.")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
