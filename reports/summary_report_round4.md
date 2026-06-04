# GreenAI Experiment — Round 4 Summary Report

Generated: 2026-06-04 16:33

## Experiment Setup (Round 4 — Efficiency Benchmark)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 10 (same as Baseline — efficiency benchmark) |
| Batch size | 32 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

> **Goal**: Use the full Round 3 hyperparameter set (all regularisation and
> augmentation improvements) but train for only 10 epochs — the same budget as
> the original Baseline. This isolates the effect of *better hyperparameters*
> from the effect of *more training time*.

---

## Hyperparameter Changes vs Round 3

| Model | Change for Round 4 |
|---|---|
| **YOLO26** | epochs 30→10 · all other params unchanged from Round 3 |
| **CNN** | epochs 30→10 · warmup_epochs 5→3 (proportional) · all other params unchanged |
| **ViT** | epochs 30→10 · warmup_epochs 5→2 (proportional) · all other params unchanged |

---

## Training Metrics (last epoch)

| Model | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 4 | 10.0 | 0.39566 | 0.24812 | 90.63 | 99.92 | 18.51 |
| CNN Round 4 | 10.0 | 0.65821 | 0.66092 | 93.31 | 99.85 | 36.65 |
| ViT (Transformers) Round 4 | 10.0 | 0.33729 | 0.42517 | 95.08 | 99.87 | 126.07 |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Model | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 4 | 1199.68 | 0.007804 | 0.002171 | 0.002299 | 0.003334 | 0.7675 |
| CNN Round 4 | 2207.33 | 0.018948 | 0.002946 | 0.009869 | 0.006133 | 1.8635 |
| ViT (Transformers) Round 4 | 7574.09 | 0.086705 | 0.032202 | 0.033457 | 0.021045 | 8.5272 |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
| YOLO26 Round 4 | ✓ | ✓ |
| CNN Round 4 | ✓ | ✓ |
| ViT (Transformers) Round 4 | ✓ | ✓ |
