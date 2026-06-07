# GreenAI Experiment — Round 5 Summary Report

Generated: 2026-06-07 15:03

## Experiment Setup (Round 5 — 20 Epoch Baseline)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 20 |
| Batch size | 16 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

> **Goal**: Evaluate baseline hyperparameters (no data augmentations or extra
> regularization beyond baseline defaults) with double the baseline epochs (20).

---

## Training Metrics (last epoch)

| Model | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 5 | 20.0 | 0.40827 | 0.27709 | 89.76 | 99.88 | 69.82 |
| CNN Round 5 | 20.0 | 0.00699 | 0.2545 | 93.23 | 99.83 | 30.19 |
| ViT (Transformers) Round 5 | 20.0 | 0.00048 | 0.28441 | 95.03 | 99.77 | 153.7 |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Model | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 5 | 4289.87 | 0.027562 | 0.00867 | 0.006974 | 0.011918 | 2.7107 |
| CNN Round 5 | 1815.05 | 0.024957 | 0.003479 | 0.016435 | 0.005043 | 2.4544 |
| ViT (Transformers) Round 5 | 9232.21 | 0.15415 | 0.044094 | 0.084409 | 0.025647 | 15.1603 |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
| YOLO26 Round 5 | ✓ | ✓ |
| CNN Round 5 | ✓ | ✓ |
| ViT (Transformers) Round 5 | ✓ | ✓ |
