# GreenAI Experiment — Round 3 Summary Report

Generated: 2026-06-01 00:54

## Experiment Setup (Round 3)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 30 (Target) |
| Batch size | 32 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

---

## Hyperparameter Changes vs Round 2

| Model | Key Changes |
|---|---|
| **YOLO26** | epochs 20→30 · lr0 0.001→0.0012 · lrf 0.01→0.005 · cos_lr=True · dropout 0.0→0.1 · weight_decay 0.05→0.001 |
| **CNN** | epochs 20→30 · dropout 0.3→0.2 · lr0 0.01→0.015 · lrf 0.01→0.005 · weight_decay 0.001→0.0005 · warmup 3→5 |
| **ViT** | epochs 20→30 · lr0 5e-5→8e-5 · lrf 0.01→0.02 · weight_decay 0.1→0.05 · warmup 3→5 · label_smoothing 0.1→0.05 |

---

## Training Metrics (last epoch)

| Model | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 3 | 30.0 | 0.33232 | 0.21692 | 91.74 | 99.93 | 109.79 |
| CNN Round 3 | 30.0 | 0.61377 | 0.64317 | 93.96 | 99.84 | 227.71 |
| ViT (Transformers) Round 3 | 30.0 | 0.28735 | 0.46722 | 95.18 | 99.13 | 365.59 |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Model | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 3 | 6731.53 | 0.039193 | 0.009833 | 0.010661 | 0.0187 | 3.8546 |
| CNN Round 3 | 13666.76 | 0.155015 | 0.006657 | 0.110392 | 0.037965 | 15.2454 |
| ViT (Transformers) Round 3 | 21942.21 | 0.325459 | 0.069511 | 0.194995 | 0.060952 | 32.0082 |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
| YOLO26 Round 3 | ✓ | ✓ |
| CNN Round 3 | ✓ | ✓ |
| ViT (Transformers) Round 3 | ✓ | ✓ |
