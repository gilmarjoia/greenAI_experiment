# GreenAI Experiment — Round 6 Summary Report

Generated: 2026-06-13 18:59

## Experiment Setup (Round 6 — 30 Epoch Baseline — Final Round)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 30 |
| Batch size | 16 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

> **Goal**: Evaluate baseline hyperparameters (no data augmentations or extra
> regularization beyond baseline defaults) with triple the baseline epochs (30).
> This is the **final** experiment round, establishing the 30-epoch baseline ceiling.

---

## Training Metrics (last epoch)

| Model | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 6 | 30.0 | 0.37756 | 0.26654 | 90.13 | 99.89 | 206.06 |
| CNN Round 6 | 30.0 | 0.00403 | 0.26541 | 93.24 | 99.81 | 68.48 |
| ViT (Transformers) Round 6 | 30.0 | 6e-05 | 0.34194 | 95.01 | 99.66 | 246.13 |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Model | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Round 6 | 12460.37 | 0.075351 | 0.020659 | 0.020084 | 0.034608 | 7.4106 |
| CNN Round 6 | 4112.49 | 0.04617 | 0.006861 | 0.02788 | 0.011429 | 4.5407 |
| ViT (Transformers) Round 6 | 14774.97 | 0.236705 | 0.068941 | 0.126721 | 0.041043 | 23.2795 |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
| YOLO26 Round 6 | ✓ | ✓ |
| CNN Round 6 | ✓ | ✓ |
| ViT (Transformers) Round 6 | ✓ | ✓ |
