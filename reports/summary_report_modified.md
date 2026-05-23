# GreenAI Experiment — Modified Models Summary Report

Generated: 2026-05-23 19:41

## Experiment Setup (Modified Phase)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 20 (Target) |
| Batch size | 32 (Target) |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

---

## Training Metrics (last epoch)

| Model | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Modified | 20.0 | 0.348 | 0.22269 | 91.62 | 99.92 | 41.54 |
| CNN Modified | 20.0 | 0.64971 | 0.65565 | 93.19 | 99.83 | 49.77 |
| ViT (Transformers) Modified | 20.0 | 0.51631 | 0.64883 | 95.17 | 99.33 | 203.45 |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Model | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 Modified | 2576.75 | 0.017802 | 0.00462 | 0.006023 | 0.007159 | 1.7508 |
| CNN Modified | 2990.11 | 0.039048 | 0.003952 | 0.026787 | 0.008308 | 3.8403 |
| ViT (Transformers) Modified | 12228.47 | 0.200333 | 0.053759 | 0.112605 | 0.033968 | 19.7023 |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
| YOLO26 Modified | ✓ | ✓ |
| CNN Modified | ✓ | ✓ |
| ViT (Transformers) Modified | ✓ | ✓ |
