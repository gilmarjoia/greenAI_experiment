# GreenAI Experiment — Summary Report

Generated: 2026-05-10 13:47

## Experiment Setup

| Parameter  | Value                                              |
|------------|----------------------------------------------------|
| Dataset    | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs     | 10                                                 |
| Batch size | 16                                                 |
| Seed       | 0                                                  |
| Precision  | AMP (FP16 where available)                         |

---

## Training Metrics (last epoch)

| Baseline           | Epochs | Train Loss | Val Loss | Top-1 Acc (%) | Top-5 Acc (%) | Train Time (min) |
|--------------------|--------|------------|----------|---------------|---------------|------------------|
| YOLO26             | 10.0   | 0.42689    | 0.264    | 90.56         | 99.91         | 48.56            |
| CNN                | 10.0   | 0.04193    | 0.2079   | 93.44         | 99.93         | 9.51             |
| ViT (Transformers) | 10.0   | 0.02121    | 0.18363  | 94.97         | 99.94         | 99.23            |

> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

| Baseline           | Duration (s) | Energy (kWh) | GPU Energy (kWh) | CPU Energy (kWh) | RAM Energy (kWh) | Emissions (gCO₂) |
|--------------------|--------------|--------------|------------------|------------------|------------------|------------------|
| YOLO26             | 3165.85      | 0.019593     | 0.004887         | 0.005907         | 0.0088           | 1.927            |
| CNN                | 576.29       | 0.004045     | 0.001115         | 0.001327         | 0.001603         | 0.3978           |
| ViT (Transformers) | 5967.29      | 0.103272     | 0.02675          | 0.059944         | 0.016578         | 10.1566          |

> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Baseline           | Results CSV | Emissions CSV |
|--------------------|-------------|---------------|
| YOLO26             | ✓           | ✓             |
| CNN                | ✓           | ✓             |
| ViT (Transformers) | ✓           | ✓             |
