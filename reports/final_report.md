# GreenAI Experiment — Final Report (4-Phase Comparison)

Generated: 2026-06-04 16:39

This report summarises the complete four-phase experimental study comparing CNN,
YOLO26, and ViT (DeiT-Tiny) models trained on Fashion-MNIST, with successive
hyperparameter refinements aimed at improving accuracy while monitoring
energy consumption and carbon emissions.

---

## 1. Experimental Phases Overview

| Phase | Epochs | Batch | Key Goal |
|:------|:------:|:-----:|:---------|
| **Baseline** | 10 | 16 | Establish reference performance with default hyperparameters |
| **Round 2 (Modified)** | 20 | 32 | Address overfitting/underfitting; add regularization & augmentation |
| **Round 3** | 30 | 32 | Fine-tune further; more epochs; balance LR decay and regularization |
| **Round 4 (Efficiency)** | 10 | 32 | Round 3 hyperparams at Baseline epoch budget — isolate hyperparameter effect |

---

## 2. Final Accuracy Comparison (Top-1, Test Set)

| Model | Baseline | Round 2 | Δ R2 | Round 3 | Δ R3 vs R2 | Round 4 | Δ R4 vs Base |
|:------|:--------:|:-------:|:----:|:-------:|:----------:|:-------:|:------------:|
| **YOLO26** | 90.56% | 91.62% | +1.06% | 91.74% | +0.12% | 90.63% | +0.07% |
| **CNN** | 93.44% | 93.19% | -0.25% | 93.96% | +0.77% | 93.31% | -0.13% |
| **ViT** | 94.97% | 95.17% | +0.20% | 95.18% | +0.01% | 95.08% | +0.11% |

> Δ values are absolute percentage point differences.

---

## 3. Carbon Emissions Comparison (gCO₂eq)

| Model | Baseline | Round 2 | Round 3 | Round 4 | R4 vs Base |
|:------|:--------:|:-------:|:-------:|:-------:|:----------:|
| **YOLO26** | 1.927g | 1.751g | 3.855g | 0.767g | **-60.2%** |
| **CNN** | 0.398g | 3.840g | 15.245g | 1.863g | **+368.5%** |
| **ViT** | 10.157g | 19.702g | 32.008g | 8.527g | **-16.0%** |

> All values in grams of CO₂ equivalent (gCO₂eq) measured by [CodeCarbon](https://codecarbon.io/).

---

## 4. Energy Consumption Comparison (kWh)

| Model | Baseline | Round 2 | Round 3 | Round 4 | R4 vs Base |
|:------|:--------:|:-------:|:-------:|:-------:|:----------:|
| **YOLO26** | 0.019593 | 0.017802 | 0.039193 | 0.007804 | **-60.2%** |
| **CNN** | 0.004045 | 0.039048 | 0.155015 | 0.018948 | **+368.4%** |
| **ViT** | 0.103272 | 0.200333 | 0.325459 | 0.086705 | **-16.0%** |

---

## 5. Round 4 — Efficiency Benchmark Analysis

### Round 4 vs Baseline (same epochs, different hyperparams)

| Model | Δ Accuracy | Δ CO₂ | Interpretation |
|:------|:----------:|:-----:|:---------------|
| **YOLO26** | +0.07% | -60.2% | Hiperparâmetros melhorados oferecem mesma acurácia com 60% menos emissões |
| **CNN** | -0.13% | +368.5% | Regularização aumenta custo; ganho de acurácia exige mais épocas para maturar |
| **ViT** | +0.11% | -16.0% | Hiperparâmetros refinados já superam o Baseline com mesmo orçamento de épocas |

### Round 4 vs Round 3 (same hyperparams, different epochs)

| Model | Δ Accuracy | CO₂ saved | Interpretation |
|:------|:----------:|:---------:|:---------------|
| **YOLO26** | -1.11% | -80.1% | Os 20 épocas extras do Round 3 foram cruciais para o ganho de +1.11% |
| **CNN** | -0.65% | -87.8% | Regularização precisa de mais épocas para convergir plenamente |
| **ViT** | -0.10% | -73.4% | Hiperparâmetros são o principal driver; épocas extras contribuem marginalmente |

---

## 6. Final Dashboard Visualization

![Final Dashboard](final_dashboard.png)

---

## 7. Hyperparameter Evolution by Model


### YOLO26

| Phase | Hyperparameters |
|:------|:----------------|
| **Baseline** | - **Epochs**: 10 · - **Batch**: 16 · - **Optimizer**: Auto (SGD) · - **LR0**: 0.01 · - **Weight Decay**: 0.0005 · - **Dropout**: 0.0 · - **Label Smoothing**: 0.0 |
| **Round 2** | - **Epochs**: 20 (↑) · - **Batch**: 32 (↑) · - **Optimizer**: AdamW (change) · - **LR0**: 0.001 (↓) · - **Weight Decay**: 0.05 (↑) · - **Dropout**: 0.0 · - **Label Smoothing**: 0.1 (↑) |
| **Round 3** | - **Epochs**: 30 (↑) · - **Batch**: 32 · - **Optimizer**: AdamW · - **LR0**: 0.0012 (↑) · - **LRf**: 0.005 (↓) · - **Weight Decay**: 0.001 (↓) · - **Dropout**: 0.1 (↑) · - **Cosine LR**: True (new) · - **Label Smoothing**: 0.1 |
| **Round 4** | - **Epochs**: 10 (↓) · - **Batch**: 32 · - **Optimizer**: AdamW · - **LR0**: 0.0012 · - **LRf**: 0.005 · - **Weight Decay**: 0.001 · - **Dropout**: 0.1 · - **Cosine LR**: True · - **Label Smoothing**: 0.1 |

**Accuracy progression**: Baseline 90.56% → Round 2 91.62% → Round 3 91.74% → Round 4 90.63%

---

### CNN

| Phase | Hyperparameters |
|:------|:----------------|
| **Baseline** | - **Epochs**: 10 · - **Batch**: 16 · - **LR0**: 0.01 · - **Dropout**: 0.0 · - **Weight Decay**: 0.0005 · - **Label Smoothing**: None · - **Augmentation**: None |
| **Round 2** | - **Epochs**: 20 (↑) · - **Batch**: 32 (↑) · - **LR0**: 0.01 · - **Dropout**: 0.3 (↑) · - **Weight Decay**: 0.001 (↑) · - **Label Smoothing**: 0.1 (new) · - **Augmentation**: Flip+Rot(10°)+Jitter (new) |
| **Round 3** | - **Epochs**: 30 (↑) · - **Batch**: 32 · - **LR0**: 0.015 (↑) · - **LRf**: 0.005 (↓) · - **Dropout**: 0.2 (↓) · - **Weight Decay**: 0.0005 (↓) · - **Warmup**: 5 epochs (↑) · - **Label Smoothing**: 0.1 · - **Augmentation**: Flip+Rot(10°)+Jitter |
| **Round 4** | - **Epochs**: 10 (↓) · - **Batch**: 32 · - **LR0**: 0.015 · - **LRf**: 0.005 · - **Dropout**: 0.2 · - **Weight Decay**: 0.0005 · - **Warmup**: 3 epochs (↓) · - **Label Smoothing**: 0.1 · - **Augmentation**: Flip+Rot(10°)+Jitter |

**Accuracy progression**: Baseline 93.44% → Round 2 93.19% → Round 3 93.96% → Round 4 93.31%

---

### ViT (Transformers)

| Phase | Hyperparameters |
|:------|:----------------|
| **Baseline** | - **Epochs**: 10 · - **Batch**: 16 · - **LR0**: 1e-4 · - **Weight Decay**: 0.05 · - **Label Smoothing**: None · - **Grad Clipping**: None · - **Augmentation**: Flip only |
| **Round 2** | - **Epochs**: 20 (↑) · - **Batch**: 32 (↑) · - **LR0**: 5e-5 (↓) · - **Weight Decay**: 0.1 (↑) · - **Label Smoothing**: 0.1 (new) · - **Grad Clipping**: max_norm=1.0 (new) · - **Augmentation**: +Rot(15°)+Jitter+Shear (↑) |
| **Round 3** | - **Epochs**: 30 (↑) · - **Batch**: 32 · - **LR0**: 8e-5 (↑) · - **LRf**: 0.02 (↑) · - **Weight Decay**: 0.05 (↓) · - **Warmup**: 5 epochs (↑) · - **Label Smoothing**: 0.05 (↓) · - **Grad Clipping**: max_norm=1.0 · - **Augmentation**: Flip+Rot(15°)+Jitter+Shear |
| **Round 4** | - **Epochs**: 10 (↓) · - **Batch**: 32 · - **LR0**: 8e-5 · - **LRf**: 0.02 · - **Weight Decay**: 0.05 · - **Warmup**: 2 epochs (↓) · - **Label Smoothing**: 0.05 · - **Grad Clipping**: max_norm=1.0 · - **Augmentation**: Flip+Rot(15°)+Jitter+Shear |

**Accuracy progression**: Baseline 94.97% → Round 2 95.17% → Round 3 95.18% → Round 4 95.08%

---


## 8. Key Takeaways

- **ViT** é o mais beneficiado pelos hiperparâmetros refinados: com apenas 10 épocas (Round 4), supera o Baseline em +0.11% *e* consome 16% menos energia. O ajuste de LR, label smoothing e augmentation são o principal driver de performance.

- **YOLO26** com hiperparâmetros otimizados (Round 4) mantém praticamente a mesma acurácia do Baseline (+0.07%) usando **60% menos energia** — evidência forte de que o AdamW e batch maior tornam o treinamento mais eficiente energeticamente.

- **CNN** precisa de mais épocas para que a regularização (dropout + augmentation) maturize. No Round 4, com 10 épocas, os hiperparâmetros melhorados quase não agregam acurácia vs Baseline (-0.13%), mas o Round 3 com 30 épocas mostra o ganho real (+0.52%).

- **Melhor trade-off acurácia × sustentabilidade**: **Round 4 YOLO26** (90.63% de acurácia com apenas 0.768g CO₂) é a opção mais verde do experimento inteiro.

- **Melhor acurácia absoluta**: **Round 3 ViT** com 95.18%.

- **GreenAI Insight**: Hiperparâmetros bem ajustados podem reduzir o custo energético do treinamento **sem sacrificar acurácia** (ViT Round 4 vs Baseline), mas para modelos menores como CNN e YOLO26, mais épocas continuam sendo necessárias para extrair o potencial completo da regularização.

---

*Report generated automatically by `generate_final_report.py`.*
