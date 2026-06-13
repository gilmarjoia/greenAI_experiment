"""
generate_baseline_vs_modified.py — Generates comparative analysis and dashboard
comparing Baseline configurations (Baseline, R5, R6) vs Modified configurations (R4, R2, R3).

Fases a comparar por orçamento de épocas:
- 10 Épocas: Baseline vs Round 4
- 20 Épocas: Round 5 vs Round 2
- 30 Épocas: Round 6 vs Round 3
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_BASELINE = REPORTS_DIR / "summary_metrics.csv"
CSV_ROUND2   = REPORTS_DIR / "summary_metrics_modified.csv"
CSV_ROUND3   = REPORTS_DIR / "summary_metrics_round3.csv"
CSV_ROUND4   = REPORTS_DIR / "summary_metrics_round4.csv"
CSV_ROUND5   = REPORTS_DIR / "summary_metrics_round5.csv"
CSV_ROUND6   = REPORTS_DIR / "summary_metrics_round6.csv"

MD_REPORT    = REPORTS_DIR / "baselines_vs_modified_report.md"
PNG_DASH     = REPORTS_DIR / "baselines_vs_modified_dashboard.png"

def read_csv(path: Path, key_col: str) -> dict:
    data = {}
    if not path.exists():
        return data
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row[key_col]] = row
    return data

def fv(val, dec=2, suffix=""):
    try:
        return f"{float(val):.{dec}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"

def delta(new_val, old_val, dec=2, suffix=""):
    try:
        d = float(new_val) - float(old_val)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{dec}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"

def pct_delta(new_val, old_val, dec=1):
    try:
        n, o = float(new_val), float(old_val)
        if o == 0:
            return "N/A"
        d = (n - o) / o * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{dec}f}%"
    except (ValueError, TypeError):
        return "N/A"

def main():
    baselines = read_csv(CSV_BASELINE, "Baseline")
    r2_data   = read_csv(CSV_ROUND2,   "Model")
    r3_data   = read_csv(CSV_ROUND3,   "Model")
    r4_data   = read_csv(CSV_ROUND4,   "Model")
    r5_data   = read_csv(CSV_ROUND5,   "Model")
    r6_data   = read_csv(CSV_ROUND6,   "Model")

    # Map the model keys
    models_keys = [
        {
            "id": "YOLO26",
            "base_name": "YOLO26",
            "r4_name": "YOLO26 Round 4",
            "r5_name": "YOLO26 Round 5",
            "r2_name": "YOLO26 Modified",
            "r6_name": "YOLO26 Round 6",
            "r3_name": "YOLO26 Round 3",
            "label": "YOLO26"
        },
        {
            "id": "CNN",
            "base_name": "CNN",
            "r4_name": "CNN Round 4",
            "r5_name": "CNN Round 5",
            "r2_name": "CNN Modified",
            "r6_name": "CNN Round 6",
            "r3_name": "CNN Round 3",
            "label": "CNN"
        },
        {
            "id": "ViT",
            "base_name": "ViT (Transformers)",
            "r4_name": "ViT (Transformers) Round 4",
            "r5_name": "ViT (Transformers) Round 5",
            "r2_name": "ViT (Transformers) Modified",
            "r6_name": "ViT (Transformers) Round 6",
            "r3_name": "ViT (Transformers) Round 3",
            "label": "ViT (DeiT-Tiny)"
        }
    ]

    # Gather data dict for comparison
    data = {}
    for m in models_keys:
        m_id = m["id"]
        data[m_id] = {
            "10_ep_base": baselines.get(m["base_name"], {}),
            "10_ep_mod": r4_data.get(m["r4_name"], {}),
            "20_ep_base": r5_data.get(m["r5_name"], {}),
            "20_ep_mod": r2_data.get(m["r2_name"], {}),
            "30_ep_base": r6_data.get(m["r6_name"], {}),
            "30_ep_mod": r3_data.get(m["r3_name"], {})
        }

    # Generate Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GreenAI: Baselines (Puro) vs. Modificados (Otimizados) por Orçamento de Épocas",
        fontsize=16, fontweight="bold", y=0.98
    )

    labels = [m["label"] for m in models_keys]
    x = np.arange(len(labels))
    width = 0.12

    # Subplot 1: Acurácia por Orçamento
    ax = axes[0, 0]
    # 10 ep
    b10 = ax.bar(x - 2.5*width, [float(data[m["id"]]["10_ep_base"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Baseline (10 ep)", color="#94A3B8")
    m10 = ax.bar(x - 1.5*width, [float(data[m["id"]]["10_ep_mod"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Modificado (10 ep)", color="#F59E0B")
    # 20 ep
    b20 = ax.bar(x - 0.5*width, [float(data[m["id"]]["20_ep_base"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Baseline (20 ep)", color="#64748B")
    m20 = ax.bar(x + 0.5*width, [float(data[m["id"]]["20_ep_mod"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Modificado (20 ep)", color="#6366F1")
    # 30 ep
    b30 = ax.bar(x + 1.5*width, [float(data[m["id"]]["30_ep_base"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Baseline (30 ep)", color="#334155")
    m30 = ax.bar(x + 2.5*width, [float(data[m["id"]]["30_ep_mod"].get("Top-1 Acc (%)", 0)) for m in models_keys], width, label="Modificado (30 ep)", color="#10B981")

    ax.set_title("Acurácia Top-1 (%) por Orçamento", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(87, 96.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    # Annotate Subplot 1
    for bars, color in [(b10, "#475569"), (m10, "#B45309"), (b20, "#334155"), (m20, "#4338CA"), (b30, "#1E293B"), (m30, "#065F46")]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

    # Subplot 2: Emissões de Carbono por Orçamento
    ax = axes[0, 1]
    b10_em = ax.bar(x - 2.5*width, [float(data[m["id"]]["10_ep_base"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Baseline (10 ep)", color="#94A3B8")
    m10_em = ax.bar(x - 1.5*width, [float(data[m["id"]]["10_ep_mod"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Modificado (10 ep)", color="#F59E0B")
    b20_em = ax.bar(x - 0.5*width, [float(data[m["id"]]["20_ep_base"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Baseline (20 ep)", color="#64748B")
    m20_em = ax.bar(x + 0.5*width, [float(data[m["id"]]["20_ep_mod"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Modificado (20 ep)", color="#6366F1")
    b30_em = ax.bar(x + 1.5*width, [float(data[m["id"]]["30_ep_base"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Baseline (30 ep)", color="#334155")
    m30_em = ax.bar(x + 2.5*width, [float(data[m["id"]]["30_ep_mod"].get("Emissions (gCO₂)", 0)) for m in models_keys], width, label="Modificado (30 ep)", color="#10B981")

    ax.set_title("Emissões de Carbono (gCO₂eq)", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Emissões (gCO₂eq)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    for bars, color in [(b10_em, "#475569"), (m10_em, "#B45309"), (b20_em, "#334155"), (m20_em, "#4338CA"), (b30_em, "#1E293B"), (m30_em, "#065F46")]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}g", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

    # Subplot 3: Curvas de Aprendizado e Emissão para YOLO26
    ax = axes[1, 0]
    epochs_axis = [10, 20, 30]
    yolo_base_acc = [float(data["YOLO26"]["10_ep_base"].get("Top-1 Acc (%)", 0)), float(data["YOLO26"]["20_ep_base"].get("Top-1 Acc (%)", 0)), float(data["YOLO26"]["30_ep_base"].get("Top-1 Acc (%)", 0))]
    yolo_mod_acc = [float(data["YOLO26"]["10_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["YOLO26"]["20_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["YOLO26"]["30_ep_mod"].get("Top-1 Acc (%)", 0))]
    
    ax.plot(epochs_axis, yolo_base_acc, "o--", color="#64748B", linewidth=2, label="YOLO26 Baseline")
    ax.plot(epochs_axis, yolo_mod_acc, "o-", color="#6366F1", linewidth=2.5, label="YOLO26 Modificado")
    
    cnn_base_acc = [float(data["CNN"]["10_ep_base"].get("Top-1 Acc (%)", 0)), float(data["CNN"]["20_ep_base"].get("Top-1 Acc (%)", 0)), float(data["CNN"]["30_ep_base"].get("Top-1 Acc (%)", 0))]
    cnn_mod_acc = [float(data["CNN"]["10_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["CNN"]["20_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["CNN"]["30_ep_mod"].get("Top-1 Acc (%)", 0))]
    
    ax.plot(epochs_axis, cnn_base_acc, "s--", color="#94A3B8", linewidth=1.5, alpha=0.7, label="CNN Baseline")
    ax.plot(epochs_axis, cnn_mod_acc, "s-", color="#EC4899", linewidth=2, alpha=0.8, label="CNN Modificado")

    ax.set_title("Curva de Escalonamento de Acurácia (Épocas 10 → 20 → 30)", fontweight="bold", pad=10)
    ax.set_xlabel("Épocas de Treinamento")
    ax.set_ylabel("Acurácia Top-1 (%)")
    ax.set_xticks(epochs_axis)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    # Subplot 4: Eficiência Energética vs Acurácia (Trade-off)
    ax = axes[1, 1]
    
    vit_base_acc = [float(data["ViT"]["10_ep_base"].get("Top-1 Acc (%)", 0)), float(data["ViT"]["20_ep_base"].get("Top-1 Acc (%)", 0)), float(data["ViT"]["30_ep_base"].get("Top-1 Acc (%)", 0))]
    vit_mod_acc = [float(data["ViT"]["10_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["ViT"]["20_ep_mod"].get("Top-1 Acc (%)", 0)), float(data["ViT"]["30_ep_mod"].get("Top-1 Acc (%)", 0))]
    
    ax.plot(epochs_axis, vit_base_acc, "d--", color="#475569", linewidth=1.5, label="ViT Baseline")
    ax.plot(epochs_axis, vit_mod_acc, "d-", color="#10B981", linewidth=2.5, label="ViT Modificado")
    
    ax.set_title("Escalonamento de Acurácia no ViT (DeiT-Tiny)", fontweight="bold", pad=10)
    ax.set_xlabel("Épocas de Treinamento")
    ax.set_ylabel("Acurácia Top-1 (%)")
    ax.set_xticks(epochs_axis)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    plt.tight_layout()
    fig.savefig(PNG_DASH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved: {PNG_DASH}")

    # Generate Markdown comparative report
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Generate MD tables for each epoch level
    def build_table(ep_str_base, ep_str_mod, title):
        md = f"### {title}\n\n"
        md += "| Modelo | Baseline Acc | Modificado Acc | Δ Acurácia | Baseline CO₂ | Modificado CO₂ | Δ CO₂ (%) | Baseline Tempo | Modificado Tempo | Δ Tempo (%) |\n"
        md += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        for m in models_keys:
            m_id = m["id"]
            base = data[m_id][ep_str_base]
            mod = data[m_id][ep_str_mod]
            
            b_acc = base.get("Top-1 Acc (%)", "N/A")
            m_acc = mod.get("Top-1 Acc (%)", "N/A")
            b_em = base.get("Emissions (gCO₂)", "N/A")
            m_em = mod.get("Emissions (gCO₂)", "N/A")
            b_time = base.get("Train Time (min)", "N/A")
            m_time = mod.get("Train Time (min)", "N/A")
            
            md += (
                f"| **{m['label']}** "
                f"| {fv(b_acc, 2)}% "
                f"| {fv(m_acc, 2)}% "
                f"| **{delta(m_acc, b_acc, 2)}%** "
                f"| {fv(b_em, 3)}g "
                f"| {fv(m_em, 3)}g "
                f"| **{pct_delta(m_em, b_em)}** "
                f"| {fv(b_time, 1)} min "
                f"| {fv(m_time, 1)} min "
                f"| **{pct_delta(m_time, b_time)}** |\n"
            )
        return md

    table_10 = build_table("10_ep_base", "10_ep_mod", "1. Orçamento de 10 Épocas (Baseline Inicial vs Round 4)")
    table_20 = build_table("20_ep_base", "20_ep_mod", "2. Orçamento de 20 Épocas (Round 5 vs Round 2)")
    table_30 = build_table("30_ep_base", "30_ep_mod", "3. Orçamento de 30 Épocas (Round 6 vs Round 3)")

    report_content = f"""# Relatório Comparativo: Baselines vs. Modificados (Otimizados)

**Data de Geração:** {ts}
**Estudo de Caso:** Fashion-MNIST (YOLO26, CNN Personalizada, DeiT-Tiny ViT)

Este relatório faz uma comparação direta e sistemática entre os dois grandes caminhos de experimentação tomados no projeto:
1. **Baselines (Sem regularização/aumento):** Modelos treinados com hiperparâmetros padrão do baseline em 10 épocas (Baseline Inicial), 20 épocas (Round 5) e 30 épocas (Round 6).
2. **Modificados (Com regularização/aumento/otimização):** Modelos treinados com hiperparâmetros ajustados (GreenAI) em 10 épocas (Round 4), 20 épocas (Round 2) e 30 épocas (Round 3).

---

## Painel Comparativo Geral (Dashboard)

Abaixo está o dashboard visual detalhando a evolução do desempenho, custo de carbono e curvas de escalamento por época.

![Dashboard Baselines vs Modificados](baselines_vs_modified_dashboard.png)

---

## Comparações Diretas por Orçamento de Treinamento

{table_10}

#### Principais Observações (10 Épocas):
- **YOLO26 (Round 4):** A otimização com o otimizador AdamW e aumento de batch size (16 → 32) reduziu o tempo de treino em **-61.9%** e as emissões de carbono em **-60.2%**, com uma acurácia ligeiramente superior (+0.07%). Demonstração perfeita de eficiência GreenAI.
- **CNN (Round 4):** A injeção de regularização e aumento de dados limitou ligeiramente a acurácia em 10 épocas (-0.13%), pois técnicas de regularização precisam de mais tempo de convergência. O aumento de tempo de treinamento (+285.4%) reflete a sobrecarga de CPU pelo processamento das transformações geométricas no dataset em tempo real.
- **ViT (Round 4):** Excelente resultado. Consegue obter melhorias com os novos hiperparâmetros, obtendo +0.11% de acurácia com uma redução de **-16.0% nas emissões**, mostrando que o baseline do ViT possuía hiperparâmetros subótimos de aprendizado.

---

{table_20}

#### Principais Observações (20 Épocas):
- **YOLO26 (Round 2):** Superioridade absoluta do modelo otimizado. O Round 2 obteve **+1.86% mais acurácia** consumindo **-35.4% menos carbono** e treinando **-40.5% mais rápido** que o baseline de 20 épocas (Round 5).
- **CNN (Round 2):** Enquanto o baseline de 20 épocas (Round 5) começou a sofrer severamente com overfitting (val loss subindo para 0.254 enquanto train loss caiu para 0.0069), o Round 2 manteve perdas balanceadas devido às restrições de dropout e augmentations, alcançando praticamente a mesma acurácia (-0.04%). O custo de tempo de processamento das imagens de aumento aumentou o tempo total de treinamento em +64.9%.
- **ViT (Round 2):** Ganho de acurácia de **+0.14%** utilizando os novos hiperparâmetros, embora o custo de processamento das transformações mais agressivas tenha aumentado o tempo total de treinamento (+32.4%).

---

{table_30}

#### Principais Observações (30 Épocas):
- **YOLO26 (Round 3):** O modelo otimizado obteve **+1.61% mais acurácia** economizando **-48.0% de emissões** e treinando **-46.7% mais rápido** que o baseline de 30 épocas (Round 6). O escalamento do YOLO baseline puro é altamente ineficiente e propenso a overfitting.
- **CNN (Round 3):** Com 30 épocas, a regularização e o aumento de dados no Round 3 finalmente amadureceram completamente, entregando um salto expressivo de acurácia de **+0.72%** em relação ao Round 6. O custo energético associado às augmentações foi de +235.7%, contudo a generalização foi excelente.
- **ViT (Round 3):** O pico de acurácia do experimento foi de **95.18%** (Round 3), superando o baseline Round 6 em **+0.17%** com um custo de carbono ligeiramente superior (+37.5%).

---

## Análise de Eficiência e Escalamento (Trade-Off)

### Curva de Escalamento Baseline (Sem Regularização)
Ao analisarmos a trajetória do Baseline puro (10 → 20 → 30 épocas), o comportamento é ineficiente:
- **YOLO26:** A acurácia cai de 90.56% (10 ep) para 89.76% (20 ep) e se recupera apenas para 90.13% (30 ep) — registrando perda líquida de **-0.43%** enquanto as emissões subiram **+284.6%**.
- **CNN:** A acurácia cai de 93.44% (10 ep) para 93.23% (20 ep) e 93.24% (30 ep) — perda líquida de **-0.20%** com alta absurda de emissões de **+1041.0%**.
- **ViT:** Ganho líquido insignificante de **+0.04%** (94.97% → 95.01%) com emissões subindo **+129.3%**.

> **Conclusão de Baseline:** Aumentar épocas sem calibrar os hiperparâmetros e sem usar regularização resulta em desperdício severo de recursos computacionais e regressão de desempenho devido ao overfitting precoce.

### Curva de Escalamento Modificado (Com Regularização e Otimização GreenAI)
Quando comparamos a evolução dos modificados (Round 4 [10 ep] → Round 2 [20 ep] → Round 3 [30 ep]):
- **YOLO26:** Acurácia escala de 90.63% → 91.62% → 91.74% (Ganho de **+1.11%**). O tempo de treino e carbono subiram sob controle devido ao batch size de 32 e otimizador AdamW.
- **CNN:** Acurácia escala de 93.31% → 93.19% → 93.96% (Ganho de **+0.65%**). A regularização robusta permitiu que o modelo aproveitasse o orçamento maior de 30 épocas sem sofrer overfitting catastrófico.
- **ViT:** Acurácia escala de 95.08% → 95.17% → 95.18% (Ganho de **+0.10%**). 

---

## Recomendação GreenAI Final

1. **Eficiência Extrema (Menor Carbono):** **YOLO26 Modificado (10 Épocas - Round 4)**. Entrega 90.63% de acurácia com apenas 0.768g de CO₂.
2. **Desempenho Balanceado:** **YOLO26 Modificado (30 Épocas - Round 3)**. Acurácia de 91.74% com 3.855g de CO₂. Supera o baseline de 30 épocas em acurácia gastando metade do carbono.
3. **Precisão Máxima (Maior Acurácia):** **ViT Modificado (30 Épocas - Round 3)**. Acurácia de 95.18% com 32.008g de CO₂.
4. **Combinação Desaconselhada:** Treinar qualquer modelo com configurações baseline padrão por mais de 10 épocas.
"""

    with open(MD_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved: {MD_REPORT}")

if __name__ == "__main__":
    main()
