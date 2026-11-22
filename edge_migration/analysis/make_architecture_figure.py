"""Generate publication-ready System Architecture diagram."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_architecture_diagram(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color palette
    c_tier_bg = "#f8f9fa"
    c_edge_blue = "#1e3d59"
    c_box_blue = "#e8f1f5"
    c_box_border_blue = "#2b580c"
    c_box_green = "#e8f5e9"
    c_box_orange = "#fff3e0"
    c_box_purple = "#f3e5f5"
    c_accent_orange = "#ff6e40"
    c_text = "#17252a"

    # Background container
    bg_rect = patches.FancyBboxPatch((1, 1), 98, 98, boxstyle="round,pad=0.5",
                                    facecolor="#ffffff", edgecolor="#cccccc", linewidth=1.5)
    ax.add_patch(bg_rect)

    # Title Banner
    ax.text(50, 96, "Proactive Edge Node Migration Architecture (FCM)",
            fontsize=13, weight="bold", ha="center", va="center", color=c_edge_blue)

    # Box 1: Edge Computing Tier
    box1 = patches.FancyBboxPatch((3, 62), 27, 30, boxstyle="round,pad=0.4",
                                 facecolor=c_box_blue, edgecolor="#3a7bd5", linewidth=1.5)
    ax.add_patch(box1)
    ax.text(16.5, 89, "1. Edge Infrastructure", fontsize=10.5, weight="bold", ha="center", color="#0056b3")
    ax.text(16.5, 83, "Heterogeneous Edge Nodes\n$N_1, N_2, \\dots, N_K$", fontsize=9, ha="center", color=c_text)
    ax.text(16.5, 75, "Hosted Microservices\n$S_1, S_2, \\dots, S_M$\nState: $m_s \\in [32, 256]$ MB", fontsize=8.5, ha="center", color="#333333")
    ax.text(16.5, 66, "Node Capacity & Network Links\nBandwidth $B_{ij}$", fontsize=8.5, ha="center", style="italic", color="#555555")

    # Box 2: Telemetry & Traces
    box2 = patches.FancyBboxPatch((36, 62), 27, 30, boxstyle="round,pad=0.4",
                                 facecolor=c_box_green, edgecolor="#2e7d32", linewidth=1.5)
    ax.add_patch(box2)
    ax.text(49.5, 89, "2. Workload Telemetry", fontsize=10.5, weight="bold", ha="center", color="#2e7d32")
    ax.text(49.5, 82, "4-Channel Metrics ($c \\in [0, 1]$):\nCPU, Mem, Request, Net", fontsize=9, ha="center", color=c_text)
    ax.text(49.5, 74, "Workload Sources:\n- Google Cluster Trace\n- Synthetic Correlated Traces", fontsize=8.5, ha="center", color="#333333")
    ax.text(49.5, 66, "Sliding Window Buffer\n$W = 30$ slots (30 min)", fontsize=8.5, ha="center", weight="bold", color="#1b5e20")

    # Box 3: Multi-Model Forecasting Engine
    box3 = patches.FancyBboxPatch((69, 62), 28, 30, boxstyle="round,pad=0.4",
                                 facecolor=c_box_purple, edgecolor="#7b1fa2", linewidth=1.5)
    ax.add_patch(box3)
    ax.text(83, 89, "3. Forecasting Engine", fontsize=10.5, weight="bold", ha="center", color="#7b1fa2")
    ax.text(83, 81.5, "Evaluated Forecasters:\n- Moving Average (Baseline)\n- ARIMA AR(1)\n- XGBoost (Multi-Output)\n- Recurrent (GRU / LSTM)", fontsize=8.5, ha="center", color=c_text)
    ax.text(83, 67, "Predicted Node Load:\n$\\hat{L}_n(t+k) = \\sum_c w_c \\hat{u}_{n,c}(t+k)$\nLead Horizon $H = 10$ slots", fontsize=8.5, ha="center", weight="bold", color="#4a148c")

    # Arrows between top boxes
    ax.annotate("", xy=(36, 77), xytext=(30, 77),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#0056b3"))
    ax.annotate("", xy=(69, 77), xytext=(63, 77),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#2e7d32"))

    # Large Center/Bottom Box: FCM Controller
    box4 = patches.FancyBboxPatch((12, 10), 76, 45, boxstyle="round,pad=0.5",
                                 facecolor=c_box_orange, edgecolor=c_accent_orange, linewidth=2)
    ax.add_patch(box4)
    ax.text(50, 52, "4. Proactive Cost-Aware Migration Controller (FCM)",
            fontsize=11.5, weight="bold", ha="center", color="#d84315")

    # Sub-steps inside FCM
    step_w = 17
    # Step A: Trigger
    s1 = patches.FancyBboxPatch((14, 18), step_w, 30, boxstyle="round,pad=0.3",
                                facecolor="#ffffff", edgecolor="#ffab91", linewidth=1.2)
    ax.add_patch(s1)
    ax.text(22.5, 44, "A. Overload Trigger", fontsize=9, weight="bold", ha="center", color="#d84315")
    ax.text(22.5, 34, "Condition:\n$\\hat{L}_i(t+k) \\geq \\theta_{\\mathrm{hi}}$\nfor $\\geq \\pi$ slots\n($\\theta_{\\mathrm{hi}}=0.85, \\pi=3$)\nSustained Overload", fontsize=8, ha="center")

    # Step B: Candidate Filtering
    s2 = patches.FancyBboxPatch((32.5, 18), step_w, 30, boxstyle="round,pad=0.3",
                                facecolor="#ffffff", edgecolor="#ffab91", linewidth=1.2)
    ax.add_patch(s2)
    ax.text(41, 44, "B. Feasibility Filter", fontsize=9, weight="bold", ha="center", color="#d84315")
    ax.text(41, 33, "Lead Time:\n$T_{\\mathrm{mig}} \\leq$ First Breach\n\nSafety Check:\n$\\max \\hat{L}_j < \\theta_{\\mathrm{safe}}$\n($\\theta_{\\mathrm{safe}}=0.70$)\n+ Memory Check", fontsize=8, ha="center")

    # Step C: Multi-Objective Scoring
    s3 = patches.FancyBboxPatch((51, 18), step_w, 30, boxstyle="round,pad=0.3",
                                facecolor="#ffffff", edgecolor="#ffab91", linewidth=1.2)
    ax.add_patch(s3)
    ax.text(59.5, 44, "C. Scoring Function", fontsize=9, weight="bold", ha="center", color="#d84315")
    ax.text(59.5, 33, "Score for node $j$:\n$\\alpha \\tilde{u}_j + \\beta \\tilde{\\ell}_j + \\gamma \\tilde{m}_j$\n\nWeights:\n$(\\alpha, \\beta, \\gamma) = (0.5, 0.3, 0.2)$\nLeast Score Wins", fontsize=8, ha="center")

    # Step D: Gain Test & Actuation
    s4 = patches.FancyBboxPatch((69.5, 18), step_w, 30, boxstyle="round,pad=0.3",
                                facecolor="#ffffff", edgecolor="#ffab91", linewidth=1.2)
    ax.add_patch(s4)
    ax.text(78, 44, "D. Gain & Actuation", fontsize=9, weight="bold", ha="center", color="#d84315")
    ax.text(78, 33, "Gain Test:\n$\\Delta \\mathrm{Peak} + 0.05\\tilde{g}_\\ell \\geq \\tau$\n($\\tau = 0.03$)\n\nLive Migration:\nTransfer $m_s$, reserve mem\n10-slot cooldown", fontsize=8, ha="center")

    # Arrows connecting steps A -> B -> C -> D
    for x_arr in (31.2, 49.7, 68.2):
        ax.annotate("", xy=(x_arr + 1.2, 33), xytext=(x_arr - 0.2, 33),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#ff5722"))

    # Large downward arrow from Forecasting Engine (Box 3) to FCM (Box 4)
    ax.annotate("", xy=(65, 55.5), xytext=(78, 62),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#7b1fa2", connectionstyle="arc3,rad=-0.15"))
    ax.text(74, 57, "$\\hat{L}_n(t+k)$", fontsize=9, weight="bold", color="#7b1fa2")

    # Feedback arrow from Actuation (Step D) back to Edge Infrastructure (Box 1)
    ax.annotate("", xy=(16.5, 61.5), xytext=(16.5, 10),
                arrowprops=dict(arrowstyle="-", lw=1.8, color="#0056b3", linestyle="--"))
    ax.annotate("", xy=(16.5, 61.5), xytext=(78, 10),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#0056b3", linestyle="--",
                                connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.text(45, 6, "Live State Migration Actuation & Updated Service Placements",
            fontsize=8.5, weight="bold", ha="center", color="#0056b3")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = Path(__file__).parents[1] / "results" / "architecture.png"
    draw_architecture_diagram(out)
    print(f"Saved architecture diagram to {out}")
