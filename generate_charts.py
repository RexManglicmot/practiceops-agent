"""Renders assets/eval_metrics.png from the model-comparison numbers in the
README's Evaluation table. Dev-only utility (needs matplotlib, not a runtime
dependency of app.py/agents.py/evaluate.py) - rerun after any evaluate.py
number changes to keep the chart in sync.
"""
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent

# Fixed categorical order (never cycled) - blue then orange, per the
# project's data-viz palette.
MODEL_COLORS = {
    "gemma2:9b (default)": "#2a78d6",
    "llama3.1:latest": "#eb6834",
}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

RESULTS = {
    "Diagnostic Agent": {
        "gemma2:9b (default)": {"Precision": 0.90, "Recall": 0.86, "F1": 0.88},
        "llama3.1:latest": {"Precision": 0.88, "Recall": 0.67, "F1": 0.76},
    },
    "Action Agent": {
        "gemma2:9b (default)": {"Precision": 1.00, "Recall": 1.00, "F1": 1.00},
        "llama3.1:latest": {"Precision": 0.99, "Recall": 1.00, "F1": 0.99},
    },
}
METRICS = ["Precision", "Recall", "F1"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor=SURFACE, sharey=True)

bar_width = 0.32
group_gap = 0.02

for ax, (agent, per_model) in zip(axes, RESULTS.items()):
    ax.set_facecolor(SURFACE)
    x = range(len(METRICS))

    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        offset = (i - 0.5) * (bar_width + group_gap)
        values = [per_model[model][m] for m in METRICS]
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            color=color,
            label=model,
            zorder=3,
        )
        for rect, val in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=SECONDARY_INK,
            )

    ax.set_title(agent, fontsize=12, color=INK, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(METRICS, fontsize=10, color=MUTED)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=9, color=MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(bottom=False, left=False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=2,
    frameon=False,
    fontsize=10,
    labelcolor=SECONDARY_INK,
)
fig.suptitle(
    "Diagnostic & Action Agent scores by model (test_cases.json, n=30)",
    fontsize=10,
    color=MUTED,
    y=1.14,
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = PROJECT_DIR / "assets" / "eval_metrics.png"
fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(f"Wrote {out_path}")
