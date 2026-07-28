"""Renders assets/confusion_matrix.png: a per-label 2x2 confusion matrix for
each issue label (Diagnostic Agent) and each action label (Action Agent).
Multi-label classification doesn't reduce to one NxN matrix - each label is
its own independent binary decision, so it gets its own TP/FP/FN/TN grid
(the standard sklearn multilabel_confusion_matrix approach). Dev-only utility
(needs matplotlib); rerun after any change to test_cases.json or the agents.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from agents import ACTION_LABELS, ISSUE_LABELS, OLLAMA_MODEL, action_agent, diagnostic_agent

PROJECT_DIR = Path(__file__).parent

with open(PROJECT_DIR / "test_cases.json") as f:
    test_cases = json.load(f)


def confusion_counts(labels: list, cases: list, expected_key: str, predicted_key: str) -> dict:
    """Per-label TP/FP/FN/TN, one binary present/absent decision per label per case."""
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in labels}
    for expected, predicted in cases:
        expected, predicted = set(expected), set(predicted)
        for label in labels:
            actual_positive = label in expected
            predicted_positive = label in predicted
            if actual_positive and predicted_positive:
                counts[label]["tp"] += 1
            elif not actual_positive and predicted_positive:
                counts[label]["fp"] += 1
            elif actual_positive and not predicted_positive:
                counts[label]["fn"] += 1
            else:
                counts[label]["tn"] += 1
    return counts


issue_cases, action_cases = [], []
for case in test_cases:
    result1 = diagnostic_agent(case["metrics"])
    issue_cases.append((case["expected_issues"], result1["issues"]))

    result2 = action_agent(case["expected_issues"])
    action_cases.append((case["expected_actions"], result2["actions"]))

issue_counts = confusion_counts(ISSUE_LABELS, issue_cases, "expected_issues", "issues")
action_counts = confusion_counts(ACTION_LABELS, action_cases, "expected_actions", "actions")

# Sequential blue ramp (light -> dark), per the project's data-viz palette.
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def ramp_color(value: int, vmax: int) -> str:
    if vmax == 0:
        return RAMP[0]
    step = min(int((value / vmax) * (len(RAMP) - 1)), len(RAMP) - 1)
    return RAMP[step]


def render(counts: dict, labels: list, title: str, out_path: Path, cols: int) -> None:
    rows = -(-len(labels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), facecolor=SURFACE)
    axes = axes.flatten() if len(labels) > 1 else [axes]

    for ax, label in zip(axes, labels):
        c = counts[label]
        vmax = max(c.values())
        # grid: rows = actual (positive, negative); cols = predicted (positive, negative)
        grid = [[c["tp"], c["fn"]], [c["fp"], c["tn"]]]
        cell_labels = [["TP", "FN"], ["FP", "TN"]]

        ax.set_facecolor(SURFACE)
        for r in range(2):
            for col in range(2):
                val = grid[r][col]
                color = ramp_color(val, vmax)
                ax.add_patch(
                    plt.Rectangle((col, 1 - r), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2)
                )
                text_color = INK if color in RAMP[:2] else "#ffffff"
                ax.text(
                    col + 0.5, 1 - r + 0.58, str(val),
                    ha="center", va="center", fontsize=13, fontweight="bold", color=text_color,
                )
                ax.text(
                    col + 0.5, 1 - r + 0.28, cell_labels[r][col],
                    ha="center", va="center", fontsize=8, color=text_color, alpha=0.85,
                )

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Predicted\npositive", "Predicted\nnegative"], fontsize=8, color=MUTED)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Actual\nnegative", "Actual\npositive"], fontsize=8, color=MUTED)
        ax.set_title(label, fontsize=10, color=INK, fontweight="bold", pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_path}")


assets_dir = PROJECT_DIR / "assets"
render(
    issue_counts, ISSUE_LABELS,
    f"Diagnostic Agent — per-label confusion matrix ({OLLAMA_MODEL}, n=30)",
    assets_dir / "confusion_matrix_diagnostic.png", cols=3,
)
render(
    action_counts, ACTION_LABELS,
    f"Action Agent — per-label confusion matrix ({OLLAMA_MODEL}, n=30)",
    assets_dir / "confusion_matrix_action.png", cols=3,
)
