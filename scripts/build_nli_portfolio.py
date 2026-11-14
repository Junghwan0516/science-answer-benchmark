"""Rebuild the UQ figure from the published aggregate report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def build(report, output):
    metrics = report["metrics"]
    models = ["zero_shot", "fine_tuned"]
    colors = ["#537399", "#147d73"]
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        fig.patch.set_facecolor("white")
        for ax, key, title in zip(axes, ["macro_f1", "correct_recall",
                "contradiction_to_entailment_rate"], ["Macro F1", "Correct-answer recall",
                "Contradiction → correct"], strict=True):
            values = [metrics[m][key] for m in models]
            bars = ax.bar([0, 1], values, width=.52, color=colors if key !=
                          "contradiction_to_entailment_rate" else [colors[0], "#b34c35"])
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1], ["Zero-shot", "Fine-tuned"])
            ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
            ax.bar_label(bars, labels=[f"{v:.3f}" if key=="macro_f1" else f"{v:.1%}" for v in values], padding=4)
            if key != "macro_f1":
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("Fine-tuning finds more correct answers but also labels more contradictions as correct",
                     fontsize=13, fontweight="bold")
        fig.text(.5, .047, "SciEntsBank UQ · 733 answers across 15 questions · Raw predictions before human review",
                 ha="center", fontsize=9, color="#34465c")
        fig.subplots_adjust(left=.065, right=.98, top=.75, bottom=.19, wspace=.43)
        fig.savefig(output / "uq_metrics.svg", metadata={"Date": None})
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55), constrained_layout=True)
        labels = ["Contradiction", "Entailment", "Neutral"]
        for ax, model in zip(axes, models, strict=True):
            matrix = np.array(metrics[model]["confusion_matrix"])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=300)
            for (i, j), n in np.ndenumerate(matrix):
                ax.text(j, i, str(n), ha="center", va="center", color="white" if n>155 else "#152e4c")
            ax.set_xticks(range(3), labels, rotation=25, ha="right", fontsize=8)
            ax.set_yticks(range(3), labels, fontsize=8)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Gold label")
            ax.set_title(model.replace("_", "-").capitalize())
        fig.savefig(output / "uq_confusion_matrices.svg", metadata={"Date": None})
        plt.close(fig)

        with (output / "calibration_thresholds.csv").open() as source:
            candidates = list(csv.DictReader(source))
        accepted = np.array([int(row["accepted"]) for row in candidates])
        precision = np.array([int(row["correct"]) for row in candidates]) / accepted
        contradiction_rate = np.array([int(row["contradictory"]) for row in candidates]) / 42
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
        for ax, values, limit, title in zip(axes, [precision, contradiction_rate],
                [.9, .05], ["Correct among accepted", "Accepted among 42 contradictions"], strict=True):
            ax.scatter(accepted, values, color=colors[0], s=12)
            ax.axhline(limit, color="#b34c35", ls="--", label=f"Target: {limit:.0%}")
            ax.axvline(30, color="#147d73", ls=":", label="Minimum: 30 accepted")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("Answers accepted at candidate threshold")
            ax.set_title(title, fontsize=10)
            ax.legend(loc="best", fontsize=8)
        fig.savefig(output / "calibration_tradeoff.svg", metadata={"Date": None})
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/nli")
    args = parser.parse_args()
    build(json.loads((args.report_dir / "uq_report.json").read_text()), args.report_dir)
