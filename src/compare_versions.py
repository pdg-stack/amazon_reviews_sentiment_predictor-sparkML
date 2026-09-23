"""
Compares several registered model versions' held-out test-set metrics side
by side, reading the metrics.json files evaluate.py --version N writes to
docs/reports/version_<N>/. Does not run any Spark job itself -- run
`python src/evaluate.py --version N` for each version first.

Produces:
  - docs/reports/model_comparison.png : grouped bar chart (accuracy/F1/AUC per version)
  - a markdown table printed to stdout, ready to paste into MODEL_HISTORY.md

Example:
    python src/evaluate.py --version 1
    python src/evaluate.py --version 2
    python src/evaluate.py --version 3
    python src/compare_versions.py --versions 1 2 3
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import REPORTS_DIR


def _load_metrics(version: int) -> dict:
    path = REPORTS_DIR / f"version_{version}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python src/evaluate.py --version {version}` first."
        )
    return json.loads(path.read_text())


def main(versions: list[int]) -> None:
    all_metrics = [_load_metrics(v) for v in versions]

    labels = [f"v{m['version']}" for m in all_metrics]
    accuracy = [m["accuracy"] for m in all_metrics]
    f1 = [m["f1"] for m in all_metrics]
    auc = [m["auc"] for m in all_metrics]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width for i in x], accuracy, width, label="Accuracy", color="#4C72B0")
    ax.bar(list(x), f1, width, label="F1", color="#55A868")
    ax.bar([i + width for i in x], auc, width, label="AUC", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score (held-out test set)")
    ax.set_title("Model version comparison")
    ax.legend()
    for i, (a, f, u) in enumerate(zip(accuracy, f1, auc)):
        ax.text(i - width, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(i + width, u + 0.005, f"{u:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    out_path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Chart written to {out_path}\n")

    print("| Version | Accuracy | F1 | AUC |")
    print("|---|---|---|---|")
    for m in all_metrics:
        print(f"| v{m['version']} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions", type=int, nargs="+", required=True, help="Registered versions to compare, e.g. --versions 1 2 3"
    )
    args = parser.parse_args()
    main(versions=args.versions)
