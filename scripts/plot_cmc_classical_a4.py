#!/usr/bin/env python
"""
Combine CMC plots for classical features (LBP, HOG, Dense SIFT) for whole images and pipeline crops.
Reads metrics from results_a4_retry/recognition_metrics_a4.json by default.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with path.open() as f:
        return json.load(f)


def plot_combined(runs, out_path: Path, title: str):
    plt.figure(figsize=(4, 3))
    for cmc, label in runs:
        ranks = list(range(1, len(cmc) + 1))
        plt.plot(ranks, cmc, marker="o", label=label)
    plt.xlabel("Rank")
    plt.ylabel("Match rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    metrics_path = Path("results_a4_retry/recognition_metrics_a4.json")
    out_dir = Path("results_a4_retry")
    metrics = load_metrics(metrics_path)

    methods = [
        ("lbp_ms", "LBP (multiscale)"),
        ("hog", "HOG"),
        ("dense_sift", "Dense SIFT"),
    ]

    whole_runs = []
    pipeline_runs = []
    for key, label in methods:
        whole_runs.append((metrics[f"whole_{key}"]["cmc"], f"Whole - {label}"))
        pipeline_runs.append(
            (metrics[f"pipeline_{key}"]["cmc"], f"Pipeline - {label}"))

    plot_combined(whole_runs, out_dir / "cmc_classical_whole_combined.png",
                  "CMC (Whole images)")
    plot_combined(pipeline_runs, out_dir / "cmc_classical_pipeline_combined.png",
                  "CMC (Pipeline)")
    print(f"Saved combined plots to {out_dir}")


if __name__ == "__main__":
    main()
