#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") == "OK" and r.get("compressed_bytes") and r["compressed_bytes"] != "0":
                rows.append(r)
    return rows


def to_float(s, default=0.0):
    try:
        return float(s)
    except Exception:
        return default


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/baseline_comparison.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmarks/results/baseline_comparison.png")
    rows = load_rows(csv_path)
    if not rows:
        print("No OK rows in", csv_path)
        return

    # Group by epsilon
    groups = {}
    for r in rows:
        eps = r.get("epsilon") or ""
        groups.setdefault(eps, []).append(r)

    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4 * max(1, len(groups))), squeeze=False)
    for ax, (eps, rs) in zip(axes[:, 0], groups.items()):
        tools = [r["tool"] for r in rs]
        ratios = [to_float(r["ratio"]) for r in rs]
        thr = [to_float(r["throughput_MBps"]) for r in rs]

        ax2 = ax.twinx()
        bars = ax.bar(range(len(tools)), ratios, color="tab:blue", alpha=0.7, label="ratio")
        ax2.plot(range(len(tools)), thr, color="tab:orange", marker="o", label="throughput")
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=20, ha="right")
        ax.set_ylabel("ratio (raw/comp)")
        ax2.set_ylabel("throughput (MB/s)")
        title = f"Baseline Comparison (epsilon={eps})" if eps else "Baseline Comparison"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        # Merge legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

