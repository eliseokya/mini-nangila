import sys
import csv
import matplotlib.pyplot as plt


def load_metrics(path):
    steps = []
    ratios = []
    with open(path, "r") as f:
        for row in f:
            if row.startswith("step,"):
                parts = row.strip().split(",")
                try:
                    # step,<n>,bucket_elems,<...>,raw_bytes,<...>,comp_bytes,<...>,ratio,<val>
                    step_idx = int(parts[1])
                    ratio = float(parts[-1])
                    steps.append(step_idx)
                    ratios.append(ratio)
                except Exception:
                    continue
    return steps, ratios


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ddp_metrics.csv"
    steps, ratios = load_metrics(path)
    if not steps:
        print("No metrics found in", path)
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, ratios, "+-", alpha=0.7)
    plt.xlabel("hook call index")
    plt.ylabel("compression ratio (raw/comp)")
    plt.title("DDP INT4 Estimated Compression per Bucket")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = path.replace('.csv', '_ratio.png')
    plt.savefig(out)
    print("Saved plot:", out)


if __name__ == "__main__":
    main()

