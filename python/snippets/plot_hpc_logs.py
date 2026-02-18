import sys
import re
import matplotlib.pyplot as plt


def parse_log(path):
    steps = []
    ratios = []
    max_errs = []
    with open(path, "r") as f:
        for line in f:
            m = re.search(r"step\s+(\d+)\s+ratio\s+([0-9.]+).+max_err\s+([0-9.]+)", line, re.IGNORECASE)
            if m:
                steps.append(int(m.group(1)))
                ratios.append(float(m.group(2)))
                max_errs.append(float(m.group(3)))
    return steps, ratios, max_errs


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_hpc_logs.py <logfile>")
        return
    path = sys.argv[1]
    steps, ratios, max_errs = parse_log(path)
    if not steps:
        print("No step lines found in", path)
        return
    # Ratio vs step
    plt.figure(figsize=(8,4))
    plt.plot(steps, ratios, "-o", alpha=0.7)
    plt.xlabel("step")
    plt.ylabel("compression ratio (raw/comp)")
    plt.title("HPC Compression Ratio vs Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hpc_ratio.png")
    # Epsilon histogram
    plt.figure(figsize=(6,4))
    plt.hist(max_errs, bins=20, alpha=0.8)
    plt.xlabel("max error per step")
    plt.ylabel("count")
    plt.title("HPC Max Error Distribution")
    plt.tight_layout()
    plt.savefig("hpc_eps_hist.png")
    print("Saved: hpc_ratio.png, hpc_eps_hist.png")


if __name__ == "__main__":
    main()

