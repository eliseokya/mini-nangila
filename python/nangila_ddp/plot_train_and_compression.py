import sys
import csv
import time
import math
import matplotlib.pyplot as plt


def load_train(path):
    ts, step, loss = [], [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == 'ts':
                continue
            try:
                ts.append(float(row[0]))
                step.append(int(row[1]))
                loss.append(float(row[2]))
            except Exception:
                pass
    return ts, step, loss


def load_ddp(path):
    ts, ratio, res = [], [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("ts,"):
                parts = line.strip().split(",")
                try:
                    # Parse as key,value pairs
                    kv = {parts[i]: parts[i+1] for i in range(0, len(parts)-1, 2)}
                    t = float(kv.get("ts", "0"))
                    r = float(kv.get("ratio", "0"))
                    m = float(kv.get("res_mean_abs", "nan"))
                    ts.append(t)
                    ratio.append(r)
                    res.append(m)
                except Exception:
                    continue
    return ts, ratio, res


def main():
    train_csv = sys.argv[1] if len(sys.argv) > 1 else "train_log.csv"
    ddp_csv = sys.argv[2] if len(sys.argv) > 2 else "ddp_metrics.csv"
    t_ts, t_step, t_loss = load_train(train_csv)
    d_ts, d_ratio, d_res = load_ddp(ddp_csv)
    if not t_ts or not d_ts:
        print("No data found; make sure CSVs exist.")
        return

    t0 = min(min(t_ts), min(d_ts))
    t_ts = [x - t0 for x in t_ts]
    d_ts = [x - t0 for x in d_ts]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t_ts, t_loss, label="loss", color="tab:blue")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(d_ts, d_ratio, ".", alpha=0.3, color="tab:orange", label="ratio (raw/comp)")
    # smooth ratio
    try:
        import numpy as np
        if len(d_ratio) > 10:
            k = max(1, len(d_ratio)//100)
            kernel = np.ones(k) / k
            smooth = np.convolve(np.array(d_ratio), kernel, mode='same')
            ax2.plot(d_ts, smooth, color="tab:orange", alpha=0.8)
    except Exception:
        pass
    ax2.set_ylabel("compression ratio", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Optionally plot residual mean abs (scaled for visibility)
    try:
        import numpy as np
        if d_res and not all([math.isnan(x) for x in d_res]):
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.1))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            ax3.plot(d_ts, d_res, ":", alpha=0.5, color="tab:green", label="mean|residual|")
            ax3.set_ylabel("mean|residual|", color="tab:green")
            ax3.tick_params(axis='y', labelcolor='tab:green')
    except Exception:
        pass

    fig.tight_layout()
    out = "train_and_compression.png"
    plt.savefig(out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
