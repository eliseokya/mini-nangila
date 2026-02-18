import sys
import csv
import matplotlib.pyplot as plt


def load_csv(path):
    ticks, ratios = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                tick = int(row[0])
                ratio = float(row[-1])
                ticks.append(tick)
                ratios.append(ratio)
            except Exception:
                continue
    return ticks, ratios


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_rotor_net.py client_metrics.csv [server_metrics.csv]")
        return
    client_csv = sys.argv[1]
    server_csv = sys.argv[2] if len(sys.argv) > 2 else None
    ct, cr = load_csv(client_csv)
    plt.figure(figsize=(8, 4))
    plt.plot(ct, cr, label="client ratio", alpha=0.8)
    if server_csv:
        st, sr = load_csv(server_csv)
        plt.plot(st, sr, label="server ratio", alpha=0.8)
    plt.xlabel("tick")
    plt.ylabel("compression ratio (raw/comp)")
    plt.title("Rotor Twin Network Compression")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = "rotor_net_ratio.png"
    plt.savefig(out)
    print("Saved:", out)


if __name__ == "__main__":
    main()

