import numpy as np
import matplotlib.pyplot as plt


def gen_signal(n=100000):
    t = np.arange(n)
    x = np.sin(t*0.01) + 0.1*np.sin(t*0.05) + 0.01*np.random.RandomState(0).randn(n)
    return x.astype(np.float32)


def ema_predict(x, beta=0.9):
    m = 0.0
    pred = np.zeros_like(x)
    for i, v in enumerate(x):
        pred[i] = m
        m = beta*m + (1-beta)*v
    return pred


def main():
    x = gen_signal()
    pred = ema_predict(x)
    r = x - pred

    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=100, alpha=0.5, label="X", density=True)
    plt.hist(r, bins=100, alpha=0.5, label="R", density=True)
    plt.legend()
    plt.title("Entropy Reduction: X vs Residual R")
    plt.tight_layout()
    plt.savefig("entropy_hist.png")
    print("Saved: entropy_hist.png")


if __name__ == "__main__":
    main()

