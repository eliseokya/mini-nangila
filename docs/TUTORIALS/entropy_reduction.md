## Tutorial: Verifying the Entropy Reduction Lemma (CPU)

Goal: Show that predictive-residual coding reduces entropy on time-correlated signals:
H(R) < H(X), where R = X − \hat{X}.

### 1) Generate Correlated Data (Rust example)

Use the throughput bench to simulate training-like signals and residuals:

```
cargo run --release --manifest-path examples/throughput_bench/Cargo.toml > /tmp/ai_bench.txt
```

This logs compression ratios (proxy for entropy reduction). For a direct entropy check, use Python below.

### 2) Direct Entropy Measurement (Python, CPU)

```
python - <<'PY'
import numpy as np

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

def entropy(v, bins=256):
    h, _ = np.histogram(v, bins=bins, density=True)
    p = h[h>0]
    return -(p*np.log2(p)).sum()*(1.0/bins)  # coarse proxy

x = gen_signal()
pred = ema_predict(x, 0.9)
r = x - pred
print("H(X)=", entropy(x), "H(R)=", entropy(r))
PY
```

Expected: H(R) < H(X) for correlated signals.

### 3) Visualize
Plot histograms/densities of X and R to illustrate sharpening around zero for R.
