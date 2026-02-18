"""
Read a Nangila chunked compressed HDF5 checkpoint and reconstruct the 1D signal.

Requirements: pip install h5py numpy
"""
import h5py
import numpy as np


def linear_predict(prev, curr):
    if curr is None:
        if prev is None:
            return None
        return prev
    if prev is None:
        return curr
    # 2*curr - prev
    return curr + (curr - prev)


def read_compressed_1d(path: str, dataset: str = "traj") -> np.ndarray:
    with h5py.File(path, "r") as f:
        grp = f[dataset]
        blob = grp["blob"][...]
        offsets = grp["offsets"][...]
        scales = grp["scales"][...]
        total_len = int(grp.attrs["total_len"])  # type: ignore
        chunk = int(grp.attrs["chunk"])  # type: ignore
        eps = float(grp.attrs["epsilon"])  # type: ignore
        _codec = grp.attrs["codec"].astype(str)  # type: ignore

    out = np.empty(total_len, dtype=np.float32)
    prev = None
    curr = None
    write_ptr = 0

    for c in range(len(scales)):
        start = int(offsets[c])
        end = int(offsets[c + 1])
        scale = float(scales[c])
        chunk_bytes = blob[start:end]
        # interpret as little-endian int16
        q = np.frombuffer(chunk_bytes.tobytes(), dtype="<i2")
        # dequantize residual
        residual = q.astype(np.float32) * scale
        # predictor
        pred = linear_predict(prev, curr)
        if pred is None:
            recon = residual
        else:
            recon = pred + residual
        # write; last chunk may be trimmed
        take = min(chunk, total_len - write_ptr)
        out[write_ptr: write_ptr + take] = recon[:take]
        write_ptr += take
        # shift history
        prev = curr
        curr = recon

    # validation (optional): epsilon bound
    # The absolute error should be <= eps (plus rounding noise)
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "traj_ckpt.h5"
    data = read_compressed_1d(path, "traj")
    print("Read", data.shape, "from", path)

