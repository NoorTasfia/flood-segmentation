"""Native ONNX Runtime CPU latency for a list of ONNX files (second-host bench).

Same protocol as bench_m2.py: 1x3x256x256, 20 warm-up + 200 timed runs,
default ORT thread pool. Usage:
    python scripts/bench_ort_cpu.py --out runs/host2/lat_ort_cpu.json name=path.onnx ...
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def bench(path: str, warmup: int = 20, runs: int = 200) -> dict:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.zeros((1, 3, 256, 256), dtype=np.float32)
    feeds = {sess.get_inputs()[0].name: x}
    for _ in range(warmup):
        sess.run(None, feeds)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    p50 = float(np.percentile(ts, 50))
    return {"p50_ms": p50, "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "std_ms": float(ts.std()), "fps": 1000 / p50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    res = {"_meta": {"runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider (default threads)",
                     "cpu": platform.processor(), "platform": platform.platform(),
                     "protocol": "1x3x256x256 zeros, 20 warm-up + 200 timed runs"}}
    for spec in a.specs:
        name, path = spec.split("=", 1)
        res[name] = bench(path)
        print(f"{name:28s} p50 {res[name]['p50_ms']:.1f} ms  fps {res[name]['fps']:.1f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
