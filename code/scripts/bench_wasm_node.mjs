// Node.js WASM latency benchmark (onnxruntime-web, single-threaded SIMD).
// Same protocol as bench_wasm.html: 1x3x256x256 zeros, 20 warm-up + 200 timed.
// Usage: node bench_wasm_node.mjs <out.json> <name=path.onnx> [<name=path.onnx> ...]
import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const [outPath, ...specs] = process.argv.slice(2);
const results = {
  _meta: {
    runtime: "onnxruntime-web " + ort.env.versions.web + " (wasm, 1 thread, SIMD)",
    node: process.version, cpu: os.cpus()[0].model, platform: `${os.platform()} ${os.release()}`,
    protocol: "1x3x256x256 zeros, 20 warm-up + 200 timed runs, p50/p95/mean/std in ms",
  },
};
for (const spec of specs) {
  const [name, file] = spec.split("=");
  try {
    const sess = await ort.InferenceSession.create(readFileSync(file), { executionProviders: ["wasm"] });
    const x = new ort.Tensor("float32", new Float32Array(3 * 256 * 256), [1, 3, 256, 256]);
    const feeds = { [sess.inputNames[0]]: x };
    for (let i = 0; i < 20; i++) await sess.run(feeds);
    const ts = [];
    for (let i = 0; i < 200; i++) {
      const t0 = performance.now();
      await sess.run(feeds);
      ts.push(performance.now() - t0);
    }
    ts.sort((a, b) => a - b);
    const mean = ts.reduce((a, b) => a + b, 0) / ts.length;
    const r = {
      p50_ms: ts[Math.floor(ts.length * 0.5)], p95_ms: ts[Math.floor(ts.length * 0.95)],
      mean_ms: mean, std_ms: Math.sqrt(ts.reduce((a, b) => a + (b - mean) ** 2, 0) / ts.length),
    };
    r.fps = 1000 / r.p50_ms;
    results[name] = r;
    console.log(`${name.padEnd(28)} p50 ${r.p50_ms.toFixed(1)} ms  fps ${r.fps.toFixed(1)}`);
  } catch (e) {
    results[name] = { error: String(e && e.message ? e.message : e) };
    console.log(`${name} FAILED: ${results[name].error}`);
  }
}
writeFileSync(outPath, JSON.stringify(results, null, 2));
