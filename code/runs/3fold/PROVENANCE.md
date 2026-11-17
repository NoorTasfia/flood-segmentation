# `runs/3fold` provenance

Authoritative source file for each quantity reported in the manuscript
(matches the `main.tex` "Numbers source of truth" header):

| Reported quantity | Authoritative file | Producer |
|---|---|---|
| Segmentation metrics, Tables 2 & 3 | `summary.json` (mean±std over folds 0–2); `results.json` (per-fold raw) | `run_full_experiment.py` → `aggregate_results.py` |
| INT8 quantization, Table 4 | **`quantized.json`** | `quantize_all.py` (commit `cf9577e`, 2026-05-16) — Percentile-99.999% static QDQ recipe |
| Latency, Table 5 | `lat_m2.json` (M2 Pro ORT-CPU), `lat_wasm.json` (browser WASM SIMD) | `bench_m2.py`, `bench.mjs` |
| Fold-level Wilcoxon | `stats.json` | `stats.py` |
| INT8 recipe ablation, Table 12 / S5.3 | `ptq_mechanism.json` (per-tensor, per-channel, HardSwish-FP32, attention-FP32, early-Convs-FP32; folds 0-2; students + teacher) | `scripts/ptq_mechanism_ablation.py` (ONNX Runtime 1.22.1, torch 2.15 CPU, 2026-11-08/09) |
| MobileViT layer-group diagnostic, S5.3 | `ptq_mobilevit_groups.json` (per-channel, one Conv group quantized at a time; fold 0 complete, folds 1-2 for stem/stage0/stage1/early) | `scripts/ptq_mobilevit_diagnostic.py` |
| Second-host latency, Table 13 / S5.4 | `../host2/lat_ort_cpu.json` (native ORT 1.22.1), `../host2/lat_wasm_node.json` (ORT-Web 1.22.0, Node 22), `../host2/lat_ort_cpu_u8s8.json` (unsigned activations); fold-0 opset-17 artifacts | `scripts/bench_ort_cpu.py`, `scripts/bench_wasm_node.mjs` (AMD Ryzen 5 5600G, Windows 10) |
| Unsigned-activation INT8 accuracy, S5.3 item 4 | `../host2/u8s8_fold0.json` (fold 0 only) | ad-hoc script, same recipe as `ptq_mechanism_ablation.py` with `activation_type=QUInt8` |
| Zero-shot INT8, Table 9 | `zero_shot_3fold.json` (aggregate of `zero_shot_fold{0,1,2}.json`); per-tensor rows = `*_onnx_int8` (exports of `quantize_all.py`), per-channel rows = `*_onnx_int8_pc` (the `pc` artifacts of `ptq_mechanism_ablation.py`, no-KD students) | `scripts/eval_zero_shot.py`, `scripts/eval_zero_shot_onnx.py` (2026-11-10), `scripts/aggregate_zero_shot.py` (std = population std, ddof=0) |
| Combined-corpus INT8, Table 11 | `../combined/quantized.json` (per-tensor) and `../combined/quantized_pc.json` (per-channel, 18 checkpoints, folds 0-2, 2026-11-10); std = sample std (ddof=1) | `scripts/quantize_combined.py`, `scripts/quantize_combined_pc.py` (ONNX Runtime 1.22.1; per-channel run on the x86 host from the Kaggle-produced `*_fp32_preproc.onnx` exports) |

Note on Table 4 vs Table 12: Table 4 keeps the original per-tensor `quantized.json`
numbers (ORT 1.22.1 on the M2 host). The `pt` variant of `ptq_mechanism.json` is a
re-run of that recipe on the x86 host from a fresh opset-17 export; it reproduces the
fold pattern (e.g. fold-1 MobileNetV3-Small IoU 0.0001 in both) but not the exact
values (fold-0 MobileNetV3-Small 0.772 vs 0.743), because the calibration batches are
drawn by a differently seeded loader. The manuscript quotes Table 4 for the per-tensor
headline and Table 12 for all within-ablation comparisons.

## Training hyperparameters (as reported in manuscript Section 4.1)

| Quantity | Value | Source |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-4 | `floodlite/train.py` (`train_teacher` line 20, `train_student_kd` line 91) |
| Learning rate, teacher | 1e-4 | `train_teacher(..., lr=1e-4)` default; notebook cell `AdamW(teacher.parameters(), lr=1e-4)` |
| Learning rate, students | 5e-4 | `train_student_kd(..., lr=5e-4)` default; notebook cell `AdamW(opt_params, lr=5e-4)` |
| Schedule | CosineAnnealingLR, T_max = epochs | `floodlite/train.py` lines 21, 92 |
| Epochs | 35 (teacher and students) | `scripts/run_full_experiment.py --epochs_teacher/--epochs_student` default 35, the producer of `results.json` per the table above. The standalone notebook `floodlite_kaggle.ipynb` defaults to 50 and was not the producer. |
| Batch size | 8, all models | `scripts/run_full_experiment.py --batch_size` default 8; combined-corpus notebook passes no override |
| Seed | 42 | `KFold(random_state=42)`; sampler seed 42 |
| Checkpoint | best validation-IoU epoch of the reported fold (optimistic selection, shared by all configurations) | `train_*` in `floodlite/train.py` |

Corrected 2026-11-17: the manuscript previously stated lr 3e-4 and batch 16 for students; neither value appears in the code.

## Known discrepancy — do NOT use the INT8 block inside `results.json`

`results.json` (commit `d5557ff`, 2026-05-03) was produced by an earlier run of
`run_full_experiment.py` whose **inline** INT8 step used a superseded PTQ recipe.
Its per-fold INT8 IoU values differ materially from `quantized.json` — e.g. fold-0
MobileNetV3-Small INT8 IoU is 0.449 in the old block vs 0.772 in `quantized.json` —
and are **not** the numbers reported in the paper. The dedicated `quantize_all.py`
sweep (`quantized.json`) supersedes it.

To prevent accidental reuse, the stale per-fold block in `results.json` has been
renamed from `"quantized"` to `"quantized__SUPERSEDED__see_PROVENANCE"`. No code in
`code/` reads that block (`aggregate_results.py` and `stats.py` read only the
`teacher` / `baseline_mobilenetv2` / `students` entries).

## Statistical significance

The powered per-image paired bootstrap for the KD and teacher-vs-student contrasts is
**`stats_bootstrap.json`** (produced by `scripts/stats_bootstrap.py --data_root <FSSD>
--ckpt_dir runs/3fold`, using the 42 per-fold checkpoints; $n=399$ pooled held-out
images, 10,000 replicates, seed-fixed). It was independently re-run and reproduces
bit-for-bit. This file backs every confidence interval and p-value in the manuscript
(Sections 3.5, 5.2, 7).

The `"bootstrap_fold0": {"skipped_reason": ...}` entry inside `stats.json` is a
**different, earlier** analysis — a fold-0-only pixel bootstrap from `scripts/stats.py`
— which was superseded by the per-image `stats_bootstrap.json` and is not referenced by
the manuscript. `stats.json` remains authoritative only for the fold-level Wilcoxon.
