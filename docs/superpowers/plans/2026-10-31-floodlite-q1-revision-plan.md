# FloodLite Q1-Revision Plan — complete work plan to submission

**Date:** 2026-10-31
**Owner:** Ibrahim Khalil (with Claude Code)
**Goal:** Lift the manuscript from "borderline Q1 / solid Q2" to a defensible Q1 submission by
(a) eliminating every manuscript↔code inconsistency, (b) adding a second, non-saturated dataset
with a genuine held-out test set and zero-shot external evaluation, and (c) re-testing the
negative-KD claim where headroom exists.
**Basis:** full-repo review + peer-review panel (2026-10-31) and the gap-filling literature
review in `docs/lit_review_gap_analysis.md`.

---

## Workstream A — Correctness fixes (P0, blockers)

| # | Item | Status | Owner/tooling |
|---|---|---|---|
| A1 | Quantization description → match code (`per_channel=False`, 80 calib images, op-type restriction); MobileViT mechanism re-grounded in FQ-ViT/RepQ-ViT | ✅ done 2026-10-31 | text |
| A2 | Abstract: "no measurable accuracy loss" removed; INT8/FP32 WASM FPS conflation fixed | ✅ done | text |
| A3 | lr/batch/seed claims in §4 + appendix match code | ✅ done | text |
| A4 | **User decision 2026-10-31: measure BOTH threading configs** (single-threaded primary, default pool appendix). Code ready: `benchmark_onnxruntime(..., intra_op_threads=)` + `bench_m2.py --threads both` (keys: `*_ort_cpu` = default pool, `*_ort_cpu_1t` = pinned). ⬜ run on M2 (~1.5 h) | ✅ done 2026-11-01 on the M2-side line (ORT 1.22.1 provenance pinned) | `bench_m2.py` on M2 |
| A5 | **Teacher ONNX re-benchmark** — `export_all.py` exports teacher.onnx; `bench_m2.py` already resolves `teacher.onnx`/`teacher_fold0.onnx` ORT targets. Run on M2 together with A4, then recompute every speedup claim (abstract, §5.4 "7–19×", §6.2, §8 "15.8×", F4) | ✅ done 2026-11-01: teacher ORT-CPU 37.6 ms → honest INT8 speedup 2.2×; all speedup claims recomputed | M2 |
| A6 | §8 conclusions: "multi-platform" → "CPU-and-browser"; new speedup numbers from A5 | ✅ done 2026-11-01 (merged with A5) | text |
| A7 | Run resp/feat Wilcoxon contrasts and fix the false "cannot be tested" sentence in §5.2; run the per-image fold-0 bootstrap (`stats.py --data_root`) to blunt the n=3 criticism | ✅ done: Wilcoxon (stats.json) + per-image bootstrap n=399 (stats_bootstrap.json) | Kaggle or M2 |
| A8 | T4: add per-fold INT8 IoU table (shows the bimodal collapse honestly; mean±std hides it) | ✅ done 2026-10-31 (per-fold column in T4 + §5.3 "differ in kind" paragraph citing ML-EXray) | text + `quantized.json` |

**Expected outcome:** manuscript states exactly what the code did; headline speedups survive
reproduction. Expect the teacher→Lite0-INT8 speedup to drop from 15.8× to roughly **2–4×**
(ORT is ~6–8× faster than PyTorch on your own student measurements). The honest framing:
"the teacher is 2–4× slower *and* 4–6× larger with no accuracy advantage" — still a clear
dominance argument, now unimpeachable.

## Workstream B — Literature integration (done today, maintenance ongoing)

- ✅ 11 verified references added; 6 claims corrected/rescoped (see `docs/lit_review_gap_analysis.md`).
- ✅ Hernández et al. (2022) full PDF read 2026-10-31 — all our claims about it confirmed (incl. Nano failures broader than stated; no quantization). See gap-analysis §4.3.
- ✅ Refs grown to 56 (2026-10-31): nemni2020rapid (UNOSAT rapid mapping), mateogarcia2021onboard (PhiSat-1 onboard flood seg), banbury2021mlperftiny, lin2020mcunet, smilkov2019tfjs, gholami2021quantsurvey, gou2021kdsurvey — all web-verified, integrated in §2.1/§2.2/§2.4/§2.5. Manuscript compiles clean (latexmk, 0 undefined citations). Remaining ~4 refs can come with the D-results writing pass.
- **Strategic note:** Schmalstieg et al. (arXiv, Sept 2026) shows the edge-flood niche is
  heating up. Submitting within ~6–8 weeks matters.

## Workstream C — Second dataset (flood-combined-fsegnet)

| # | Item | Effort | Gate |
|---|---|---|---|
| C1 | ✅ **GATE PASSED 2026-11-01**: 663 FSSD × 3,009 combined images, pHash hamming ≤ 8 → **0 hits in any role** (not even train-side). FSSD and combined are fully disjoint; D1/D2 cleared. Report: `code/runs/fssd_overlap_report.json`; gate-result sentence added to §3.1 | done | **PASSED** |
| C2 | ✅ done 2026-10-31. `floodlite/manifest.py` (torch-free split logic: exclude/duplicate handling, duplicate-aware kaggle folds, fixed FloodNet test, external zero-shot) + `floodlite/data.py` `make_combined_loaders()`/`source_balanced_sampler()` + PIL fallback for the 45 mislabelled .jpg (GIF/WebP) files. 13 torch-free tests pass locally; 6 loader tests skip locally (torch DLL broken on the Windows box) and must be run once on Kaggle/M2 | done | after C1 |
| C3 | ✅ done 2026-10-31. FloodNet CDLA-Permissive ✓, AIFloodSense CC BY 4.0 ✓, Kaggle "Flood Area Segmentation" (faizalkarim) = **CC0: Public Domain** (confirmed via Kaggle public API `licenseNameNullable` + page schema.org metadata). All 290 images keepable; masks self-annotated by uploader (Label Studio) — cite the Kaggle page in §4 | done | cleared |
| C4 | ✅ done 2026-10-31. §3.1 "Datasets" now has a "Combined generalization dataset" paragraph (sources + licences + roles + water definition + balanced sampler + no-flood-image rationale), citing karim2022floodarea / rahnemoonfar2021floodnet / simantiris2026aifloodsense. One TODO marker holds the spot for the C1 gate-result sentence. Compiles clean | done | gate sentence pending C1 run |

## Workstream D — New experiments (Kaggle T4, free tier)

**Matrix (reduced grid):** teacher + 3 students × {none, comb} × 3 folds on the combined
dataset = **21 training runs**; plus zero-shot evaluation of all existing FSSD checkpoints
on AIFloodSense `external_test` (no training).

| Run | What | Compute | Expected outcome |
|---|---|---|---|
| D1 | ✅ all 3 folds done; written up as §5.8 + T9 (3-fold means). **Findings:** (a) FP32 students transfer to AIFloodSense at 0.67–0.73 IoU, teacher no better (0.708), MobileViT best (0.730); KD gap zero-shot ±0.006 → negative-KD result extends beyond the training distribution. (b) **FAR 71–100% on water-free FloodNet scenes for every FP32 model** (FSSD has no negatives → "no flood" never supervised) — disqualifying for monitoring; motivates D2 directly. (c) **INT8 degradation compounds under shift:** Lite0 −0.088 (vs −0.043 in-domain), MobileNetV3-INT8 collapses to 0.057 flood-IoU (macro 0.39 is a predict-nothing artifact), MobileViT-INT8 halves (0.73→0.35). Manuscript: T6 + §5.6 written (fold-0 numbers, TODO → 3-fold means); §4.3 metric definitions extended (per-image macro, empty-image rule, flood-IoU, FAR). Output: `runs/3fold/zero_shot_fold{0,1,2}.json` | done (f0) | **Three new findings** |
| D2 | ✅ DONE 2026-11-03 (3 folds, ~10 h each; §5.9 + T10). FAR 0.70–1.00 → 0.03–0.09; comb-KD +flood-IoU (sig.) but inherits teacher false alarms. Combined-dataset training (21 runs) | ~45–55 T4-h ≈ 1.5–2 wk of free quota | Non-saturated regime: expected flood-IoU 0.6–0.8. **Decision point:** if KD helps here → paper upgrades to "KD helps iff headroom exists" (mechanism + boundary = Q1-grade). If KD still doesn't help → negative result generalises. Both outcomes publishable. |
| D3 | 🟡 running 2026-11-08 on Kaggle CPU (`floodlite-d3-ptq`, `scripts/quantize_combined.py`, all 3 students × none/comb × 3 folds). INT8 PTQ of combined-trained Lite0 (+ optionally the other two) | hours, M2 | Checks PTQ-stability finding holds off-FSSD |
| D4 | (Optional, stretch) one extra hardware target — Pi 4 or AWS Graviton (`bench_graviton.sh` exists) | 1–2 days setup | Restores some "multi-platform" breadth |

**Skip on combined data:** full 4-config KD grid, WASM re-runs (architecture properties, not
dataset properties — one sentence justifies this).

## Workstream E — Manuscript v3 rewrite + venue

1. Integrate D1–D3 into §4/§5 (new table: combined-dataset results + external zero-shot; per-fold INT8 table from A8).
2. Reframe contribution 2 based on D2's outcome.
3. Venue decision (recommendation): **Int. J. Applied Earth Observation & Geoinformation (JAG, Q1)** primary — efficient-EO-deployment fit; *Environmental Modelling & Software* (Q1) alternate; IJDRR only if a DRR-operational subsection is added; *Remote Sensing* as fast fallback.
4. Re-run review panel (fresh pass) on v3; fix; humanize pass; Zenodo DOI; `[your-username]` substitution; delete placeholder appendix; submit.

## Timeline (calendar, free-tier constraints)

- **Week 0 (now):** A7–A8, C1–C3, B leftovers; user runs A4/A5 on the M2 (~1 h total).
- **Week 1:** D1 zero-shot; start D2 queue on Kaggle.
- **Weeks 2–3:** D2 completes; D3; draft new results sections.
- **Week 4:** E1–E3, internal re-review, polish.
- **Week 5:** submission package (cover letter, highlights, disclosure, Zenodo).

## Risks

| Risk | Mitigation |
|---|---|
| FSSD↔combined duplicate leakage found (C1) | Drop affected images from combined val/test roles; document in §4 |
| D2 exceeds free T4 quota | Trim to 2 architectures (Lite0 + MobileViT) × {none, comb}; 14 runs |
| KD result on combined data ambiguous (within noise) | Report with per-image bootstrap (A7 machinery reused); frame as boundary-condition evidence |
| Scooped by Schmalstieg JSTARS acceptance | Cite as concurrent; our RGB/CPU/PTQ-stability scope is disjoint; submit fast |
| Teacher-ORT re-benchmark shrinks headline | Reframe to dominance (accuracy+size+latency) — already drafted in A5 note |

---

## Progress log

### 2026-10-31 (session 1)
- Workstream A: A1–A3 manuscript corrections applied (see `docs/lit_review_gap_analysis.md` §3).
- Workstream B: 11 verified refs added; claims 1–6 corrected/rescoped.
- Plan + gap-analysis documents created.

### 2026-10-31 (session 2)
- **C2 complete (code):**
  - `floodlite/manifest.py` — torch-free manifest loading + fold splits for
    flood-combined-fsegnet. Semantics: train = kaggle(fold≠f) + extra_train;
    val = kaggle(fold=f) minus `duplicate_of` copies; test = `floodnet_test`
    (424, identical across folds); external = `external_test` (94);
    `external_pool` never used; `exclude` rows dropped everywhere.
  - `floodlite/data.py` — `make_combined_loaders()` (dict of train/val/test/
    external loaders), `source_balanced_sampler()` (inverse-frequency weights,
    seeded), `_read_rgb()` PIL fallback for cv2-unreadable files (45 combined
    files named .jpg are WebP/PNG/GIF).
- **C1 complete (code):** `floodlite/phash.py` (DCT pHash, imagehash-compatible,
  chunked hamming matrix) + `scripts/check_fssd_overlap.py` (CLI gate; JSON
  report to `runs/fssd_overlap_report.json`; exit 1 on eval-role hits).
  Windows fix: case-insensitive rglob deduplication in `collect_fssd_images`.
- **Tests:** new `tests/test_phash.py` (5), `tests/test_manifest.py` (8),
  `tests/test_check_fssd_overlap.py` (2, planted-duplicate + clean fixtures),
  `tests/test_combined_loaders.py` (6, torch — skip locally). Existing test
  modules + conftest patched with `pytest.importorskip(..., exc_type=Exception)`
  so the suite runs on machines with a broken/absent torch (this Windows box:
  torch 2.14.1+cpu fails with WinError 1114 c10.dll). Suite: **15 passed,
  5 modules skipped** locally.
- **Data:** kagglehub installed; FSSD + combined datasets downloading to
  `C:\Users\user\orca\flood-data\{fssd,combined}` for the real C1 gate run.
- **A8 done:** T4 gained an "INT8 IoU per fold (0/1/2)" column
  (MobileNetV3 0.772/0.000/0.657; Lite0 0.903/0.885/0.867; MobileViT
  0.718/0.464/0.244) and §5.3 now distinguishes the two failure modes
  (bimodal constant-background collapse vs. graded degradation), citing
  ML-EXray for the former.
- **A7 (Wilcoxon half) done:** `scripts/stats.py` extended to all no-KD vs.
  {resp,feat,comb} contrasts with per-fold deltas; torch import made lazy so
  the fold-level stats run anywhere; `runs/3fold/stats.json` regenerated
  (9 contrasts; resp p=0.25 on MobileNetV3/MobileViT, feat p=0.75–1.0,
  comb p=0.25–1.0 — no contrast favours KD). §5.2's false "cannot be tested"
  sentence replaced with the real numbers; §4.4 updated to match.
  Fold-0 per-image bootstrap still ⬜ (needs torch → Kaggle/M2).
- **C3 done:** Kaggle Flood Area Segmentation (faizalkarim) licence confirmed
  **CC0: Public Domain** via Kaggle public API; `karim2022floodarea` added to
  refs.bib (now 49 entries). All three combined-dataset sources cleared.
- **User decisions (AskUserQuestion):** (1) M2 benchmark in BOTH threading
  configs — `benchmark_onnxruntime` gained `intra_op_threads`, `bench_m2.py`
  gained `--threads {default,single,both}` (default both; `_1t` key suffix);
  torch-free ORT test added (`tests/test_benchmark_ort.py`). (2) Fix torch on
  the Windows box via VC++ 2015–2022 x64 redistributable, then run D1 +
  bootstrap + loader tests locally. (3) Checkpoints "on this Windows machine"
  — **but no \*.pt found under the repo/Downloads/Documents/orca; path needed
  from user.**
- **Still open:** A4/A5 M2 run, A6 (§8 text after A5), A7 bootstrap, C1 gate
  run on real data (downloads in progress), checkpoint location, torch fix
  (user installs VC++ redist), loader tests on a torch machine, Hernández
  full-text skim.

### 2026-11-01
- **C1 GATE PASSED** (see Workstream C table): 0 pHash hits, FSSD ⊥ combined.
  §3.1 gate sentence added; manuscript compiles clean.
- **Checkpoints recovered:** user's zip (`Downloads/Rme Msc Materials/iielts/3fold.zip`)
  held everything — 42 .pt → `code/runs/3fold/`, 28 .onnx → `code/exports/`
  (incl. **teacher_fold0_fp32.onnx, 23.4 MB** — the missing A5 artifact; T1
  teacher FP32 cell filled, §8 teacher-size ratio now grounded at 4.5×).
  Repo JSONs kept as source of truth (zip JSONs skipped where present).
- **D1 zero-shot + A7 bootstrap launched locally** (background, CPU):
  `eval_zero_shot.py` (fold-0 teacher/baseline/students none+comb + FP32/INT8
  ONNX on floodnet_test 424 + external_test 94, per-image metrics)
  → `runs/3fold/zero_shot.json`; then `stats.py --data_root` fold-0 per-image
  bootstrap → updated `stats.json`.
- **Remaining user-side:** M2 benchmark session (A4/A5 latency legs — the
  teacher ONNX exists now, so it's just `bench_m2.py --threads both` plus a
  fresh `export_all.py` only if per-fold teacher exports are wanted);
  commit+push decision for D2 on Kaggle.
- **Per-image metric semantics fixed (TDD, suite 39 passed):** the first D1 run
  exposed that `compute_metrics` scored empty-GT/empty-pred images as IoU 0
  (`0/(0+eps)`), and **215/424 floodnet_test images contain no flood** — a
  perfect model would cap at ~0.49 macro IoU. Now: empty-vs-empty = correct
  rejection (IoU 1.0); `summarise_per_image` additionally reports
  `iou_flood_mean/std` (flood-containing images only), `false_alarm_rate`
  (water predicted on water-free images), and the n per group. First-run
  qualitative signal (to confirm in v2): external_test FP32 IoUs ~0.70–0.73,
  and **INT8 drops much more under domain shift** (Lite0 0.70→0.61,
  MobileViT 0.73→0.35) — a genuinely new finding if it holds. D1+bootstrap
  re-running with corrected semantics.

### 2026-11-01 — merge with the parallel M2-side revision line + push
- Remote main had **16 parallel commits** (co-author work on the M2):
  **A4/A5 were done there** — teacher ORT-CPU 37.6 ms (ORT 1.17) / 36.7 ms
  (1.22.1), honest INT8-vs-teacher speedup **2.2×**, threading claim fixed,
  provenance pinned; plus Sen1Floods11 OOD eval (all models collapse to IoU
  ≈0.16, their T6/§5.7), a measured HardSwish-vs-ReLU6 activation-range
  diagnostic (12–17×), pooled per-image bootstrap n=399 (KD significance
  story), decision matrix T7, error taxonomy T8, co-authors
  (Chawdhury/Khalil/Haque), **IJDRR targeted**, Zenodo/CITATION scaffolding.
- Integration: 7 code/docs commits cherry-picked cleanly onto origin/main
  (suite 41 passed after merge); the manuscript commit was **re-applied
  semantically** onto their revision instead of textually: kept their
  measured numbers/structure, re-fixed what they still had wrong (Hernández
  "reports no latency" claim — false; per-channel-weights claim — code says
  per_channel=False; "multi-platform" in §1/§8; ghost utils.py in appendix),
  re-added the 18 verified refs (38→56), and added the within-regime
  zero-shot as **§5.8 + T9** (renumbered; their T6=Sen1Floods11) with
  abstract/§1-contribution-4/§6/§7/§8 tie-ins. Compiles clean.
- **Pushed to origin/main (cdbed68..d1c69da)** after switching the machine's
  GitHub identity to NoorTasfia (`gh auth setup-git`; m-ibrahim-khalil has
  pull-only access).
- A4/A5/A6 are now effectively **done** (their measured 2.2× + my
  CPU-and-browser rescoping); remaining A-items: none. D2 queue is unblocked
  (repo main now has the combined-dataset code).

### 2026-11-01 (continued) — D1 integrated throughout the manuscript
- **New §5.6 + T6** (fold-0 numbers, TODO → 3-fold): three findings written up.
- **Stale-claim sweep after D1:** §7 "cross-domain evaluation unverified" →
  replaced with what D1 measured and what remains (training-side limitation,
  Sen1Floods11 modality); abstract gained the zero-shot sentence (transfer
  parity, FAR 71–100%, INT8-shift compounding); §1 gained contribution 4
  (zero-shot deployment reality check, overlap-verified); §6 deployment
  recommendations now carry the unattended-monitoring caveat + TODO(A5) marker
  on the 274 ms teacher sentence; §8 gained result 4 and future work reworded
  (OOD no longer "deferred"); appendix artifact list extended with
  zero_shot_fold{0,1,2}.json + fssd_overlap_report.json. Compiles clean.
- **`scripts/aggregate_zero_shot.py`** written (TDD, torch-free): merges
  per-fold zero-shot JSONs into cross-fold mean±std → `zero_shot_3fold.json`;
  T6 swap is mechanical once folds 1–2 finish (running in background).
  Suite: **41 passed, 2 xfailed**.

### 2026-10-31 (session 2, continued)
- **Torch fixed on the Windows box** (VC++ redistributable installed by user);
  full FloodLite deps installed. **Entire test suite now runs natively:
  34 passed, 2 xfailed, 0 skipped** — combined-loader, sampler, gate,
  benchmark-threading, and zero-shot tests all verified on real torch.
  (One env fix: hand-built ONNX test fixtures pin `ir_version = 8`.)
- **A6 (safe half):** §8 "multi-platform" → "CPU-and-browser"; the invalid
  15.8× sentence replaced by grounded IoU/size facts + a TODO comment wired to
  the A5 re-benchmark. T1 confirms teacher ONNX size was never measured.
- **D1 script written (TDD):** `scripts/eval_zero_shot.py` — per-image (macro)
  metrics on `floodnet_test` + `external_test` for PyTorch checkpoints AND
  FP32/INT8 ONNX exports (domain-shift degradation comparison); per-image IoU
  lists kept in the JSON for later paired stats. Shared helpers moved into
  `floodlite/metrics.py` (`evaluate_per_image`, `summarise_per_image`).
- **D2 driver ready:** `run_full_experiment.py` gained `--dataset combined`
  (manifest-driven loaders; val/test/external per-image metric blocks),
  `--kd_configs` (reduced grid), `--skip_quant`. FSSD path/schema unchanged.
- **Blocked only on:** combined download (background, ~2 h), checkpoint
  location (user), M2 session (user).
- **Checkpoint hunt (user said "on Kaggle"):** exhaustively searched the
  account `tasfianoorchowdhury` via the Kaggle API — the `floodlite` notebook
  has **no committed run** (lastRun: None → no API-retrievable output), no
  dataset holds the 42 .pt files, and every other kernel's output is
  FSegNet/knee-project material. **User decision: copy them from the M2** into
  `code/runs/3fold/` (also copy `code/exports/*.onnx` for the D1 INT8/FP32
  comparison). GitHub remote confirmed public: NoorTasfia/flood-segmentation.
- **D2 Kaggle notebook prepared:** `code/notebooks/floodlite_combined_kaggle.ipynb`
  — one fold per 12 h session (FOLD param, 7 runs/fold), clones the public
  repo, auto-locates manifest.csv, resumes via `--skip_if_exists` with a
  previous session's output attached, prints val/test/external IoU per model.
  **Prerequisite: the uncommitted combined-dataset code must be committed and
  pushed before the notebook can run** (the clone pulls main).
