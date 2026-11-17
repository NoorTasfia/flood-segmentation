# FloodLite Resubmission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the FloodLite benchmark paper into a methods paper (gated distillation + sensitivity-guided mixed-precision PTQ), tested on a third external dataset (RescueNet), restructured per the Q1-anatomy audit, and retargeted to the Journal of Real-Time Image Processing.

**Architecture:** Two small additions to the existing training/quantization code (a correctness gate in `floodlite/losses.py` + `train.py`; a greedy group-selection script on top of `scripts/ptq_mechanism_ablation.quantize`), one new loader for RescueNet, and the stats/aggregation scripts extended to the new configurations. Training runs on Kaggle T4 through the existing notebook path; everything else runs locally on CPU. Manuscript edits follow once numbers land.

**Tech Stack:** PyTorch 2.8 / smp 0.4 / timm 1.0 (Kaggle); onnxruntime 1.22.1 quantization (local `venv_q`); pytest; elsarticle now, Springer template at the end.

**Spec:** `docs/superpowers/specs/2026-11-17-floodlite-resubmission-spec.md`

## Global Constraints

- Manuscript numbers come only from the per-quantity files named in `code/runs/3fold/PROVENANCE.md` and `code/runs/combined/`; never from the `quantized__SUPERSEDED__see_PROVENANCE` block.
- `code/runs/` is gitignored: results JSONs need `git add -f`.
- Kaggle account-changing actions (notebook push/run, dataset publish) take a dry run first and an explicit "yes" from the user before `--yes`. Never echo `KAGGLE_API_TOKEN`.
- Downloaded archives go in their own empty directory; scripts that read them run with `python -I`.
- The Windows box's torch is broken: torch-dependent tests run on Kaggle or the M2; locally they skip (`pytest.importorskip(..., exc_type=Exception)`). ONNX/quantization work uses the scratchpad `venv_q` (`$S/venv_q/Scripts/python.exe`, torch 2.15 cpu, onnxruntime 1.22.1).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Do not edit the EMS-submitted workspace `C:\Users\user\orca\workspaces\flood-segmentation\xyz`.
- Alpha=0.5, beta=0.1, T=4, lr 5e-4 (students), 35 epochs, batch 8, seed 42, source-balanced sampler: identical to the "comb" runs so the gate is the only variable.
- The blind manuscript must not contain the Kaggle handle, author names, or repository URLs.

## Review Focus

1. A gated batch where the teacher is wrong on every pixel: the response loss must be 0 (not NaN from a 0/0 mean). Test in Task 2.
2. Feature-gate downsampling when the decoder-block-2 feature map is 32x32 and the mask is 256x256: the weights must be area-averaged, not nearest-sampled, and `mask.sum()==0` must not divide by zero. Test in Task 2.
3. Mixed-precision selection must never read the validation fold: the sensitivity and stopping criterion use calibration images only. Test in Task 6 (selection is a pure function of a sensitivity dict; the script passes calibration-loader IoU).
4. RescueNet label palette: the water class index must be verified from the mask files, not assumed; a wrong index silently gives FAR = 0. Test in Task 7 (unit test on a synthetic mask + a one-off histogram check on real files).
5. Results merging: `results_gated_fold{N}.json` merged into `results_fold{N}.json` must not overwrite the existing `none`/`comb` blocks. Test in Task 3.

---

### Task 0: Snapshot the current state

**Files:**
- Modify: none (git only)

- [ ] **Step 1: Commit the uncommitted manuscript, metadata and script changes** (excluding `docs/~$oodEDL_Mamba_Research_Plan.docx` and `submission_package/`)

```bash
cd C:/Users/user/orca/flood-segmentation
git add -A -- ':!docs/~$oodEDL_Mamba_Research_Plan.docx' ':!submission_package'
git add -f code/runs/3fold/*.json code/runs/combined/*.json code/runs/host2/*.json
git commit -m "docs(manuscript): IJDRR package state, per-channel re-quantization, figures F6-F8, two-author list

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git tag ijdrr-rejected-2026-11-15
```

- [ ] **Step 2: Verify** `git status --short | wc -l` prints 0 apart from the two excluded paths.

---

### Task 1: Code/paper consistency and selection-bias statement (no compute)

**Files:**
- Modify: `manuscript/sections/04_experimental_setup.tex` (Implementation paragraph)
- Modify: `manuscript/sections/03_methods.tex` (Datasets paragraph, after the three-fold disclosure)
- Modify: `code/runs/3fold/PROVENANCE.md` (add a "Training hyperparameters" block)

- [ ] **Step 1: Confirm the values used for the archived runs.** Run: `grep -nE 'epochs_|lr=|BATCH' code/notebooks/floodlite_kaggle.ipynb code/scripts/run_full_experiment.py code/floodlite/train.py`. Expected: teacher lr 1e-4, students 5e-4, batch 8, epochs 35 (script default). If the notebook that produced `runs/3fold` used 50 epochs, the PROVENANCE block must say which.
- [ ] **Step 2: Edit Section 4.1** to read: AdamW, learning rate 1e-4 (teacher) and 5e-4 (students), weight decay 1e-4, cosine annealing, batch size 8 for all models, 35 epochs. Remove "or 16 (students)".
- [ ] **Step 3: Add one paragraph to Section 3.1** after the three-fold disclosure: "Checkpoints are the best-validation-IoU epoch of each run, and Tables 2-5 report that same held-out fold, so the in-distribution numbers carry a small optimistic selection bias; the external test sets of Sections 5.7-5.9 are never used for selection and are free of it."
- [ ] **Step 4: Record the hyperparameters in PROVENANCE.md** under a new heading with the exact values and the file/line they come from.
- [ ] **Step 5: Build and commit.** Run: `cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex`. Expected: 0 undefined references. Commit: `docs(manuscript): align hyperparameters with code; state checkpoint-selection bias`.

---

### Task 2: Gated distillation losses (TDD)

**Files:**
- Modify: `code/floodlite/losses.py`
- Modify: `code/floodlite/train.py:58-110` (`train_student_kd`)
- Modify: `code/scripts/run_full_experiment.py:65-67,136-139`
- Test: `code/tests/test_losses_gated.py`

**Interfaces:**
- Produces: `teacher_correct_mask(teacher_logits: Tensor[B,1,H,W], target: Tensor[B,1,H,W], threshold: float = 0.5) -> Tensor[B,1,H,W]` (float 0/1, detached).
- Produces: `response_kd_loss(student_logits, teacher_logits, T=4.0, mask: Tensor | None = None) -> Tensor` (mean over masked pixels; returns `0.0` tensor when `mask.sum()==0`).
- Produces: `FeatureKDLoss.forward(f_s, f_t, mask: Tensor | None = None) -> Tensor` (mask area-averaged to `f_t.shape[-2:]` with `F.adaptive_avg_pool2d`; weighted MSE `sum(w * (g(f_s)-f_t)^2) / (w.sum()*C + eps)`).
- Produces: `train_student_kd(..., gate: bool = False)`; when `gate` is true both KD terms receive `teacher_correct_mask(t_logits, y)`.
- Produces: KD config name `"gated"` in `run_full_experiment.py` (`use_resp=use_feat=True`, `gate=True`); checkpoint `{sname}_gated_fold{N}.pt`; results key `{sname}_gated`.

- [ ] **Step 1: Write the failing tests** in `code/tests/test_losses_gated.py` (torch via `importorskip`):

```python
def test_mask_marks_teacher_correct_pixels():
    t = torch.tensor([[[[ 3., -3.], [ 3., -3.]]]]); y = torch.tensor([[[[1., 0.], [0., 1.]]]])
    assert teacher_correct_mask(t, y).tolist() == [[[[1., 1.], [0., 0.]]]]

def test_masked_response_equals_unmasked_when_teacher_perfect(tiny_batch):
    x, y = tiny_batch; t = (y * 2 - 1) * 6; s = torch.randn_like(y)
    full = response_kd_loss(s, t); m = teacher_correct_mask(t, y)
    assert torch.allclose(response_kd_loss(s, t, mask=m), full)

def test_masked_response_is_zero_when_teacher_all_wrong(tiny_batch):
    x, y = tiny_batch; t = (1 - y) * 12 - 6; s = torch.randn_like(y)
    assert response_kd_loss(s, t, mask=teacher_correct_mask(t, y)).item() == 0.0

def test_feature_mask_is_area_averaged_and_safe():
    loss = FeatureKDLoss(4, 8); f_s = torch.randn(1, 4, 32, 32); f_t = torch.randn(1, 8, 32, 32)
    mask = torch.zeros(1, 1, 256, 256); mask[..., :128, :] = 1  # top half correct
    assert loss(f_s, f_t, mask=mask).item() > 0
    assert loss(f_s, f_t, mask=torch.zeros(1, 1, 256, 256)).item() == 0.0
```

- [ ] **Step 2: Run them** on a torch-capable machine: `pytest code/tests/test_losses_gated.py -v`. Expected: FAIL (`teacher_correct_mask` not defined). On the Windows box they skip; run on the M2 or in the Kaggle notebook's first cell.
- [ ] **Step 3: Implement** the three functions per the Interfaces block in `floodlite/losses.py`; thread `gate` through `train_student_kd` (compute the mask inside the `no_grad` teacher block); add `"gated"` to `--kd_configs` choices and the `use_resp/use_feat/gate` mapping in `run_full_experiment.py`.
- [ ] **Step 4: Run the tests.** Expected: 4 PASS. Also run `pytest code/tests -q` to confirm nothing else broke.
- [ ] **Step 5: Commit:** `feat(kd): ground-truth-gated response and feature distillation`.

---

### Task 3: Gated KD runs on Kaggle and results merge

**Files:**
- Create: `code/notebooks/floodlite_combined_gated_kaggle.ipynb` (copy of `floodlite_combined_kaggle.ipynb` with the changes below)
- Create: `code/scripts/merge_results.py`
- Test: `code/tests/test_merge_results.py`
- Output: `code/runs/combined/results_fold{0,1,2}.json` (gated blocks added), checkpoints in `C:\Users\user\orca\flood-data\combined-ckpts\`

**Interfaces:**
- Produces: `merge_results(base: dict, extra: dict, fold: int) -> dict` adds every key of `extra[f"fold_{fold}"]["students"]` to `base[f"fold_{fold}"]["students"]`, raises `KeyError` if a key already exists.
- Consumes: Task 2's `"gated"` config.

- [ ] **Step 1: Write the failing test** for `merge_results` (existing key raises; new key appears; other folds untouched).
- [ ] **Step 2: Implement** `code/scripts/merge_results.py` with CLI `--base runs/combined/results_fold0.json --extra <path> --fold 0` writing back to `--base`.
- [ ] **Step 3: Run the test.** Expected: PASS. Commit: `feat(scripts): merge_results for per-config result files`.
- [ ] **Step 4: Prepare the notebook.** Inputs: dataset `tasfianoorchowdhury/flood-combined-fsegnet`, private dataset `tasfianoorchowdhury/floodlite-combined-ckpts` (teacher checkpoints). The run cell copies `teacher_fold{N}.pt` into `--out_dir` and calls:

```
python scripts/run_full_experiment.py --data_root /kaggle/input/flood-combined-fsegnet \
  --dataset combined --kd_configs gated --skip_quant --skip_if_exists \
  --fold N --folds 1 --out_dir /kaggle/working/runs
```

One fold per kernel version (fits the 12 h limit; three students x 35 epochs took under 5 h per fold in D2).
- [ ] **Step 5: Dry run the push** with the kaggle skill (`notebook-push <dir>` without `--yes`), show the user the output, and stop for their yes. Then push with `--yes`, run fold 0, `notebook-wait`.
- [ ] **Step 6: Fetch outputs**, save `results.json` as `code/runs/combined/results_gated_fold0.json`, run `merge_results.py`, copy checkpoints to `combined-ckpts`. Repeat Steps 5-6 for folds 1 and 2.
- [ ] **Step 7: Verify** `python -c "import json;d=json.load(open('code/runs/combined/results_fold0.json'));print([k for k in d['fold_0']['students'] if k.endswith('_gated')])"` lists three models. Commit with `git add -f`: `results: gated-KD students on the combined corpus (3 folds)`.

---

### Task 4: Statistics for the gated contrast

**Files:**
- Modify: `code/scripts/stats_combined.py` (`main`, and `load_test_iou` to also load `pred_flood` per image if not already present)
- Modify: `code/scripts/aggregate_combined.py` (include `_gated` rows in `summary.json`)
- Test: `code/tests/test_stats_combined.py`
- Output: `code/runs/combined/stats_combined.json` (regenerated), `code/runs/combined/summary.json`

**Interfaces:**
- Produces: for each student, blocks `gated_vs_none` and `gated_vs_comb` with the same fields the existing `comb_vs_none` block has (`iou_flood_delta`, `far_delta`, each with `mean, ci_lo, ci_hi, p`), plus `did` for gated computed exactly as `_did(none_fa, gated_fa, t_fa)`.
- Consumes: Task 3's merged results.

- [ ] **Step 1: Write the failing test** on a synthetic results dict: gated with identical FAR to none gives `far_delta.mean == 0`; the contrast keys exist for all three students.
- [ ] **Step 2: Generalise** the model-pair loop to `KD_TAGS = ("none", "comb", "gated")` and emit the two new blocks. Run: `python scripts/stats_combined.py --data_root C:/Users/user/orca/flood-data/combined --out runs/combined/stats_combined.json` (in `venv_q`, numpy only).
- [ ] **Step 3: Run tests and `aggregate_combined.py`.** Expected: PASS; `summary.json` has `_gated` rows. Record the headline (gated IoU gain vs none, gated FAR delta CI, DiD CI) in PROVENANCE.md. Commit with `git add -f`.

---

### Task 5: INT8 of the gated students and zero-shot evaluation

**Files:**
- Modify: none (reuse `scripts/quantize_combined_pc.py --kd gated`, `scripts/quantize_combined.py --kd gated`, `scripts/eval_zero_shot_onnx.py`)
- Output: `code/runs/combined/quantized_pc.json`, `quantized.json` (gated entries added)

- [ ] **Step 1: Run** in `venv_q`: `python scripts/quantize_combined_pc.py --data_root <combined> --exports_dir C:/Users/user/orca/flood-data/combined-exports --ckpt_dir C:/Users/user/orca/flood-data/combined-ckpts --kd gated --out runs/combined/quantized_pc.json` and the per-tensor equivalent. One model per process (memory), `--num_workers 0`.
- [ ] **Step 2: Verify** both JSONs contain `*_gated` entries for folds 0-2 with `val`, `test`, `external` blocks and `false_alarm_rate`. Commit with `git add -f`: `results: INT8 PTQ of gated-KD students`.

---

### Task 6: Sensitivity-guided mixed-precision PTQ

**Files:**
- Create: `code/floodlite/mixed_precision.py`
- Create: `code/scripts/ptq_mixed_precision.py`
- Test: `code/tests/test_mixed_precision.py`
- Output: `code/runs/3fold/ptq_mixed.json`, `code/runs/combined/ptq_mixed_mobilevit.json`, `code/runs/host2/lat_mixed.json`, artifacts `<work_dir>/<model>_fold{N}_mixed_int8.onnx`

**Interfaces:**
- Produces: `select_fp32_groups(sensitivity: dict[str, float], loss_after: Callable[[list[str]], float], tau: float = 0.02) -> list[str]` (pure: sort groups by sensitivity descending; add one at a time; stop when `loss_after(selected) <= tau`; return selected; returns all groups if never satisfied).
- Produces: `group_prefixes(model_name: str) -> dict[str, tuple[str, ...]]` returning the ONNX node-name prefixes per encoder stage for all four networks (MobileViT groups copied from `ptq_mobilevit_diagnostic.GROUPS`; CNNs: stem/blocks per timm/smp naming, decoder as one group).
- Produces: `int8_mac_share(onnx_path) -> float` = MACs of quantized Conv/MatMul nodes / MACs of all Conv/MatMul nodes.
- Consumes: `scripts.ptq_mechanism_ablation.quantize(preproc, out, batches, *, per_channel, op_types, exclude)` and `preprocess(fp_path)`; calibration batches as in that script (80 train images of the fold).

- [ ] **Step 1: Write the failing tests:** `select_fp32_groups` with sensitivity `{a:.3,b:.1,c:.01}` and `loss_after` returning `.3,.1,.015` for 0,1,2 groups selected returns `[a,b]`; returns `[]` when `loss_after([]) <= tau`; returns all when loss never drops; `int8_mac_share` on a tiny two-Conv ONNX with one node excluded equals the MAC ratio.
- [ ] **Step 2: Run tests.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `floodlite/mixed_precision.py` and the script with CLI `--data_root --ckpt_dir --work_dir --models ... --folds ... --tau 0.02 --out`. Per model and fold: preprocess, calibrate once, compute `sensitivity[g]` = calibration-IoU drop with only g quantized (reuse the diagnostic's "only" mode), run `select_fp32_groups`, write the mixed artifact, then evaluate on the validation fold and record `fp32_iou, mixed_iou, selected, int8_mac_share, size_mb`. Calibration-set IoU is the only quantity the selection sees.
- [ ] **Step 4: Run tests.** Expected: PASS. Commit: `feat(ptq): sensitivity-guided mixed-precision INT8 recipe`.
- [ ] **Step 5: Run** on FSSD folds 0-2 for `mobilenetv3_small efficientnet_lite0 mobilevit_xxs teacher` (ckpts in `C:\Users\user\orca\flood-data\floodlite-ckpts`) and on combined MobileViT-XXS none/comb/gated, folds 0-2. Expected: CNNs select `[]`; MobileViT selects early groups and lands within 0.02 of FP32 on validation, or the run records that the budget was not met.
- [ ] **Step 6: Zero-shot and latency** for the MobileViT mixed artifacts: `scripts/eval_zero_shot_onnx.py --suffix mixed_int8 --tag int8_mixed`, and `scripts/bench_ort_cpu.py --out runs/host2/lat_mixed.json <specs>` on the Ryzen (and `bench_m2.py` on the M2 when the co-author can run it; mark pending otherwise).
- [ ] **Step 7: Commit results** with `git add -f`: `results: mixed-precision PTQ (FSSD 3 folds, combined MobileViT, host-2 latency)`.

---

### Task 7: RescueNet zero-shot test set

**Files:**
- Create: `code/scripts/prepare_rescuenet.py`
- Modify: `code/floodlite/data.py` (add `make_rescuenet_loader(root, *, batch_size=8, img_size=256, num_workers=0) -> DataLoader`)
- Modify: `code/scripts/eval_zero_shot.py` and `eval_zero_shot_onnx.py` (add `--rescuenet_root`; emit a `rescuenet` block beside `afs` and `floodnet_test`)
- Test: `code/tests/test_rescuenet.py`
- Output: `C:\Users\user\orca\flood-data\rescuenet\{Image,Mask,manifest.csv}`, `code/runs/rescuenet/overlap_report.json`, `code/runs/rescuenet/zero_shot.json`

**Interfaces:**
- Produces: `binarise_rescuenet_mask(label: np.ndarray, water_index: int) -> np.ndarray` (uint8 0/255).
- Produces: `manifest.csv` columns `image,mask,source_image,width,height,flood_fraction` with `source = rescuenet_test`.

- [ ] **Step 1: Download** `royzhao2004/rescuenet-test` (2.4 GB, CDLA-Permissive) with `kaggle:kaggle dataset-download` into the empty dir `C:\Users\user\orca\flood-data\rescuenet_raw`. Read-only action, no dry run needed.
- [ ] **Step 2: Verify the water index.** Run `python -I` over 20 label PNGs printing `np.unique` and compare with the RescueNet class list (0 background, 1 water, 2-5 buildings, 6 vehicle, 7-8 roads, 9 tree, 10 pool). Record the verified index in the script as a constant with a comment citing the file inspected. Do not proceed on an assumption.
- [ ] **Step 3: Write the failing tests:** `binarise_rescuenet_mask` maps only the water index to 255; the loader yields `(x[8,3,256,256], y[8,1,256,256])` from a synthetic 10-image root.
- [ ] **Step 4: Implement** the script (resize nothing; copy JPGs, write binary PNG masks, manifest) and the loader (reuse `FSSD(pairs, img_size, train=False)`).
- [ ] **Step 5: Overlap check:** run `scripts/check_fssd_overlap.py --fssd_root <fssd> --combined_root <rescuenet> --out runs/rescuenet/overlap_report.json` and again with `--fssd_root <combined>`. Expected: 0 hits at Hamming <= 8 (FloodNet and RescueNet are different events; if hits appear, exclude them and say so in the paper).
- [ ] **Step 6: Evaluate** every FSSD-trained and combined-trained checkpoint (none/comb/gated; FP32 via `eval_zero_shot.py`, INT8 per-tensor/per-channel/mixed via `eval_zero_shot_onnx.py`) on RescueNet; write `runs/rescuenet/zero_shot.json` with the same per-model structure as `runs/3fold/zero_shot_fold{N}.json`. Record `n_noflood_images`.
- [ ] **Step 7: Run tests, commit** code and (with `-f`) results: `feat(data): RescueNet test split as third zero-shot set; results`.

---

### Task 8: Tables and figures for the new results

**Files:**
- Create: `manuscript/tables/T14_gated_kd.tex`, `manuscript/tables/T15_mixed_precision.tex`, `manuscript/tables/T16_rescuenet.tex`
- Modify: `code/scripts/make_tables.py` (emit the three tables from the JSONs; no hand-typed numbers)
- Modify: `code/scripts/make_graphical_abstract.py` (panel 3 becomes "gated distillation keeps the gain, drops the false alarms")
- Modify: `manuscript/figures/` via `make_detection_gallery.py` (add a gated-model column)

- [ ] **Step 1: Extend `make_tables.py`** with three generators reading `summary.json`, `stats_combined.json`, `ptq_mixed.json`, `lat_mixed.json`, `rescuenet/zero_shot.json`. T14 rows: student x {none, comb, gated}; columns flood-IoU (FloodNet test), FAR, delta vs none with CI, DiD with CI. T15 rows: network x fold; columns per-channel loss, mixed loss, groups kept FP32, INT8 MAC share, size, Ryzen p50 (M2 p50 if available). T16: RescueNet IoU (all), IoU (flood), FAR for FSSD-trained, combined-trained, gated, and their INT8 variants.
- [ ] **Step 2: Regenerate** all tables and figures; `latexmk` build; check 0 undefined references. Commit: `docs(manuscript): tables T14-T16 and figures for gated KD, mixed precision, RescueNet`.

---

### Task 9: Manuscript restructure (audit items 1-6, 10-15)

**Files:**
- Modify: `manuscript/main.tex` (title), `sections/00_abstract.tex`, `01_introduction.tex`, `02_related_work.tex`, `03_methods.tex`, `05_results.tex`, `06_discussion.tex`, `07_limitations.tex`, `08_conclusions.tex`, `tables/T1_footprints.tex`
- Create: `manuscript/tables/T0_positioning.tex`, `manuscript/sections/10_supplementary.tex`

- [ ] **Step 1: Title.** Replace with the [Method] for [Task] with [Distinguishing feature] form; working title: "Gated distillation and sensitivity-guided INT8 for deployable flood segmentation: three failures that accuracy benchmarks hide". Update `CLAUDE.md`, `CITATION.cff`, `.zenodo.json`.
- [ ] **Step 2: Introduction.** Keep the funnel; end it with four numbered gaps (G1 deployability unmeasured on GPU-free hardware; G2 PTQ collapse reported but undiagnosed; G3 flood-only corpora never audited on dry scenes; G4 KD judged on IoU only); then four verb-first one-sentence contributions C1-C4 each naming its table (C1 benchmark, Tables 5/13; C2 diagnosis + mixed-precision recipe, Tables 12/15; C3 false-alarm audit + data requirement, Tables 9/10/16; C4 error-inheritance + gated KD, Tables 10/14). Delete Section 1.2; keep the roadmap paragraph. State each of the four general rules once, in the paragraph after the contributions.
- [ ] **Step 3: Related work.** Add `T0_positioning.tex` (study, modality, hardware, quantization, KD, dry-scene evaluation, limitation, this paper) covering Karci, Hernandez, Mateo-Garcia, Nie, Bhattacharjya, Schmalstieg; replace the five scattered "to our knowledge" sentences with one closing subsection "Synthesis" that restates G1-G4.
- [ ] **Step 4: Methods.** Add 3.4.1 "Ground-truth-gated distillation" (equations: mask, masked response, weighted feature) and 3.5.1 "Sensitivity-guided mixed precision" (the three-step recipe, tau, calibration-only selection). Justify alpha, beta, T (Hinton/FitNets defaults; kept fixed so configurations differ by one variable) and block 2 (1/8 resolution, last stage where both decoders have matching spatial size). Add a dataset-card table for the combined corpus with the spec's counts; name it by constituents; remove the Kaggle handle. Move the toolchain history (Conv2dSame, CUDA calibration, dynamo) to the appendix.
- [ ] **Step 5: Results.** Add 5.10 "Gated distillation" (T14) and extend 5.3 with the mixed-precision result (T15); add RescueNet columns/rows (T16) to 5.8 and 5.9. Move Section 5.7 (Sen1Floods11, T6) and T13 to `10_supplementary.tex` with one summary sentence each in the main text.
- [ ] **Step 6: Discussion and conclusions.** Rewrite conclusions as the four rules with one number each; discussion 6.1 and 6.3 lose the "earlier draft"/"we had attributed" passages (state the finding once); Table 1 gains a "why included" column; one INT8-loss range (fold means) everywhere.
- [ ] **Step 7: Abstract.** Five moves, at most eight numbers, move 3 names the two methods. Check `python $S/plain_abstract.py` word count <= 250 (JRTIP has no hard limit, but keep it).
- [ ] **Step 8: Word count and build.** `texcount -merge main.tex` <= 10,500 words in the main text. `latexmk` build, 0 undefined. Commit: `docs(manuscript): restructure around gaps G1-G4 and methods; supplementary split`.

---

### Task 10: Quality passes

**Files:**
- Modify: manuscript sections as flagged

- [ ] **Step 1: Humanizer detect** (`humanizer`, academic profile) on all `sections/*.tex`; fix P0/P1 flags.
- [ ] **Step 2: Q1-anatomy audit** (`q1-paper-anatomy`, audit mode) on the rebuilt PDF; every checklist row must be Pass or a justified N/A; fix remaining findings.
- [ ] **Step 3: Number audit:** `scripts/make_tables.py --check` (add this flag: re-derive every table from JSON and diff against the committed `.tex`). Expected: no diff.
- [ ] **Step 4: Commit:** `docs(manuscript): humanizer and anatomy passes`.

---

### Task 11: Retarget to the Journal of Real-Time Image Processing

**Files:**
- Create: `manuscript/jrtip/main_jrtip.tex` (Springer `svjour3` wrapper that `\input`s the same `sections/` and `tables/`)
- Modify: `manuscript/cover_letter.tex`, `manuscript/SUBMISSION_CHECKLIST.md`, `submission_package/README.md`
- Delete from the package: IJDRR highlights and declaration (keep in git history)

- [ ] **Step 1: Verify guidelines** on the JRTIP "Submission guidelines" page (template, length, figure format, blind or not, OA options). Record them in `SUBMISSION_CHECKLIST.md` with the date checked. Do not assume IJDRR's rules carry over.
- [ ] **Step 2: Wrapper.** `main_jrtip.tex` loads `svjour3` with the journal option, defines the same macros as `main.tex` (`\teacherIoU` etc. via a shared `macros.tex` extracted from `main.tex`), and inputs the sections. Build with `latexmk`. Expected: 0 undefined references, figures within the text width.
- [ ] **Step 3: Cover letter** addressed to JRTIP: one paragraph on the two methods, one on the three external test sets, one disclosing the EM&S and IJDRR desk decisions, two suggested reviewers only if the user supplies names.
- [ ] **Step 4: Zenodo.** Tag `v2.0-jrtip`, mint the DOI (user action through Zenodo-GitHub integration), put the DOI in the data-availability statement.
- [ ] **Step 5: Package.** Rebuild `submission_package/` (manuscript PDF, source zip, cover letter, figures as separate files if required, declaration if required). Commit: `docs(manuscript): JRTIP submission package`.

---

### Task 12 (optional, after Task 3): Seed repeats for the headline FSSD claims

**Files:**
- Modify: `code/scripts/run_full_experiment.py` (add `--seed`, pass to `make_loaders`/torch/numpy seeding)
- Output: `code/runs/3fold/seeds_fold0.json`

- [ ] **Step 1:** Add `--seed` (default 42) and a `--models` filter. Kaggle kernel: fold 0, seeds 1 and 2, `teacher` + three no-KD students (8 runs, about 6 h). Dry run, user yes, push.
- [ ] **Step 2:** Report mean and std over seeds in a one-row supplementary table; add one sentence in Section 4.4. Commit with `-f`.

---

## Self-review notes

- Spec coverage: dataset decision (Tasks 7, 9 Step 4); gated KD (2-5, 8-9); mixed precision (6, 8-9); audit items (1, 9, 10); venue (11); budget (3, 6, 7, 12).
- Review Focus 1-5 are pinned by tests in Tasks 2, 2, 6, 7, 3.
- Names used across tasks: `"gated"` config and `{sname}_gated` keys (2, 3, 4, 5, 8); `select_fp32_groups`, `int8_mac_share` (6, 8); `make_rescuenet_loader`, `rescuenet` block (7, 8).
