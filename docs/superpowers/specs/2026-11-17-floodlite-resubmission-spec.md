# FloodLite resubmission: decisions (spec)

Date: 2026-11-17. Status after two desk rejections: Environmental Modelling & Software
(2026-11-09, no reason given) and IJDRR (2026-11-15, "technically correct and within scope"
but "not enough methodological novelty or sufficiently generalized research findings").

## 1. Diagnosis

The science is sound (leakage-verified test sets, 399-image paired bootstrap, one-variable PTQ
ablation, negative results reported). The paper is rejected on framing: it is a benchmark
report with findings attached. It proposes no method, states no numbered gaps, and phrases its
four generalisable findings as FSSD facts. Full audit: Q1-anatomy review of 2026-11-17
(conversation record; findings 1-16 reproduced in the plan's Task list).

## 2. Dataset decision

**Keep the combined corpus for training. Add RescueNet as a third zero-shot test set.**

The combined corpus (Kaggle `tasfianoorchowdhury/flood-combined-fsegnet`, v2, 3,009 pairs,
1.0 GB) is fit for purpose:

| Source | Pairs | Water-free | Licence | Role |
|---|---|---|---|---|
| FloodNet v1.0 train+val (Rahnemoonfar et al. 2021) | 1,825 | 856 | CDLA-Permissive | training |
| FloodNet v1.0 test | 424 | 215 | CDLA-Permissive | held-out test |
| Kaggle Flood Area Segmentation | 290 | 0 | CC0 | training + model-selection folds |
| AIFloodSense (Simantiris et al. 2026) | 94 (+376 pool) | 0 | CC BY 4.0 | external test only |

75% of it is FloodNet, the most-cited RGB flood-segmentation dataset. pHash dedup against FSSD:
0 hits. 94 FloodNet images with undocumented water were excluded (`ambiguous_label`); 19 in-set
duplicates are flagged in the manifest. All of this is already used in the paper (Sections 5.8,
5.9; Tables 9-11).

What must change is presentation, not data: the paper must name the corpus by its constituents
("FloodNet v1.0 + Flood Area Segmentation, AIFloodSense held out"), cite FloodNet as the primary
dataset, give a dataset-card table with the counts above, and describe the Kaggle upload as a
repackaging released with the code. The Kaggle handle must not appear in the blind manuscript.

RescueNet (Rahnemoonfar, Chowdhury, Murphy, Scientific Data 2023; 4,494 UAV images, 11 classes
including `water`, Hurricane Michael 2018, CDLA-Permissive) is the "famous bigger" dataset that
fits: same imaging regime as FloodNet but a different hurricane, state and year, with a water
class and dry scenes. Using its official test split (450 images, Kaggle mirror
`royzhao2004/rescuenet-test`, 2.4 GB) as a zero-shot test set costs inference only and makes every
claim (false alarms, per-channel repair, KD inheritance, gated KD) tested on three independent
external sets. Training on RescueNet is out of scope (23 GB, would re-run every experiment).

## 3. New methods (turn two diagnoses into fixes)

### 3.1 Ground-truth-gated distillation ("gated KD")

Finding it answers: on the combined corpus, combined KD raises flood IoU by +0.026 to +0.067
but raises the false-alarm rate by +0.048 to +0.062, concentrated on scenes where the teacher
itself false-alarms (difference-in-differences +0.14 to +0.19).

Method: during training the ground truth is known, so distil only where the teacher is right.
Let `c = 1[(sigmoid(z_T) >= 0.5) == y]` per pixel. Response term: the temperature-scaled BCE is
averaged over pixels with `c = 1` only. Feature term: the MSE at decoder block 2 is weighted by
`c` downsampled (area-average) to the feature resolution. Task term unchanged. On a water-free
image where the teacher predicts water, every predicted-water pixel has `c = 0`, so the student
receives no signal to imitate the false alarm. Hyperparameters alpha=0.5, beta=0.1, T=4 unchanged
so the only difference from "comb" is the gate.

Hypothesis to test (pre-registered here): gated KD keeps at least half of the comb-KD flood-IoU
gain on the FloodNet test split and shows no significant FAR increase over no-KD (95% bootstrap
CI on the FAR difference includes 0) and a DiD CI that includes 0.

Runs: 3 students x 3 folds on the combined corpus, gated configuration only (the teacher
checkpoints already exist). Per-channel INT8 of the 9 gated checkpoints. Zero-shot on
AIFloodSense, FloodNet test, RescueNet test.

### 3.2 Sensitivity-guided mixed-precision PTQ

Finding it answers: per-channel scales repair every CNN but MobileViT-XXS keeps a 0.22-0.33 IoU
loss located in its stem and stages 0-1; keeping those three groups FP32 leaves -0.024/-0.087/-0.187,
fold-dependent.

Method: a greedy recipe on top of per-channel static QDQ. (1) Measure each layer group's
sensitivity `s_g` = IoU drop on the calibration images when only group g is quantized.
(2) Sort groups by `s_g` descending; move groups to FP32 one at a time until the loss of the
resulting mixed artifact on the calibration images is <= tau (tau = 0.02 IoU) or every group is
FP32. (3) Report the artifact's IoU on the validation fold and the zero-shot sets, its size, the
share of Conv/MatMul MACs left in INT8, and its latency on both CPUs. Selection uses calibration
images only, never the validation fold.

Applied to all four FSSD networks and the combined-trained MobileViT-XXS (which collapses in two
of three folds under plain per-channel). Expected: the three CNNs select zero groups (so the
recipe reduces to per-channel and costs nothing), MobileViT-XXS selects its early groups.

## 4. Manuscript changes (from the audit)

Structural (decide the desk): new title in [Method] for [Task] with [Distinguishing feature]
form; numbered gaps at the end of the introduction; contributions as one-sentence verb-first
bullets each tied to a table; delete Section 1.2 "Scope and pivot"; state the four general rules
once each and make them the spine of abstract, contributions and conclusions; add a related-work
positioning table and a closing synthesis.

Rigor: fix the code/paper hyperparameter mismatch (code: teacher lr 1e-4, students 5e-4, batch 8,
35 epochs); state that best-epoch checkpoints are selected on the reported fold and that the
external sets are untouched; justify alpha, beta, T and decoder block 2; add a "why included"
column to Table 1; use one INT8-loss range everywhere (fold means).

Presentation: remove the six lab-notebook passages; cut to about 10,000 words (Table 6, Table 13
and the Sen1Floods11 subsection to supplementary); abstract with eight numbers and a "what is
new" move; mint the Zenodo DOI before submission.

## 5. Venue

Journal of Real-Time Image Processing (Springer, hybrid). Deployment latency, quantization and
distillation are its core topics. Verify the current author guidelines (template, length limit,
double-blind or not) before formatting; do not assume.

Fallback: the Ecological Informatics transfer stays open until 2027-02-08.

## 6. Out of scope

Quantization-aware training; new backbones; energy measurement; training on RescueNet;
Raspberry Pi / Jetson timing. Each stays in Limitations as future work.

## 7. Compute budget

Kaggle T4: gated KD about 12-15 h (3 students x 3 folds, ~1.5 h each); optional seed repeats
~4 h. Local CPU: mixed-precision search ~4 h (group sensitivities are already computed for
MobileViT fold 0); RescueNet inference ~1 h; quantization of 9 checkpoints ~1 h.
