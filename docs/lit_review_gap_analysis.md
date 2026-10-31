# Literature Review — Gap Analysis & Claim Verification (2026-10-31)

Narrative, gap-filling review for the FloodLite manuscript (not a systematic review).
Every entry below was verified against a primary source (publisher page, arXiv abstract,
or Crossref record) before being added to `manuscript/refs.bib`.

---

## 1. Claim verdicts (manuscript statements checked against the literature)

| # | Manuscript claim (pre-revision) | Verdict | Action taken |
|---|---|---|---|
| 1 | "every paper reports IoU …; **none reports per-platform inference latency**" (abstract, §1, §2.1) | **FALSE.** Hernández et al. (Remote Sens. 14:223, 2022) benchmark PSPNet/DeepLabV3/U-Net inference times on Jetson Nano, TX2, and AGX Xavier (e.g. ~0.29 s GPU inference; largest models fail to run on the Nano; ~14 s CPU baseline). | Reworded in abstract, §1 ¶3, §2.1: the defensible gap is **commodity CPU / browser latency + post-quantization accuracy + PTQ stability**, which no prior work reports. |
| 2 | "To our knowledge, **no prior work has applied KD to flood segmentation**" (§2.2) | **FALSE as of 2026-09-17.** Schmalstieg, Mueller & Samek, arXiv:2609.20441 (submitted to IEEE JSTARS): Prithvi-EO-2.0 (~300M) → EfficientViT-B0 (~0.7M), QAT INT8 TensorRT on Jetson Xavier NX, 5.57 ms @512², Sen1Floods11/STURM-Flood/WorldFloods-v2. Concurrent work (FloodLite draft: May 2026). | §2.2 now cites it as concurrent work and differentiates: modality (satellite MSI vs. RGB), KD goal (label-budget expansion vs. response/feature transfer), quantization route (QAT-GPU vs. PTQ-CPU). The narrower claim — no work ablates classical response/feature KD for flood segmentation — stands. |
| 3 | "First systematic **multi-platform** edge-deployment benchmark" (title of contribution 1, conclusions) | **Overbroad.** Besides #1 and #2: Nie et al. (Remote Sens. 18:1418, 2026) do real-time CPU-only SAR flood segmentation on satellite terminals; Bhattacharjya et al. (arXiv:2608.15410) benchmark VLM flood reasoning-segmentation latency/energy on Jetson AGX Xavier. | Contribution 1 rescoped to "first systematic **CPU-and-browser** edge-deployment benchmark"; §2.1 now cites all three. **TODO: conclusions (§8) still says "multi-platform" — update with the teacher-latency re-benchmark pass.** |
| 4 | MobileViT PTQ failure mechanism ("rank-deficient per-channel statistics") | Mechanism as stated didn't match code (`per_channel=False`). Literature supports a corrected mechanism: FQ-ViT (IJCAI 2022) identifies **severe inter-channel variation at LayerNorm inputs** and **extremely non-uniform attention distributions** as the two PTQ killers in transformers; RepQ-ViT (ICCV 2023) same diagnosis. | §1 contribution 3, §3.6, §5.3 rewritten to the per-tensor story with citations. |
| 5 | MobileNetV3/HardSwish PTQ fragility (our empirical finding) | **Corroborated.** FrostNet (arXiv 2020): hard-swish add/multiply ops generate extra quantization error. ML-EXray (MLSys 2022): on-device quantized MobileNetV3 can produce constant output / 0% accuracy depending on kernel. 2026 edge-accelerator study: HardSwish and self-attention architectures most vulnerable to INT8 degradation. | §2.4 now cites ML-EXray; FQ-ViT/RepQ-ViT cover the attention side. |
| 6 | KD negative-result framing | **Strengthened by:** Cho & Hariharan (ICCV 2019) — bigger/better teacher often not a better teacher; Stanton et al. (NeurIPS 2021) — fidelity and generalization diverge. Both now cited in §2.2. | Done. |

## 2. New references added to `refs.bib` (11)

| Key | Where used | Why |
|---|---|---|
| `rahnemoonfar2021floodnet` | §4 (combined dataset, planned), data availability | FloodNet original citation (IEEE Access 9:89644) — required by CDLA licence etiquette |
| `bonafilia2020sen1floods11` | §7 limitations / §99 data availability | Sen1Floods11 proper citation (CVPRW 2020) |
| `simantiris2026aifloodsense` | §4 (external zero-shot test, planned) | AIFloodSense (Remote Sens. 18(6):938, 2026; CC BY 4.0) |
| `schmalstieg2026distillation` | §1, §2.1, §2.2 | Concurrent KD+INT8 edge flood work — must be cited or a reviewer will do it for us |
| `nie2026gated` | §2.1 | Concurrent CPU real-time SAR flood segmentation |
| `bhattacharjya2026floodreasonbench` | §2.1 | Concurrent edge flood benchmark (VLM reasoning-seg) |
| `cho2019efficacy` | §2.2 | KD efficacy negative result |
| `stanton2021does` | §2.2 | KD fidelity ≠ generalization |
| `lin2022fqvit` | §1, §2.4, §5.3 | ViT PTQ failure mechanism (LayerNorm inter-channel variation, attention non-uniformity) |
| `li2023repqvit` | §2.4, §5.3 | ViT PTQ scale-reparameterization; same diagnosis |
| `qiu2022mlexray` | §2.4 | Field evidence of quantized MobileNetV3 silent collapse |

## 3. Manuscript edits applied today

- `00_abstract.tex`: "none reports latency" reworded; "no measurable accuracy loss vs teacher" → "95% of the teacher's IoU"; INT8-vs-FP32 WASM FPS conflation fixed.
- `01_introduction.tex`: ¶3 gap statement corrected (Hernández exception); contribution 1 rescoped to CPU-and-browser; contribution 3 quantization recipe corrected to per-tensor + op-type restriction, mechanism re-grounded in FQ-ViT.
- `02_related_work.tex`: Hernández described accurately; concurrent 2026 work added (Schmalstieg, Nie, FloodReasonBench); KD-negative-result citations added; PTQ paragraph extended.
- `03_methods.tex`: calibration 200 → **80 images** (matches `quantize_all.py` `n_calib_batches=10` × batch 8); quantization scheme corrected to `per_channel=False` + op-type restriction.
- `04_experimental_setup.tex`: lr corrected to 1e-4 (teacher) / 5e-4 (students); batch 8 for all; seed claim scoped to the KFold split; M2 benchmark threading described truthfully (ORT default thread pool, not single-threaded).
- `05_results.tex`: MobileViT PTQ-failure bullet rewritten to the per-tensor mechanism with citations.
- `09_appendix_reproducibility.tex`: removed reference to nonexistent `floodlite/utils.py`; seed claim honest.
- `tables/T4_quantization.tex`: caption 200 → 80 calibration images.

## 4. Still open (not fixable by literature/text alone)

1. **Teacher latency confound (T5, §5.4, §6.2, §8, abstract "15.8×")** — teacher row is PyTorch-CPU, students are ORT-CPU. Requires exporting the teacher to ONNX and re-benchmarking on the M2. All speedup claims recompute from that.
2. **§8 conclusions** still says "first systematic multi-platform" and "15.8× speedup" — update together with (1).
3. ~~**Hernández full text**~~ **RESOLVED 2026-10-31** — full PDF obtained (MDPI deploy mirror) and read. Confirmed: (a) grid = {PSPNet, UNet, DeepLabV3+} × {ResNet152, EfficientNet, MobileNet} on Jetson Nano/TX2/AGX Xavier + HPC reference, GPU *and* CPU times; (b) Nano failures are even broader than we state — DeepLabV3+ cannot run on the Nano with *any* encoder, PSPNet fails with ResNet152/EfficientNet, UNet-ResNet fails, UNet-EfficientNet runs only on Xavier (CPU ≈ 14 s/image), UNet-MobileNet runs on Nano GPU but not CPU; (c) no quantization anywhere in the paper. Every claim in our §1/§2.1 about this work is accurate.
4. Reference count now 48; target ~60 for an Elsevier Q1 (add DRR-domain and energy-measurement refs during the next writing pass).
