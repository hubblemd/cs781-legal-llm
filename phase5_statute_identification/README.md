# Phase 5: Legal Statute Identification (LSI)

**Task.** Given a case document, predict every statute ID that applies (multi-label classification; 100 statutes
occur in dev). Primary metric: macro-F1 over statutes. Secondary metrics: micro-F1 and subset accuracy. Every case
gets at least one valid, unique ID.

**Best submission:** Llama, layer-wise fine-tuning (gradual unfreezing) → **0.3586** macro-F1 (Codabench).

## Files

| Notebook | Runs |
|---|---|
| `CS781_Phase5_LSI_setA.ipynb` | Prefix LM full FT (teacher) · Prefix LM LoRMA · Llama adapters · RetNet KD |
| `CS781_Phase5_LSI_setB.ipynb` | Llama gradual unfreezing · T5 partial (top 4) · Llama prompt tuning · keyword-context ablation |

## Model

1. **Long documents:** the case is cut into 512-token windows, and 6 windows spread evenly over the whole document are
   kept (3,072 tokens). The ablation instead takes the 6 windows with the most citation words ("Section", "Act",
   "Article", "u/s", …).
2. **Window vectors:** backbone encoding + mean pooling + a learned embedding of the window's position.
3. **Label-wise attention** (CAML/LAAT style): each statute has its own attention over the 6 windows, so different
   statutes can look at different parts of the case. One sigmoid per statute.
4. **Imbalance:** BCE with a per-statute positive weight `w = min(30, √(#neg / #pos))`, plus oversampling of cases with
   less frequent statutes.
5. **Thresholds:** a global threshold and per-statute thresholds (for statutes with ≥ 3 dev cases) are compared by
   2-fold cross-fitting on dev; the winner is refit on all of dev.
6. **Training:** 3 epochs × 8,000 cases, 8 cases per optimiser step (2 × gradient accumulation 4), fp16; the best
   epoch on dev macro-F1 is kept.

## Techniques (from scratch)

* **Full fine-tuning** of the Prefix LM, lr 3e-5 (also the KD teacher).
* **LoRMA** on the Prefix LM, lr 3e-4.
* **Adapters** on Llama, lr 5e-4.
* **Knowledge distillation** into a 6-layer RetNet: soft-target BCE on the teacher's per-statute probabilities, T = 2,
  α = 0.5.
* **Gradual unfreezing** of Llama, lr 5e-5.
* **Partial fine-tuning** of the T5 encoder (top 4 layers), lr 1e-4.
* **Prompt tuning** on Llama (32 vectors), lr 3e-2.

## Notebook structure

Config (`SETS`, `RUN_SET`) → setup → Phase-1 code → backbone wrapper + methods → multi-label model (window selection,
label-wise attention, positive weights, rare-label sampling, threshold cross-fitting, metrics) → data → runner → run
set.

## Results (final submission per architecture)

| Model | Technique | Codabench | Dev | Dev (global thr.) | Micro-F1 | Trainable | Train (min) | GPU (GB) |
|---|---|---|---|---|---|---|---|---|
| Llama | Layer-wise (gradual) | **0.3586** | **0.377** | 0.348 | 0.466 | 109.8M | 40 | 4.6 |
| Prefix LM | Full FT (teacher) | 0.3549 | 0.368 | 0.342 | 0.460 | 109.8M | 49 | 4.9 |
| RetNet (6 layers) | KD | 0.3432 | 0.356 | 0.330 | 0.448 | 70.9M | 75 | 6.4 |
| T5 encoder | Partial (top 4) | 0.2325 | 0.244 | 0.219 | 0.333 | 28.6M | 20 | 1.7 |

Other runs, not final: adapters 0.3440, LoRMA 0.3375, prompt tuning 0.3003, keyword windows 0.3390.

## Observations

* Layer-wise unfreezing is slightly ahead of full fine-tuning; dev and Codabench rank the runs in the same order.
* Distillation keeps 97% of the teacher with 35% fewer parameters.
* Per-statute thresholds won the cross-fitting in all 8 runs (+≈0.03).
* Statutes with 50–500 training cases reach only ≈0.29 macro-F1, against ≈0.41 for frequent ones. Statute 5 is most
  often replaced by 12, 13 or 14.
* Subset accuracy is only ≈3%, while micro-F1 is ≈0.46.
* Choosing windows by citation words did not beat spreading them evenly.
