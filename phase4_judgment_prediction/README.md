# Phase 4: Court Judgment Prediction (CJPE)

**Task.** Given an Indian Supreme Court judgment, predict the outcome: 1 = appeal accepted, 0 = rejected. Metric:
macro-F1 over both classes.

**Best run:** Prefix LM, full fine-tuning → **0.732** tuned macro-F1 on the corrected dev set.

## Files

| Notebook | Runs |
|---|---|
| `CS781_Phase4_CJPE_setA.ipynb` | Prefix LM full FT (teacher) · RetNet KD · Prefix LM LoRMA · Llama adapters |
| `CS781_Phase4_CJPE_setB.ipynb` | T5 BitFit · RetNet partial (top 4) · Llama gradual unfreezing · section ablation |

The two notebooks are identical except `RUN_SET`. Each set fits one 12-hour Kaggle session.

## Data note (leak correction)

The first release of the dev file (3,505 cases) contained all 1,517 test cases with labels. The corrected dev file has
994 cases (497 per class) and no overlap with test. Our models were never trained on dev; it is only used to pick the
epoch and the threshold. Set A ran on the first release and set B on the corrected one. **All runs are compared on the
corrected 994 cases**, and test scores are not used for evaluation (instructor's guidance).

## Model

1. **Which text:** the last 2,048 tokens of the judgment, where the court's own analysis is. The ablation reads the
   first 2,048 tokens instead.
2. **Chunks:** 4 × 512 tokens. Each chunk is encoded by the backbone and mean-pooled.
3. **Chunk position:** a learned embedding of the position counted from the end of the document (right-aligned).
4. **Attention pooling** over chunks → dropout → 2-way linear layer.
5. **Imbalance and threshold:** balanced class weights. After training, the class-1 probability threshold is tuned on
   dev (0.05–0.95).
6. **Budget:** 3 epochs × 6,000 random training cases (pool up to 24,000), 2 documents per step, gradient accumulation
   to 8, fp16, one T4. The epoch with the best tuned dev macro-F1 is kept.

## Techniques (from scratch)

| Technique | Model | Learning rate | Details |
|---|---|---|---|
| Full fine-tuning | Prefix LM | 3e-5 | |
| LoRMA | Prefix LM | 3e-4 | r 16, α 32 |
| Adapters | Llama | 5e-4 | bottleneck 64 |
| Knowledge distillation | RetNet, layers 0, 2, 4, 6, 8, 10 | – | 0.5·CE + 0.5·T²·KL, T = 2 |
| BitFit | T5 encoder | 1e-3 | |
| Partial fine-tuning | RetNet | 1e-4 | top 4 layers |
| Gradual unfreezing | Llama | 5e-5 | top 1/3, then 2/3, then all layers |
| Section ablation | Prefix LM, LoRMA | 3e-4 | reads the first part |

## Notebook structure

Config (`SETS`, `RUN_SET`) → setup → Phase-1 code (unchanged) → backbone wrapper + methods → long-document model
(`select_text`, `encode_cases`, `CJPEModel`, `best_threshold`, `unfreeze_top`, training) → data → runner → run set
(summary table and uniquely named Codabench zips).

## Results (corrected dev, 994 cases)

| Set | Model | Technique | Tuned F1 | F1 @ 0.5 | Trainable | Train (min) | GPU (GB) |
|---|---|---|---|---|---|---|---|
| A | Prefix LM | Full FT (teacher) | **0.732** | **0.716** | 109.7M | 39 | 3.9 |
| A | RetNet (6 layers) | KD | 0.725 | 0.649 | 70.8M | 58 | 4.8 |
| B | Llama | Gradual unfreezing | 0.724 | 0.643 | 109.7M | 29 | 3.6 |
| A | Prefix LM | LoRMA | 0.713 | 0.697 | 3.2M | 42 | 3.1 |
| A | Llama | Adapters | 0.697 | 0.654 | 2.5M | 33 | 2.1 |
| B | RetNet | Partial (top 4) | 0.649 | 0.616 | 30.9M | 58 | 3.0 |
| B | T5 encoder | BitFit | 0.579 | 0.444 | 0.21M | 20 | 1.5 |
| B | Prefix LM | LoRMA, *first* part | 0.628 | 0.611 | 3.2M | 40 | 3.1 |

## Observations

* **The end of the judgment carries the signal:** the same LoRMA model scores 0.713 on the last part and 0.628 on the
  first part.
* **Distillation:** the 6-layer student keeps 99% of the teacher, and does much better than training only the top 4
  layers of the full RetNet.
* Gradual unfreezing is the best set-B run.
* PEFT is 0.02–0.035 below full fine-tuning, since there are thousands of long training documents; BitFit is too weak.
* Threshold tuning helps every model (+0.016 to +0.135).

## Limitations

* Only 18,000 case visits were used, and dev F1 was still rising at epoch 3.
* At most 2,048 tokens are read per judgment.
* Set A's threshold was chosen on the leaked dev file.
* Single seed.
