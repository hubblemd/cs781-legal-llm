# Phase 3: Rhetorical Role Prediction

**Task.** Label every sentence of a judgment with one of 13 rhetorical roles: Fact, Issue, ArgumentPetitioner,
ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled,
RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None. There are two domains, Corporate Law (CL) and IT.
Metric: macro-F1.

**Best submission:** ensemble of three models → **0.4709** macro-F1 (Codabench). Best single model: Llama + LoRA, 0.4665.

## File

`CS781_Phase3_RR.ipynb`: 8 training runs plus the ensemble, in one Kaggle session.

## Setup

One model is trained on CL + IT train. The epoch is selected on the official CL + IT dev sets (about 2,100
sentences), and each domain is reported separately.

## Model

1. **Document context.** Consecutive sentences are packed into 512-token windows (separated by an end token). A new
   window starts halfway through the previous one, and predictions of overlapping windows are averaged. A
   sentence-only mode is kept for the ablation.
2. **Sentence vector.** The backbone encodes the window (bidirectional / encoder / echo, as in Phase 2), and the
   token states of each sentence are mean-pooled.
3. **Head.** A BiGRU over the window's sentence vectors, plus an embedding of the sentence's relative position in the
   document (10 buckets), then a linear layer.
4. **Imbalance.** Class weights ∝ 1/√count and label smoothing 0.1.
5. **Ensemble.** Average of the label probabilities of the three best runs on dev.

## Techniques (all from scratch; rank 16, α 32)

| Technique | Model | Learning rate | Notes |
|---|---|---|---|
| Full fine-tuning | Prefix LM | 3e-5 | baseline and KD teacher |
| LoRMA (compulsory) | Prefix LM | 3e-4 | |
| LoRA | Llama | 5e-4 | |
| rsLoRA | T5 encoder | 2e-4 | scale α/√r |
| IA³ | RetNet | 3e-3 | |
| Prompt tuning | Llama | 3e-2 | 32 soft vectors |
| Knowledge distillation (compulsory) | RetNet, 6 of 12 layers | – | loss 0.5·CE + 0.5·T²·KL, T = 2 |
| Context ablation | Prefix LM, LoRMA | 3e-4 | sentence-only |

Up to 15 epochs, patience 5, batch 8, head lr 1e-3.

## Notebook structure

Config (`EXPERIMENTS`) → setup → Phase-1 tokenizer and models (unchanged) → task code (sentence packing, windows,
BiGRU head, position embedding, macro-F1 per domain) → PEFT methods → training loop → data → experiment runner → run
all plus the ensemble.

## Results (final submissions)

| Model | Technique | Codabench | Dev | Dev CL | Dev IT | Trainable |
|---|---|---|---|---|---|---|
| Ensemble (3 models) | probability average | **0.4709** | **0.495** | 0.426 | 0.498 | – |
| Llama | LoRA | 0.4665 | 0.486 | 0.430 | 0.505 | 5.0M |
| Prefix LM | Full FT (teacher) | 0.4599 | 0.461 | 0.370 | 0.472 | 112.2M |
| Prefix LM | LoRMA | 0.4552 | 0.465 | 0.401 | 0.483 | 5.7M |
| RetNet (6 layers) | KD from Prefix LM | 0.4248 | 0.450 | 0.344 | 0.476 | 73.3M |
| RetNet | IA³ | 0.4245 | 0.445 | 0.363 | 0.485 | 2.7M |
| Llama | Prompt tuning | 0.3966 | 0.388 | 0.312 | 0.385 | 2.7M |
| T5 encoder | rsLoRA | 0.3914 | 0.405 | 0.388 | 0.372 | 3.9M |
| Prefix LM | LoRMA, sentence only (ablation) | 0.3516 | 0.313 | 0.244 | 0.341 | 3.1M |

The ensemble is Llama-LoRA + Prefix-LM-LoRMA + Prefix-LM-full. Trainable counts include the BiGRU head (2.7M).

## Observations

* **Document context matters most:** the same LoRMA model gets 0.4552 with context and 0.3516 on isolated sentences.
* **PEFT matches full fine-tuning** while training about 2% of the backbone; with about 100 training documents, fewer
  updated weights also overfit less.
* LoRMA oscillated at lr 1e-3 and became stable at 3e-4.
* The distilled RetNet keeps 92% of its teacher's score.
* Ensembling helps because the models agree on only about 65% of dev sentences.
* IT is easier than CL for almost every model. The most frequent errors are Fact / Argument / Precedent predicted as
  Ratio, and ArgumentPetitioner ↔ ArgumentRespondent.
