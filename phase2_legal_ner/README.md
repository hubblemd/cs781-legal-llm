# Phase 2: Legal Named Entity Recognition (L-NER)

**Task.** For an Indian court judgment, predict every entity as a character span `[start, end, label]` with one of 12
labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO. Metric: **strict
macro-F1**, where a prediction counts only if start, end and label all match.

**Best submission:** Prefix LM, full fine-tuning, word-vote span decoding → **0.5067** strict macro-F1 (Codabench).

## File

`CS781_Phase2_LNER.ipynb`: one Kaggle run trains all 6 experiments in sequence and writes one `submission.zip` per
experiment.

## Data and validation

Training documents come from fold1 + fold2; the test fold is hidden. No dev split is given, so **15% of the training
documents** (split by document) are held out for model selection.

## Pipeline

1. **Character offsets.** The tokenizer returns, for every token, the exact characters it covers. Punctuation runs
   such as `).` are split into single characters so that entity boundaries fall on token boundaries.
   *Alignment ceiling:* gold spans → BIO tags → spans recovers **99.1%** macro-F1, so tokenisation loses almost nothing.
2. **Long documents.** 512-token windows with stride 384; logits of overlapping windows are averaged.
3. **Two-sided context.** See the shared backbone wrapper in the root README (Prefix LM bidirectional prefix, T5
   encoder, Llama without causal mask, RetNet echo).
4. **Tag head.** The LM head is dropped and a linear layer (768 → 25 BIO tags) is put on every token.
5. **Span decoding: `word_vote`.** One tag per word (sub-word probabilities summed), then one label per span (majority
   of its words). This fixes names broken into pieces such as RESP/APP/RESP, which was the largest error type of the
   first iteration (896 of 5,143 predicted spans were ≤ 3 characters).
6. **Training.** AdamW, linear warm-up (6%) and decay, fp16, batch 8, tag-head lr 1e-3, up to 25 epochs with early
   stopping on validation strict macro-F1 (patience 5).

## Notebook structure

| Cell | Content |
|---|---|
| config | `EXPERIMENTS`, all hyperparameters, `SMOKE_TEST` |
| setup | device, Hugging Face token, data download, Phase-1 model discovery |
| Phase-1 code | `LegalBPETokenizer`, the four model classes (unchanged from Phase 1) |
| task code | JSON parsing, annotation checks, document split, tokenisation with offsets, BIO conversion, strict scorer, windows |
| methods | LoRA / LoRMA / adapter / BitFit wrappers, `NERModel` (backbone wrapper + tag head), `apply_method`, `keep_layers` (KD student) |
| training | features, collate, window-averaged prediction, `decode_word_vote`, evaluation, class weights, training loop with early stopping |
| 1. Data | loading, checks, alignment ceiling |
| 2. Runner | build → train → validate → test predictions → `submission.zip`, `metrics.json` |
| 3. Run | runs every experiment within the 11-hour budget and prints the comparison table |

## Techniques (rank 16, α 32, attention + FFN for the low-rank methods)

* **Full fine-tuning**: all weights.
* **LoRMA** (compulsory): `W' = (I + (α/r)·B A)·W`, B = 0 at the start.
* **Houlsby adapters**: bottleneck 64 after each sub-layer output.
* **Partial fine-tuning**: all 6 layers of the T5 encoder.
* **Knowledge distillation** (compulsory): teacher = Prefix LM full fine-tuning; student = RetNet with 6 of 12 layers.
  Loss = 0.5·CE + 0.5·T²·KL(student‖teacher) with T = 2.

## Results (final submissions)

| Model | Technique | Codabench | Val | Trainable |
|---|---|---|---|---|
| Prefix LM | Full FT (lr 1e-4) | **0.5067** | 0.426 | 109.6M |
| Llama | Full FT (lr 1e-4) | 0.4900 | 0.437 | 109.5M |
| Prefix LM | LoRMA (lr 1e-3) | 0.4795 | 0.437 | 3.1M |
| Llama | Houlsby adapters (lr 1e-3) | 0.4633 | 0.420 | 2.4M |
| T5 encoder | Partial (6 layers, lr 2e-4) | 0.3669 | 0.297 | 42.5M |
| RetNet (6 layers) | KD from Prefix LM | 0.3668 | 0.347 | 70.6M |

**Span-decoding ablation** (validation strict macro-F1, same logits):

| Run | Lenient BIO | Strict BIO | Word vote |
|---|---|---|---|
| Prefix LM full | 0.358 | 0.409 | **0.426** |
| Prefix LM LoRMA | 0.343 | 0.399 | **0.437** |
| Llama full | 0.353 | 0.427 | **0.437** |
| T5 partial | 0.212 | 0.259 | **0.297** |

## Observations

* The Prefix LM (pretrained with a bidirectional prefix) gives the best tagger.
* Word-vote decoding adds 0.07–0.08 validation F1 over greedy BIO at no training cost.
* LoRMA trains 3% of the parameters and keeps 95% of full fine-tuning's score (0.4795 vs 0.5067).
* The 6-layer distilled RetNet keeps 72% of the teacher's score.
* DATE and COURT reach about 0.9 F1. APP/RESP and A.COUNSEL/R.COUNSEL are confused because only the surrounding words
  (for example "for the appellant") decide the role.
* Boundary errors remain the largest error group.

## Limitations

* Llama was pretrained on 600M tokens and the other backbones on 200M, so part of the gap between models is
  pretraining, not the fine-tuning method.
* Flat BIO cannot represent nested entities; single seed; one validation split.
