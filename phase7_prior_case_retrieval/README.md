# Phase 6: Abstractive Summarisation of Supreme Court judgments

**Task.** Generate an abstractive summary of a judgment. Primary metric: BERTScore-F1. Secondary: ROUGE-1 and ROUGE-L.
Training data: 7,030 judgments with reference summaries; 120 are held out by a hash of the document id, and 60 of
those are summarised for validation. Test: 100 judgments.

The task is long-input *and* long-output. Judgments have a median of 3,129 words and summaries a median of 639 words,
while one generation of our models fits 1,024 positions.

**Best submission:** Prefix LM + LoRA, round B → **0.8463** BERTScore-F1 (Codabench).

## Files

| Notebook | Runs |
|---|---|
| `CS781_Phase6_Summ_setA.ipynb` (round A) | Prefix LM full FT (teacher / reference) · RetNet KD · Prefix LM LoRMA · T5 Houlsby adapters |
| `CS781_Phase6_Summ_setB.ipynb` (round B) | Prefix LM LoRA · Prefix LM LoRMA v2 · Llama Pfeiffer adapters · Prefix LM prefix tuning · Llama prompt tuning · single-prompt ablation |

## Model

1. **Conditional generator** p(summary | judgment):
   * Prefix LM: judgment as the bidirectional prefix.
   * T5: judgment in the encoder.
   * Llama and RetNet: judgment and summary in one causal sequence.

   The loss is on summary tokens only. 704 judgment tokens + 288 summary tokens per generation.
2. **Segment-wise summarisation.** The judgment is cut into up to 6 consecutive segments. Each reference sentence is
   aligned to the segment it shares the most content words with, giving one (segment, partial summary) training pair
   per segment (≈24k pairs). At test time one part is generated per segment and the parts are concatenated in order,
   with repeated sentences removed.
3. **Context selection inside a segment:** the opening and closing sentences plus the most central sentences, in
   document order, up to 704 tokens.
4. **Decoding:** greedy with a KV cache (verified equal to full recomputation for all four architectures), no repeated
   trigrams, and a **per-document length budget**. Summary length is predicted from judgment length by a log-log fit
   on the training set and split over the segments (0.85–1.15 of each share).
5. **Clean-up:** leading punctuation and an unfinished last sentence of each segment are removed.

## Techniques (from scratch)

* LoRA and LoRMA (rank 16).
* Houlsby adapters (after attention and FFN) and Pfeiffer adapters (after FFN only).
* Prompt tuning (32 soft vectors initialised from token embeddings).
* Prefix tuning (per-layer key/value prefixes produced by an MLP).
* Word-level knowledge distillation: KL between teacher and student next-token distributions, T = 2, α = 0.5;
  teacher = Prefix LM full FT, student = RetNet with 6 of 12 layers.

Full fine-tuning is used only as the teacher and as a reference.

**Two rounds:**

| | Round A | Round B |
|---|---|---|
| Training | 3 epochs × 7,000 pairs | 4 epochs × 12,000 pairs |
| Segments | ≤ 4 | ≤ 6 |
| Length | fixed | per-document budget |
| Clean-up | no | yes |

Round B was designed from round A's outputs, which showed three problems: (i) the loss was still falling; (ii) every
summary had about the same length; (iii) 68% of summaries ended mid-sentence.

## Results (final submission per technique)

| Model | Technique | Round | Codabench | Val BERTScore | Val R-1 | Val R-L | Words | Trainable |
|---|---|---|---|---|---|---|---|---|
| Prefix LM | LoRA | B | **0.8463** | **0.8494** | **0.532** | **0.264** | 720 | 2.4M |
| Llama | Pfeiffer adapters | B | 0.8455 | 0.8470 | 0.525 | 0.260 | 717 | 2.4M |
| Prefix LM | LoRMA | B | 0.8452 | 0.8486 | 0.528 | 0.260 | 724 | 3.0M |
| Llama | Prompt tuning | B | 0.8391 | 0.8398 | 0.494 | 0.237 | 756 | 0.02M |
| Prefix LM | Prefix tuning | B | 0.8375 | 0.8389 | 0.484 | 0.236 | 724 | 9.9M |
| RetNet (6 layers) | KD | A | 0.8282 | 0.8324 | 0.426 | 0.224 | 570 | 70.6M |
| T5 | Houlsby adapters | A | 0.8213 | 0.8259 | 0.391 | 0.210 | 586 | 3.0M |
| Prefix LM | Full FT (reference) | A | 0.8436 | 0.8464 | 0.508 | 0.257 | 585 | 109.5M |

Reference summaries average 737 words.

## Observations

* Validation BERTScore is 0.003–0.005 above Codabench and ranks the runs in the same order, so all decisions were made
  on validation.
* LoRMA went from 0.8410 (round A) to 0.8452 (round B), showing that the recipe matters as much as the technique.
* LoRA, LoRMA and Pfeiffer adapters are within 0.001 of each other; prompt and prefix tuning are weaker and repeat
  more.
* Segment-wise generation is essential: single-prompt ablation ROUGE-1 0.375 vs 0.532.
* KD keeps 98% of the teacher's BERTScore.
* Factual errors (invented case names and citations) occur and were not measured.
