# Phase 7: Prior Case Retrieval (PCR)

**Task.** For a query judgment, return exactly 7 unique candidate case IDs from the split's own candidate pool, ranked
by relevance (never the query itself). Metric: micro-F1@7 (with P@7 and R@7).

**Best so far:** Llama + IA³ dual encoder fused with BM25 → **0.3947** dev micro-F1@7 (set A). Set B results will be
added when its run finishes.

## Files

| Notebook | Runs |
|---|---|
| `CS781_Phase7_PCR_setA.ipynb` | Prefix LM full FT (teacher) · RetNet KD · Prefix LM LoRMA · Llama IA³ |
| `CS781_Phase7_PCR_setB.ipynb` | Llama LoRMA · Llama full FT (teacher) · RetNet KD from the Llama teacher · Llama IA³ (4 epochs) · ensemble |

## Method

1. **Lexical baseline: BM25.** Our own sparse implementation with hashed n-gram counts, so no vocabulary is kept in
   memory. k1, b, the n-gram order (unigrams + bigrams, + trigrams in set B) and the query term weighting are tuned
   on dev. BM25 is both a reference submission and the fusion partner.
2. **Dense dual encoder on a Phase-1 backbone.** Each case is cut into 8 windows of 512 tokens spread over the
   document. Each window is encoded by the backbone (bidirectional; RetNet echo) and mean-pooled. The case vector is
   the mean of its window vectors, followed by a 256-d projection and L2 normalisation. Relevance = cosine similarity.
3. **Contrastive training (InfoNCE, τ = 0.05).**
   * Each training item is (query, relevant case, hard negative).
   * Hard negatives are the query's top-20 BM25 candidates that are *not* relevant: they share wording but are not
     cited, which pushes the model past word overlap.
   * The other cases of the batch serve as in-batch negatives, and other cases relevant to the same query are masked
     out.
   * 4 random windows per case at each step.
4. **Knowledge distillation:** KL between the teacher's and the student's similarity distributions over the batch
   candidates (T = 2, α = 0.5). The student is a RetNet with 6 of 12 layers.
5. **Fusion:** score = w·z(dense) + (1−w)·z(BM25), with z-scores per query and w chosen on dev (w ≥ 0.2, so the
   Phase-1 model always contributes).
6. **Ensemble (set B):** z(BM25) + Σᵢ aᵢ·z(denseᵢ), with the weights chosen on dev by coordinate ascent. It also uses
   saved set-A score files if they are attached.

## Set A → set B

| Problem in set A | Change in set B |
|---|---|
| KD and LoRMA ran out of GPU memory: RetNet echo turns each 512-token window into 1,024 positions, LoRMA keeps extra activations, and the failed run's tensors were not released before the next run. | Gradient checkpointing (verified to give identical gradients), guaranteed GPU clean-up after every run (tested with a forced failure), `expandable_segments`, LoRMA first. |
| The best BM25 setting was at the edge of the grid (k1 = 2, b = 0.9, bigrams). | Wider grid: k1 up to 4, b up to 1.0, trigrams; fusion weight in steps of 0.05. |
| Llama gave the best dense model: before fine-tuning 0.260 vs 0.238; after, IA³ 0.311 vs full-FT PrefixLM 0.258. | Llama backbone for LoRMA, full FT (lr 1e-5) and as KD teacher. |
| IA³ was still improving at epoch 3. | IA³ gets 4 epochs. |

## Notebook structure

Config (`SETS`, `RUN_SET`) → setup → Phase-1 code → backbone wrapper + methods (with gradient checkpointing) →
retrieval core (data parsing, BM25, ranking + F1@k, document windows, `RetrModel`, contrastive and KD losses,
training) → 1. data (splits, BM25 tuning, hard negatives, BM25 reference submission, tokenisation) → 2. experiment
runner → 3. run set → 4. ensemble (set B).

## Results

**Set A (dev micro-F1@7):**

| Run | Before FT (dense) | Dense | Fused | w | P@7 | R@7 | Trainable |
|---|---|---|---|---|---|---|---|
| BM25 only (k1 2.0, b 0.9, bigrams) | – | – | 0.3658 | – | 0.398 | 0.338 | – |
| Prefix LM, full FT | 0.238 | 0.258 | 0.3791 | 0.4 | 0.413 | 0.350 | 109.7M |
| **Llama, IA³** | 0.260 | 0.311 | **0.3947** | 0.7 | 0.430 | 0.365 | 0.24M |
| RetNet, KD | out of GPU memory, rerun in set B | | | | | | |
| Prefix LM, LoRMA | out of GPU memory, rerun in set B | | | | | | |

**Set B:** *to be added.*

## Observations (set A)

* The dense models alone are below BM25, but fusing them with BM25 beats BM25 alone (+0.029 for IA³).
* IA³, with 0.24M trainable parameters, is better than full fine-tuning of the Prefix LM. With few training queries,
  updating fewer weights overfits less, and the Llama backbone starts from better representations.
