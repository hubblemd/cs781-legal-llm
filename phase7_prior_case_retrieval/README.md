# Phase 7: Prior Case Retrieval (PCR)

**Task.** For a query judgment, return exactly 7 unique candidate case IDs from the split's own candidate pool, ranked
by relevance (never the query itself). Metric: micro-F1@7 (with P@7 and R@7).

**Best submission:** Llama full fine-tuning dual encoder fused with BM25 (set B) → **0.4335** micro-F1@7 (Codabench).

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

**Codabench (hidden test, micro-F1@7):**

| Set | Submission | Codabench |
|---|---|---|
| B | **Llama full FT + BM25** | **0.4335** |
| B | Llama LoRMA + BM25 | 0.4294 |
| B | RetNet KD (teacher Llama full FT) + BM25 | 0.4288 |
| B | BM25 only (k1 4.0, b 1.0, uni+bi+trigrams) | 0.4282 |
| B | Ensemble (BM25 + LoRMA) | 0.4265 |
| A | Prefix LM full FT + BM25 | 0.3810 |
| A | Llama IA³ + BM25 | 0.3739 |
| A | BM25 only (k1 2.0, b 0.9, uni+bigrams) | 0.3733 |

**Set B (dev micro-F1@7):**

| Run | Before FT (dense) | Dense | Fused | w | P@7 | R@7 | Trainable | Train (min) | GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|
| BM25 only | – | – | 0.4480 | – | 0.488 | 0.414 | – | – | – |
| Llama LoRMA | 0.260 | 0.310 | **0.4503** | 0.4 | 0.490 | 0.416 | 3.25M | 122 | 2.4 |
| Llama full FT (teacher) | 0.260 | 0.302 | 0.4480 | 0.2 | 0.488 | 0.414 | 109.7M | 121 | 3.1 |
| RetNet KD (6 layers) | 0.219 | 0.207 | 0.4447 | 0.2 | 0.484 | 0.411 | 70.8M | 102 (2 epochs) | 5.5 |
| Llama IA³ (4 epochs) | 0.260 | 0.309 | 0.4447 | 0.2 | 0.484 | 0.411 | 0.24M | 119 | 11.5 |

**Set A (dev micro-F1@7):**

| Run | Before FT (dense) | Dense | Fused | w | Trainable |
|---|---|---|---|---|---|
| BM25 only | – | – | 0.3658 | – | – |
| Prefix LM, full FT | 0.238 | 0.258 | 0.3791 | 0.4 | 109.7M |
| Llama, IA³ | 0.260 | 0.311 | 0.3947 | 0.7 | 0.24M |
| RetNet KD / Prefix LM LoRMA | out of GPU memory (fixed in set B) | | | | |

## Observations

* **Better BM25 gave the largest gain.** Trigrams with stronger length normalisation (k1 4.0, b 1.0, binary query
  weights) raised BM25 alone from 0.3733 to 0.4282 on Codabench (dev 0.366 → 0.448). Legal citations and statutory
  phrases are multi-word, so exact three-word phrases are a strong relevance signal.
* **The Phase-1 dense models help on top of BM25, but only a little.** Every fused set-B submission except the
  ensemble beats BM25 alone on Codabench (+0.0006 to +0.0053). Alone, the dense models stay far below BM25 (dev about
  0.31 vs 0.448): legal relevance here depends on exact citations, which a 110M model trained on few queries does not
  capture as well as lexical overlap.
* **Fine-tuning improves the dense model:** dev dense F1 0.260 → 0.302–0.310 for all Llama methods. LoRMA (3% of the
  parameters), IA³ (0.2%) and full fine-tuning end within 0.01 of each other.
* **Memory fix worked:** with gradient checkpointing, LoRMA needed 2.4 GB and full fine-tuning 3.1 GB (set A's full FT:
  13.6 GB), and KD (5.5 GB) and LoRMA, which crashed in set A, both finished.
* **KD:** the 6-layer RetNet student is the weakest dense model (0.207 dev; only 2 epochs fitted in its time cap), but
  fused with BM25 it still scores 0.4288 on Codabench, close to its teacher.
* **The dev-tuned ensemble did not help** (0.4265). Its weights collapsed to BM25 + LoRMA, and with a small dev set the
  weight choice overfits; a single fused model was more reliable.
* Dev and Codabench rank the set-B runs differently (dev: LoRMA best; test: full FT best). Differences of 0.005 are
  within the noise of the small dev set.
