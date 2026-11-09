# Phase 1: Legal language model pretrained from scratch

**Task.** Given a 512-word legal prompt, generate a 128-word continuation. Metric: BERTScore-F1 (primary), ROUGE-L and
BLEU-4. Constraints: 50M–300M parameters, trained from scratch on the provided corpus only, trainable on free Kaggle GPUs.

**Final model:** Llama-style decoder, 109.5M parameters, 549.3M training tokens, peak lr 6e-4, decoded with
temperature 0.7, top-p 0.9, repetition penalty 1.2 → **BERTScore-F1 0.8365** (ROUGE-L 0.1831, BLEU-4 0.0491).

## Files

| File | Purpose |
|---|---|
| `legal_bpe.py` | `LegalBPETokenizer`: our byte-level BPE (pure Python, standard library only) |
| `CS781_Phase1_Tokenizer_OwnBPE.ipynb` | trains the tokenizer, tokenises train/validation data in parallel, saves the token cache (`uint16`) + `meta.json` |
| `CS781_Phase1_Arch1_Decoder_Llama_ownBPE.ipynb` | Architecture 1: decoder-only (Llama-style), **final model** |
| `CS781_Phase1_Arch2_EncoderDecoder_T5_ownBPE.ipynb` | Architecture 2: encoder–decoder (T5-style) |
| `CS781_Phase1_Arch3_Hybrid_PrefixLM_ownBPE.ipynb` | Architecture 3: Prefix LM (UniLM-style hybrid) |
| `CS781_Phase1_Arch4_Efficient_RetNet_ownBPE.ipynb` | Architecture 4: Retentive Network |
| `CS781_Phase1_Generate_Llama.ipynb` | generation-only notebook: loads trained weights and runs several decoding settings, one submission each |

## Data

* Corpus: `Exploration-Lab/CS-781-Capstone` (>4.5B tokens): Supreme Court, High Courts, Tribunals, Acts. No dev split
  is given, so we build our own.
* **Document-level split:** each document goes to validation with probability 0.05 until the source's quota is
  filled, so no judgment appears in both sets. Documents are shuffled per source; training batches come from a
  random permutation of 1024-token blocks each epoch (every batch mixes all sources).
* Architecture comparison: 200M train / 5M validation tokens, mix SC 0.40 / HC 0.30 / Trib 0.15 / Acts 0.15.
* **Final mix:** **SC 0.50 / HC 0.20 / Trib 0.15 / Acts 0.15**, 600M budget. Supreme Court text is only 6.9% of the
  corpus (~301M tokens); the final mix uses all of it once, giving 549.3M training tokens.

## Tokenizer (`legal_bpe.py`)

* Byte level: base vocabulary = 256 byte values, so there is no `<unk>`.
* Pre-tokenisation regex: words with their leading space, numbers in chunks of ≤3 digits, punctuation runs, whitespace.
  Merges never cross these boundaries. The split is lossless: `decode(encode(x)) == x` (tested).
* Training: count pre-tokenised words, then repeatedly merge the most frequent adjacent pair. Pair counts are updated
  incrementally and the best pair comes from a lazy max-heap.
* Encoding: merges applied by rank, results cached per word, 4 worker processes.
* Vocabulary 32,000 (256 + 31,743 merges + EOS id 31,999), trained on 100M characters (210k unique words, ~40 s of
  merging). Common legal words are single tokens (" appellant", " impugned", " IPC"). Fertility is 1.361 tokens/word
  (1.350 on the final mix), and ids fit in `uint16`.

## Architectures (d = 768, 32k vocabulary)

| # | Model | Params | Key design |
|---|---|---|---|
| 1 | Llama-style decoder | 109.5M | causal attention, RoPE, pre-norm RMSNorm, SwiGLU (2048), no biases, tied embeddings, SDPA + KV cache; 12 layers × 12 heads |
| 2 | T5-style encoder–decoder | 123.7M | 6 bidirectional encoder + 6 causal decoder blocks with cross-attention; each block split at a random P ∈ [512, 768]: encoder reads [0, P), decoder predicts [P, 1024) |
| 3 | Prefix LM (UniLM-style) | 109.5M | one stack, bidirectional inside the prefix, causal over the continuation, segment embedding; random split P ∈ [128, 768], loss on [P, 1024) |
| 4 | RetNet | 116.7M | multi-scale retention with per-head decay γ_h = 1 − 2^(−5−h); parallel form for training, recurrent O(1)-per-token form for generation; GELU FFN (3072) |

Parameter check for Arch 1: per layer 4·768² (attention) + 3·768·2048 (SwiGLU) ≈ 7.08M; ×12 = 84.9M; + tied embedding
32,000·768 = 24.6M → 109.5M (matches the measured 109,529,856).

## Training setup (shared by all four)

AdamW, β = (0.9, 0.95), weight decay 0.1, peak lr 3e-4 (final Llama: 6e-4), 300 warm-up steps then cosine decay to 10%
of the peak, gradient clipping 1.0, mixed precision. Global batch 32 × 1024 tokens; one epoch (6,103 steps for 200M
tokens; 16,762 for 549M). Kaggle 2×T4 with `nn.DataParallel`.

**Evaluation:**
* Validation loss on the full 5M-token set.
* **Continuation loss:** loss on the last 256 tokens of each block given the first 768. It is identical in all four
  notebooks and is the fair cross-architecture comparison, because architectures 2–3 only train on part of each block.
* Codabench BERTScore / ROUGE-L / BLEU-4.
* Generation diagnostics: repeated 4-grams, outputs shorter than 128 words, distinct-bigram ratio.

**Bug found and fixed:** mid-training validation used the *first* 40 validation batches, which contain only Supreme Court
text (tokens are stored source by source). It now uses evenly spaced blocks (`np.linspace`).

## Notebook structure (architecture notebooks; all four are identical except the model cells)

1. Hugging Face login.
2. GPU check.
3. Configuration.
4. Tokenizer.
5. Data download.
6. Train/validation token sets (cached).
7. 1024-token blocks.
8. Datasets.
9. Model:
   * 9b: architecture;
   * 9c: instantiation;
   * 9d: self-test that incremental (KV-cache / recurrent) decoding equals the parallel forward pass;
   * 9e: training objective.
10. `LegalLMTrainer` (AMP, warm-up + cosine, best-checkpoint selection, loss curves).
11. Training.
12. Checkpoint selection and perplexity.
13. Plots.
14. Save model.
15. (Optional) Hugging Face upload.
16–17. Test prompts.
18–19. Generation (temperature, top-p, repetition penalty, stop at 128 words).
20–22. `prediction.json` and the Codabench zip.

## Results

**Architecture comparison** (same 200M tokens, tokenizer, optimiser and decoding):

| | Llama | Prefix LM | RetNet | T5 |
|---|---|---|---|---|
| Parameters | 109.5M | 109.5M | 116.7M | 123.7M |
| Tokens/s (2×T4) | 24.5k | 24.6k | 17.4k | 35.2k |
| **Continuation loss (ppl)** | **2.259 (9.57)** | 2.388 (10.89) | 2.566 (13.01) | 2.811 (16.63) |
| BERTScore-F1 | 0.8234 | 0.8238 | **0.8270** | 0.8130 |
| Repeated 4-grams (median) | 38% | 34% | **10%** | 32% |

RetNet had the best BERTScore but much worse loss. The diagnostics show why: at near-greedy decoding (T 0.1), Llama
and the Prefix LM loop sentences. **We chose Llama**, because it is the best language model and its weakness
(repetition) can be fixed by decoding alone. The Prefix LM is the second backbone, as it is bidirectional over its input.

**Decoding experiments** (`CS781_Phase1_Generate_Llama.ipynb`, same weights, no retraining):

| Setting | Repeated 4-grams (median) | Outputs < 128 words | Distinct bigrams | BERTScore |
|---|---|---|---|---|
| T 0.1, p 0.95, rep 1.1 (baseline) | 38% | 15 | 0.184 | 0.8234 |
| **T 0.7, p 0.9, rep 1.2** | **0%** | **0** | 0.379 | **0.8333** |
| T 0.7, p 0.9, rep 1.1, no-repeat 6-gram | 0.8% | 2 | 0.346 | – |
| T 0.9, p 0.9, rep 1.2, no-repeat 6-gram | 0% | 0 | 0.479 | – |
| greedy, rep 1.2, no-repeat 6-gram | 1.6% | 7 | 0.285 | – |

**Final runs** (Llama, 549.3M tokens, final mix, best decoding):

| | lr 3e-4 | lr 6e-4 |
|---|---|---|
| Continuation loss (ppl) | 2.082 (8.02) | **2.052 (7.78)** |
| BERTScore / ROUGE-L / BLEU-4 | 0.8363 / 0.1813 / 0.0457 | **0.8365 / 0.1831 / 0.0491** |

**BERTScore progression over Phase 1:**

| Step | BERTScore-F1 |
|---|---|
| Pilot (GPT-2 tokenizer, 50M tokens) | 0.8226 |
| Own BPE, 200M tokens | 0.8234 |
| Better decoding | 0.8333 |
| 549M tokens (SC-weighted mix) | **0.8365** |

About three quarters of the gain came from decoding, one quarter from more pretraining data.

## Reproducing the final model

In the tokenizer notebook set `TRAIN_TOKENS = 600_000_000`, the final `TRAIN_MIX` (0.50 / 0.20 / 0.15 / 0.15) and
`EXISTING_TOKENIZER_DIR` (reuse the tokenizer). In `..._Arch1_Decoder_Llama_ownBPE.ipynb` set `TOKEN_CACHE_DIR` to that
cache, the same `TRAIN_TOKENS` / `TRAIN_MIX`, `LEARNING_RATE = 6e-4`, and decoding `TEMPERATURE = 0.7`, `TOP_P = 0.9`,
`REPETITION_PENALTY = 1.2`. The notebooks as committed hold the 200M-token architecture-comparison settings.
