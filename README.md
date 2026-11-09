# CS781 Capstone: a legal language model built from scratch, applied to seven Indian legal NLP tasks

**Course:** CS781 (LLMs), IIT Kanpur, 2026 · **Author:** Madhav Dwivedi (261110034)

A law firm cannot send privileged documents to third-party APIs, so it needs a compact, self-hosted legal
language model. This repository contains the full pipeline we built for that:

1. **Phase 1:** our own byte-level BPE tokenizer and four language-model architectures (110–124M parameters),
   pretrained from scratch on the course's Indian legal corpus. The best one (a Llama-style decoder) and the
   runner-up (a Prefix LM) are the backbones for every later phase.
2. **Phases 2–7:** six downstream legal tasks from the IL-TUR benchmark, each solved by adapting our own
   Phase-1 models with full fine-tuning and parameter-efficient methods. **Every method (LoRA, LoRMA, rsLoRA,
   IA³, adapters, BitFit, prompt and prefix tuning, partial and layer-wise fine-tuning, knowledge distillation)
   is implemented from scratch in PyTorch; no PEFT library is used.**

No pretrained weights, external data or commercial APIs are used anywhere.

## Results at a glance (final submissions)

| Phase | Task | Metric | Best submission | Score |
|---|---|---|---|---|
| 1 | Legal text continuation (128 words) | BERTScore-F1 | Llama decoder, 549M tokens, T 0.7 / top-p 0.9 / rep. 1.2 | **0.8365** |
| 2 | Legal named entity recognition | strict macro-F1 | Prefix LM, full fine-tuning, word-vote decoding | **0.5067** |
| 3 | Rhetorical role prediction | macro-F1 | Ensemble (Llama LoRA + Prefix LM LoRMA + Prefix LM full) | **0.4709** |
| 4 | Court judgment prediction | macro-F1 (corrected dev) | Prefix LM, full fine-tuning | **0.732** |
| 5 | Legal statute identification | macro-F1 | Llama, layer-wise (gradual unfreezing) | **0.3586** |
| 6 | Abstractive summarisation | BERTScore-F1 | Prefix LM + LoRA, segment-wise generation | **0.8463** |
| 7 | Prior case retrieval | micro-F1@7 | Llama full fine-tuning dual encoder fused with tuned BM25 | **0.4335** |

Scores are Codabench (hidden test) unless marked dev. Phase 4 is reported on the corrected dev set, as instructed
after the data-leak correction. Every phase README has the full
table for all submissions.

## Repository layout

```
.
├── README.md                          this file
├── requirements.txt
├── phase1_pretraining/                tokenizer, 4 architectures, generation-only decoding notebook
├── phase2_legal_ner/                  L-NER (character spans, 12 labels)
├── phase3_rhetorical_roles/           sentence-level rhetorical roles (13 labels)
├── phase4_judgment_prediction/        accepted / rejected (binary)
├── phase5_statute_identification/     multi-label statute IDs
├── phase6_summarisation/              long-input, long-output summaries
└── phase7_prior_case_retrieval/       rank 7 prior cases per query case
```

Each phase folder has its own README with the task, data, architecture, methodology, implementation (cell by
cell), hyperparameters, all results and observations. The notebooks contain code only; all explanation is in the
READMEs.

## Shared design (used by every phase)

* **Tokenizer.** `LegalBPETokenizer` (`phase1_pretraining/legal_bpe.py`): byte-level BPE written in pure Python, 32,000
  tokens (256 bytes + 31,743 merges + end-of-text), trained on 100M characters of the corpus, 1.36 tokens per
  word. Byte-level means there is no unknown token. For token tagging (Phases 2–3) it also returns the exact character
  span of every token.
* **Backbones.** The four Phase-1 checkpoints, loaded with exactly the Phase-1 model code:
  Llama-style decoder (109.5M, 549M pretraining tokens), Prefix LM (109.5M, 200M), T5-style encoder–decoder
  (123.7M, 200M), RetNet (116.7M, 200M).
* **Backbone wrapper** (`NERModel.hidden`). Gives every backbone two-sided context for understanding tasks:
  Prefix LM = whole input as the bidirectional prefix; T5 = encoder only; Llama = causal mask removed
  (bidirectional); RetNet (causal by construction) = *echo*: the window is fed twice, `[x; x]`, and the second
  copy is read, so every token has seen the whole window.
* **Adaptation methods** (`apply_method`). All implemented as small `nn.Module` wrappers around the frozen
  `nn.Linear` layers:
  * **Low-rank:** LoRA `W x + (α/r)·B A x`; rsLoRA (scale `α/√r`); LoRMA (multiplicative, `W' = (I + (α/r)·B A)·W`).
  * **Scaling:** IA³ (learned scaling of keys, values and FFN activations).
  * **Adapters:** Houlsby (after attention and FFN) and Pfeiffer (after FFN only); BitFit (biases only).
  * **Soft prompts:** prompt tuning (soft input vectors) and prefix tuning (per-layer key/value prefixes from an MLP).
  * **Layer-level:** partial fine-tuning (top-k layers) and gradual unfreezing (layer-wise).
  * **Knowledge distillation:** the student is a 6-layer RetNet keeping layers 0, 2, 4, 6, 8, 10. Teacher and student
    share one tokenizer, so different architectures can be distilled token by token.
* **Experiment runner.** Each notebook has one config cell (`EXPERIMENTS` / `SETS`) and runs all its experiments in
  sequence on one Kaggle T4 within a 12-hour session. For every run it saves `metrics.json`, `run_config.json` and a
  Codabench-ready `submission.zip`. Every notebook also has a `SMOKE_TEST` switch that runs the whole pipeline on CPU
  with tiny models and synthetic data.

## How to run

1. Kaggle notebook, **GPU T4**, Internet on.
2. Add a Kaggle secret `HF_TOKEN` (or `HF`) with access to the gated course dataset
   `Exploration-Lab/CS-781-Capstone`; data is downloaded with `huggingface_hub`.
3. Phases 2–7: attach the four Phase-1 model folders (`config.json`, `pytorch_model.bin`,
   `legal_bpe_tokenizer.json`) as Kaggle datasets; they are found automatically by their `arch` field.
4. Edit only the first code cell (experiment list), then *Run all*. Outputs go to `/kaggle/working/`.

Python packages: see `requirements.txt` (PyTorch, NumPy, scikit-learn, huggingface_hub, matplotlib, ijson,
bert-score). Seed 42 everywhere.

## Report

The LaTeX report (methods, experiments, analysis, limitations) is submitted separately on Hello IITK as
`CS781-CP-Madhav-Dwivedi-261110034-report.pdf`.

## References

IL-TUR (Joshi et al., 2024); LoRA (Hu et al., 2021); LoRMA (Bihany, Patel & Modi, 2025); rsLoRA (Kalajdzievski,
2023); IA³ (Liu et al., 2022); Houlsby adapters (2019); Pfeiffer adapters (2021); BitFit (Ben Zaken et al., 2022);
prompt tuning (Lester et al., 2021); prefix tuning (Li & Liang, 2021); knowledge distillation (Hinton et al., 2015);
RoPE (Su et al.); RMSNorm (Zhang & Sennrich); SwiGLU (Shazeer); T5 (Raffel et al.); UniLM (Dong et al.);
RetNet (Sun et al., 2023); BERTScore (Zhang et al.); BM25 (Robertson & Zaragoza).
