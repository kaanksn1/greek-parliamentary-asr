# Kemal Arif Karabacıoğlu — 220911802 — kemalarif

Person 3, owner of the HMM/CTC family (`docs/TEAM.md`): MFCC + GMM-HMM,
plain CTC, Greek wav2vec 2.0 ASR and HuBERT ASR. This report lists my own work:
the model code on branch `feature/p3-ctc-hmm` (PR #9, merged 19 November) and
the result analysis and slide draft on branch `docs/p3-kemal-results`. It does
not claim Samed's data pipeline, Adnan's sequence models, Ozan's presentation
or Serdar's integration, central training, test inference and scoring. All
full training and benchmark runs of my models were executed by Serdar on his
machine (21 November); I did not run them.

## Baseline implementations (standard methods, my implementation work)

- **MFCC + GMM-HMM** (`greek_asr/models/gmm_hmm.py`): wrote a conventional HMM
  recognizer in PyTorch instead of installing Kaldi: Kaldi-compatible MFCC + Δ +
  ΔΔ, grapheme units with a silence model, 3-state HMMs with diagonal GMMs,
  flat start, batched Viterbi forced alignment with optional silences, mixture
  EM with splitting and variance flooring, a Witten-Bell character n-gram LM from
  train text, a frame-synchronous Viterbi decoder with frameless word boundaries,
  dev tuning of LM weight and insertion penalty, and train/tune/predict CLIs.
- **Plain CTC** (`greek_asr/models/ctc.py`): conv subsampling + BiLSTM + CTC
  trained from scratch; train-only vocabulary and CMVN, SpecAugment, padding
  masking after each conv layer, CTC feasibility check with reported exclusions,
  nonfinite-loss stop, dev-loss checkpoint selection, greedy decoding over valid
  frames, train/predict CLIs.
- **Shared CTC helpers** (`greek_asr/models/ctc_utils.py`): CTC collapse and
  valid-length greedy decoding, feasibility policies, length-bucketed batches,
  in-memory annotation-tag removal for dev loss/tuning, run environment records.

## Published checkpoints (baselines, no training by our group)

- **wav2vec 2.0** (`greek_asr/models/wav2vec.py`): standard inference for
  `lighteternal/wav2vec2-large-xlsr-53-greek` at its pinned revision, connected
  to the shared runner as `--backend wav2vec`; checked the model card, tokenizer
  and blank/added-token layout; double letters separated by a blank are kept.
- **HuBERT search**: searched Hugging Face for Greek HuBERT ASR checkpoints and
  recorded the result in `configs/kemal_model_sources.json`. The only checkpoint
  with a Greek CTC head (`ntinouldinho/hubert_greek_6gram`) has no license or
  model card; I prepared its standard inference (`--backend hubert`) and left
  the use decision to the integration owner, who did not select it for the
  benchmark (`docs/reviews/KEMAL_PR9_REVIEW_TR.md`).
- Small change to the shared runner `greek_asr/infer.py`: two backend entries,
  dotted module paths, a per-backend source record and code hashes of the backend
  module. The four existing backends behave as before.

## Own training recipe (to be reported separately from baselines)

- **HuBERT + CTC head** (`greek_asr/models/hubert.py train/predict`): CTC
  fine-tuning of the multilingual `utter-project/mHuBERT-147` encoder (Greek in
  its pretraining) with a new character head; frozen CNN, optional frozen-transformer
  warm-up, true encoder output lengths in the loss, Hugging Face checkpoints.
  This is the HuBERT path used in the benchmark; Serdar ran its training
  (1,500 updates) centrally.

## Configs, tests and documentation

- `configs/kemal/`: full and small-budget measurement configs for GMM-HMM, CTC and
  HuBERT-CTC; the full CTC and HuBERT budgets stay `null` until measured.
- 45 new tests in `tests/test_models_gmm_hmm.py`, `tests/test_models_ctc.py` and
  `tests/test_models_wav2vec_hubert.py`. Locally (macOS, CPU Torch 2.9.0) the full
  suite ran 164 tests, all passed, none skipped. No GPU check was possible locally.
- Ran the real wav2vec and published HuBERT checkpoints on the three train-pilot
  clips on my CPU through the shared runner: `results/runtime_smoke/kemal_*_train_cpu/`.
- `docs/KEMAL_BASELINES.md`, plus the related rows in `README.md` and `docs/BASELINES.md`.

## Result analysis (21 November)

After the central runs I analysed the returned score, error-example and
training files of my four systems (`docs/KEMAL_BASELINES.md`, "Results and
analysis"); I only read these results, nothing was computed or tuned on test:

- wav2vec 2.0 (published): WER 57.81%, CER 20.30%, my best system; its errors are
  mostly near-homophone misspellings.
- MFCC + GMM-HMM (my implementation): WER 89.15%, CER 49.67%; lower CER than the
  attention and RNN-T baselines with non-overlapping speaker-bootstrap CIs,
  comparable to off-the-shelf Whisper-small; the training likelihood converged.
- HuBERT-CTC (99.89% / 82.18%) and plain CTC (99.95% / 91.37%) are undertrained:
  dev loss was still falling at the last allowed update; plain CTC emits one
  "word" per clip, so most errors are deletions.
- Data observation: a few test references are much shorter than their audio
  (5 of 8,679 clips under 4 characters per second); they produce the
  highest-error examples. The frozen test was not changed.
- Wrote the 2-slide draft with speaker notes for Ozan: `docs/KEMAL_SLIDES_DRAFT_TR.md`.
