# Kemal's baselines: MFCC + GMM-HMM, plain CTC, wav2vec 2.0, HuBERT

> **22 November integration clarification:** the analysis below describes the original four conditions, before the separately registered [extended run](../results/benchmark_extended/run_record.json). It does not rank all group-trained checkpoints; the Whisper fine-tuning conditions are separate. Extended HuBERT improved, while extended Plain CTC remains degenerate, so duration alone is not an established explanation of both original failures. Single-system intervals are marginal speaker-cluster intervals; only the matched Whisper difference has an explicitly paired interval. Overlapping confidence intervals do not establish statistical equivalence. Read the dated comparisons with these limits and the [current README](../README.md).

> **21 November completion note:** all selected data delivery and seven training allocations are complete; all twelve final systems were evaluated. Current evidence: [training](../results/training/SUMMARY.md), [benchmark](../results/benchmark/SUMMARY.md), [reproduction guide](FINAL_BENCHMARK.md). Earlier pilot-stage pending statements below describe their dated review stage, not current delivery.

Owner: Kemal Arif Karabacıoğlu (`kemalarif`), branch `feature/p3-ctc-hmm`
(`docs/TEAM.md`, person 3). This document describes code, configs and small
checks that exist in the repository. The sections after "Results and analysis"
were written on 19 November, before any full run; the central training and the
frozen-test benchmark were completed on 21 November on Serdar's RX 7800 XT /
32 GB machine (not by me), and are analysed below.

## Results and analysis (21 November, frozen full test)

Source: `results/benchmark/*/score.json` and `results/training/*/run.json`,
produced centrally by the common scorer on all 8,679 frozen test clips (82
speaker/recording clusters). I only read these files; no number below was
computed on test by me or tuned from test.

| System (provenance) | WER % | CER % | WER 95% CI | CER 95% CI | Word S / D / I |
|---|---:|---:|---|---|---|
| wav2vec 2.0, published `lighteternal` checkpoint | 57.81 | 20.30 | 56.36–59.35 | 19.31–21.39 | 41,511 / 7,625 / 4,489 |
| MFCC + GMM-HMM, our implementation + training | 89.15 | 49.67 | 88.52–89.78 | 48.58–50.74 | 58,121 / 22,341 / 2,232 |
| HuBERT-CTC, our CTC training of mHuBERT-147 | 99.89 | 82.18 | 99.86–99.91 | 81.70–82.64 | 12,558 / 80,090 / 3 |
| Plain CTC, our training from scratch | 99.95 | 91.37 | 99.93–99.96 | 91.27–91.50 | 8,629 / 84,077 / 0 |

CIs are the scorer's paired speaker-bootstrap intervals (1,000 resamples). One
training seed per system; they do not cover seed variation.

**Training that produced these models** (`results/training/SUMMARY.md`): about
23,600 selected HParl train clips (30.9 h, the owner's measured data/storage
budget, not the 99.5 h manifest). GMM-HMM: 25 EM iterations on CPU, 3.15 h, two
items excluded as too short for their transcripts. Plain CTC: 2,300 updates at
batch 32 (about 3 epochs), 11 min on GPU, 70 items excluded by the CTC
feasibility check. HuBERT-CTC: 1,500 updates at effective batch 32 (about 2
epochs), 49 min, one item excluded.

**What the numbers say**

1. **The published wav2vec is my best system** and the only one of the four that
   produces mostly readable Greek. Its errors are mainly substitutions of
   near-homophone spellings (e.g. `σκόπευα` → `σκόπεβα`, `σήμερα` → `σίμερα`,
   `ο`/`ω`, `ι`/`η`), which hurt WER much more than CER (57.8 vs 20.3). It was
   fine-tuned on read speech (Common Voice, CSS10), not parliament speech.
2. **GMM-HMM is the strongest of our own trained models.** Its CER (49.67) is
   lower than the attention (60.96) and RNN-T (56.49) baselines with
   non-overlapping speaker-bootstrap CIs, and its WER is close to RNN-T (88.60;
   CIs overlap). Against the off-the-shelf Whisper-small (CER 48.42, CI
   44.47–52.97) the CIs overlap, so it is comparable, not better. Its training
   log-likelihood rose from −108.76 (iteration 1) to −100.50 per frame (iteration
   25) and changed by only 0.11 over iterations 18–25, i.e. it
   converged within this recipe. The grapheme-level decoder
   without a word lexicon explains the high WER: most words contain at least one
   wrong letter even when the CER is about 50%.
3. **Both CTC models are undertrained, not broken.** Their dev CTC loss was still
   falling at the last allowed update and the best checkpoint was the last one
   (plain CTC 3.19 → 3.04 → 3.03; HuBERT-CTC 10.50 → 3.15 → 2.52). Plain CTC
   emits exactly one "word" per clip (8,679 hypothesis words for 8,679 clips,
   e.g. `ταααααααααααα`), so 84,077 of its 92,706 word errors are deletions: at
   this budget it learned the character prior but not word boundaries. HuBERT-CTC
   has 95 empty outputs and 80,090 deletions. Both need many more updates than the
   2–3 epochs that fit the shared deadline; from-scratch CTC on about 31 h is
   also known to need much more data or pretraining.
4. **A few references look much shorter than their audio.** The median test
   reference rate is 14.1 characters per second, but 5 clips have under 4 and
   45 under 6. The highest-error examples of wav2vec and GMM-HMM are such clips:
   `hparl:anastasia_xristodoulopoulou-olomeleia_22_01545000_01548000_3_1` is
   6.5 s with the reference `διαστάσεις`, and both models transcribe a whole
   sentence after it (insertions). This is a possible reference/audio mismatch
   in the source data; following the professor's rule the frozen test was not
   edited, and these clips stay in every system's score.

Owner review, 19 November: all 164 tests passed independently in the existing
ROCm runtime. GMM-HMM CPU, plain CTC GPU, real mHuBERT-147 + CTC GPU training
and published wav2vec GPU inference passed three-train-clip diagnostics.
This is not representative budget measurement or full training. The owner
selected mHuBERT-147 + our CTC training as the benchmark's HuBERT path; the
published candidate without a license/card is not selected. Exact-commit
evidence and limits are in `docs/reviews/KEMAL_PR9_REVIEW_TR.md`.

| Family | Implementation | Provenance | Ready | Pending (central) |
|---|---|---|---|---|
| MFCC + GMM-HMM | `greek_asr/models/gmm_hmm.py` | our implementation + training of a standard method | train, dev tuning, predict, 13 tests, pilot train→predict | budget measurement, full train, dev tuning, dev/test predictions |
| Plain CTC | `greek_asr/models/ctc.py` | our implementation + training from scratch | train, predict, 20 tests (shared CTC helpers and config checks included), pilot train→predict | `max_steps` from a measured pilot, full train, predictions |
| wav2vec 2.0 + CTC head | `greek_asr/models/wav2vec.py` via `greek_asr.infer --backend wav2vec` | published Greek checkpoint, no training by us | real checkpoint transcribed the 3 train-pilot clips (CPU) | GPU dev/test inference |
| HuBERT + CTC head (published) | `greek_asr/models/hubert.py` via `greek_asr.infer --backend hubert` | published checkpoint, no training by us | historical CPU diagnostics retained | not selected for the benchmark: no license/model card/training-data statement |
| HuBERT + CTC head (own training) | `greek_asr/models/hubert.py train/predict` | our CTC fine-tuning of `utter-project/mHuBERT-147` | train, predict, tests with a tiny local encoder | budget measurement, full train, predictions; report as our training |

## Environment

No new dependency and no native toolkit are added. Everything runs in the
existing isolated Transformers runtime (`requirements-rocm.lock.txt`,
`scripts/setup_transformers_rocm.sh`): AMD ROCm Torch 2.9.1, torchaudio 2.9.0,
Transformers 4.57.6, NumPy 1.26.4, huggingface_hub 0.36.2, Python 3.12.
Use `--device cuda` for GPU on ROCm (with `HSA_ENABLE_DXG_DETECTION=1` on WSL);
an unavailable GPU raises an error, there is no silent CPU fallback.
GMM-HMM normally runs on CPU; its tensors also run on `cuda`.

Local development used macOS (Apple Silicon), Python 3.14, CPU Torch 2.9.0,
torchaudio 2.9.0, Transformers 4.57.6 and NumPy 2.5.5. This is not the GPU
runtime; no GPU check was possible locally and none is claimed.

```bash
P3_PYTHON="$HOME/.venvs/project3-greek-asr-transformers/bin/python"
export HSA_ENABLE_DXG_DETECTION=1
```

## Data and test boundary

All configs read the shared manifests from Samed's data PR #2:
`data/manifests/{train,dev}.jsonl` and `data/manifests/references/{train,dev}.jsonl`
(HParl-only training, official HParl session split, not speaker-disjoint).
For the HParl + CSS10 condition, point `train_manifest`/`train_references` of
a copied config to `data/manifests/train_hparl_css10.jsonl` and
`data/manifests/references/train_hparl_css10.jsonl`; CSS10 audio (22.05 kHz)
is resampled in memory only. Mixing ratio and matched budgets are Adnan/Serdar's
decision (`data/manifests/conditions.json`).

- Vocabularies, grapheme units, the character LM and CMVN statistics are learned
  from the train references/audio only.
- Dev is used for checkpoint selection (dev loss) and GMM-HMM decoder tuning.
  Dev references contain `<spoken_noise>`-style markers (the cleaned train
  targets do not); they are removed **in memory** for dev loss/tuning only. The
  final scoring normalization is the common evaluation's (Ozan), not this code's.
- `--split test` needs `--allow-test` and a committed `--test-protocol` whose
  manifest hash matches (shared `greek_asr.infer._test_protocol`). No code here
  reads test references, and no test inference was run.
- Prediction never receives reference text; every requested ID gets exactly one
  `{"id","text"}` row (genuine empty output kept). A failed run leaves
  `predictions.partial.jsonl` and `run.json` with `status: failed`.

## 1. MFCC + GMM-HMM (`greek_asr.models.gmm_hmm`)

A conventional HMM recognizer implemented directly in PyTorch instead of
Kaldi/HTK (no native install on the central machine). Recipe follows the
monophone stage of HTK/Kaldi (`train_mono.sh`) with **graphemes** as units:
Greek orthography is close to phonemic and no licensed Greek pronunciation
lexicon is part of the project data.

- **Features:** Kaldi-compatible MFCC (`torchaudio.compliance.kaldi.mfcc`, 25 ms /
  10 ms, 23 mel bins, 13 cepstra, dither 0) + Δ + ΔΔ = 39 dims; per-utterance
  cepstral mean normalization.
- **Units / lexicon:** each non-space train character is a unit, plus `<sil>`; a word's
  pronunciation is its letter sequence (grapheme lexicon). 3-state left-to-right
  HMMs with self-loops; diagonal-covariance GMM per state.
- **Training:** flat start (global mean/variance), equal alignment, then
  `num_iterations` rounds of batched Viterbi forced alignment (optional silence
  between words and at both ends) and per-state mixture EM; components double at
  `split_iterations` up to `max_gaussians_per_state`; variance floor =
  `variance_floor_fraction` × global variance; self-loop probabilities re-estimated.
  Items with fewer frames than mandatory states are reported in `run.json`
  (`excluded`) and handled by `infeasible_policy` (cap `max_excluded_fraction`).
- **LM:** character n-gram (order 2 or 3) over graphemes + space + boundary,
  Witten-Bell interpolation with an add-one unigram, train references only.
- **Decoder:** frame-synchronous Viterbi over (last n−1 LM tokens incl. current
  unit, HMM state); a word boundary is the silence unit or a zero-duration skip.
  `lm_weight` and `insertion_penalty` are selected by `tune` on a seeded random
  dev sample (`tuning.dev_utterances`, default 200).
- **Limits:** no triphones/LDA/MLLT/SAT, no speaker CMVN, no word vocabulary or
  word LM (open-vocabulary grapheme recognizer, weaker than a word-based Kaldi
  system); LM order ≤ 3 because the decoder keeps a dense state space; all train
  MFCCs are held in RAM (≈ 5.6 GB for 99.5 h).

```bash
"$P3_PYTHON" -m greek_asr.models.gmm_hmm train --config configs/kemal/gmm_hmm_measure.json --output outputs/kemal_gmm_measure --device cpu
"$P3_PYTHON" -m greek_asr.models.gmm_hmm train --config configs/kemal/gmm_hmm.json --output outputs/kemal_gmm_hparl --device cpu
"$P3_PYTHON" -m greek_asr.models.gmm_hmm tune --model outputs/kemal_gmm_hparl/model.pt --config configs/kemal/gmm_hmm.json --output outputs/kemal_gmm_tune --device cpu
"$P3_PYTHON" -m greek_asr.models.gmm_hmm predict --model outputs/kemal_gmm_hparl/model.pt --decoder-settings outputs/kemal_gmm_tune/tuning.json --manifest data/manifests/dev.jsonl --split dev --output outputs/kemal_gmm_dev --device cpu
```

The first command is a timing/memory measurement on a 2-hour speaker-grouped
subset with 4 iterations; its model is not a baseline result.

## 2. Plain CTC (`greek_asr.models.ctc`)

Standard supervised acoustic encoder + CTC head trained from scratch (Graves et
al., 2006; Deep Speech 2 style conv + BiLSTM encoder, Amodei et al., 2016;
SpecAugment, Park et al., 2019). No pretrained weights; separate from wav2vec/HuBERT.

- **Input:** 80-dim log-mel (`common.LogMelFrontend`), global mean/std from up to
  `cmvn_utterances` seeded train utterances (stored in the checkpoint).
- **Encoder:** two 3×3 stride-2 conv layers (4× time subsampling, 40 ms frames),
  padded frames zeroed after every conv layer, linear projection, 4-layer BiLSTM
  (320 per direction), dropout, linear CTC head. 9.28 M parameters with the
  default config (58 train characters + blank + `<unk>` = 60 outputs).
- **Vocabulary:** `<blank>` (id 0), `<unk>` (never a target, never printed), train characters.
- **Loss:** `torch.nn.functional.ctc_loss` with true encoder output and target lengths,
  `zero_infinity=False`; a nonfinite loss stops the run. Before training, every item
  is checked for CTC feasibility (labels + adjacent repeats ≤ output frames). On the
  current HParl train manifest, 246 of 76,302 items (0.32%) need more than 4×
  subsampling allows; they are listed in `run.json` under the default
  `exclude_and_report` policy (cap 1%).
- **Training:** AdamW, linear warmup + linear decay, gradient clipping, seeded
  length-bucketed batches, SpecAugment on train only, dev CTC loss every
  `dev_eval_every` updates; `checkpoint.best.pt` (lowest dev loss) and `checkpoint.last.pt`.
- **Decoding:** greedy best path over each utterance's valid frames only.
- **Budget (owner update, 20 November):** measured allocations and selected
  cohort paths are now in the configs and `configs/training_budget.json`.
  Plain CTC uses 2,300 updates; HuBERT 1,500 (batch 8 × accumulation 4).
  An unset budget still fails. Used RAW train delivery and corrected CSS10
  audit are pending; the budget is not final-test permission.

```bash
"$P3_PYTHON" -m greek_asr.models.ctc train --config configs/kemal/ctc_measure.json --output outputs/kemal_ctc_measure --device cuda
"$P3_PYTHON" -m greek_asr.models.ctc train --config configs/kemal/ctc.json --output outputs/kemal_ctc_hparl --device cuda --max-steps <measured>
"$P3_PYTHON" -m greek_asr.models.ctc predict --checkpoint outputs/kemal_ctc_hparl/checkpoint.best.pt --manifest data/manifests/dev.jsonl --split dev --output outputs/kemal_ctc_dev --device cuda --batch-size 16
```

## 3. wav2vec 2.0 + CTC head (published, `--backend wav2vec`)

`lighteternal/wav2vec2-large-xlsr-53-greek` at revision
`c5c82840b689b827a2029deefe82670c7c5809a0` (already in the historical
`configs/model_sources.lock.json`). Model card: language `el`, Apache-2.0,
XLSR-53 (`Wav2Vec2ForCTC`, 24 layers) with a Greek character CTC head (55 outputs,
`[PAD]`=54 blank, `|` word delimiter), fine-tuned on Common Voice Greek and CSS10
Greek, 16 kHz. CSS10 is also our training-only secondary source; HParl is not in
its declared training data. Standard inference, no training by our group:
checkpoint feature extractor → model → greedy CTC over all frames (batch 1) →
tokenizer characters. No LM. The tokenizer's added `<s>`/`</s>` lie outside the
55 logits and are never produced. Outputs can contain the checkpoint's own
apostrophe (e.g. `σ'`); normalization is the common scorer's decision.

```bash
"$P3_PYTHON" -m greek_asr.infer --manifest data/manifests/dev.jsonl --backend wav2vec --split dev --output outputs/kemal_wav2vec_dev
```

## 4. HuBERT + CTC head

### Research result

Hugging Face search on 19 November 2026 (model API: `search=hubert` with
`filter=el`, `search=hubert greek`, `filter=hubert&pipeline_tag=automatic-speech-recognition`):

| Candidate | Finding | Use |
|---|---|---|
| `ntinouldinho/hubert_greek_6gram` @ `e89272763ec7320f5cf57638dfdedc3ecaa1ce86` | `HubertForCTC` large (24 layers) with a Greek character CTC head; **no README/model card, no license, no stated training data**; bundles a 6-gram KenLM | not selected by owner on 19 November; retain historical CPU evidence only |
| `utter-project/mHuBERT-147` @ `7ad3fc0bc5106c58c9c13526abccad527150d135` | multilingual HuBERT base encoder, `el` in its 147 languages, CC-BY-NC-SA-4.0, **no ASR head** | encoder for our own CTC fine-tuning only |
| `m3hrdadfi/hubert-{base,large}-greek-speech-emotion-recognition`, `ameerazam08/hubert-base-greek-ser` | Greek emotion classifiers | rejected: not ASR |
| `facebook/hubert-large-ls960-ft`, `facebook/hubert-large-ll60k` | English ASR head / English bare encoder | rejected: wrong language |

Pins and reasons are in `configs/kemal_model_sources.json`, a new source
record; the historical `configs/model_sources.lock.json` bytes are unchanged
(a test compares its hash with the earlier runtime records).

### Published candidate (`--backend hubert`)

Historical research adapter only, **not selected for the benchmark**. It uses
the same `HFCTCBackend` path as wav2vec (`HubertForCTC`, greedy CTC, no KenLM).
The example below describes that optional adapter, not a planned benchmark run;
the selected HuBERT benchmark path is our licensed-encoder training below.

```bash
"$P3_PYTHON" -m greek_asr.infer --manifest data/manifests/dev.jsonl --backend hubert --split dev --output outputs/kemal_hubert_published_dev
```

### Own HuBERT + CTC fine-tuning (`greek_asr.models.hubert train`)

Our group's training of a standard family: label it as our training, separate
from off-the-shelf baselines (professor's rule). `utter-project/mHuBERT-147`
(95 M parameters) + a new linear CTC head over the train characters. The CNN
feature encoder is frozen; the transformer and head are fine-tuned, with the
transformer additionally frozen for the first `freeze_transformer_updates` (500)
updates. HuBERT time masking `mask_time_prob` 0.05, AdamW 5e-5, warmup 1000,
batch 8 × 4 accumulation, float32. Zero-padded raw audio without an attention
mask, as the checkpoint's preprocessor specifies; the CTC loss uses each item's
true CNN output length (20 ms frames). Same feasibility, dev-loss, checkpoint
and budget rules as plain CTC. Checkpoints are Hugging Face directories
(`checkpoint-best/`, `checkpoint-last/`) with `greek_asr_ctc.json` (vocabulary).
The encoder's [model card](https://huggingface.co/utter-project/mHuBERT-147)
declares CC-BY-NC-SA-4.0. Preserve the source attribution and model-license
terms when distributing derived weights; the repository's MIT code license
does not relicense the encoder or its derived weights.

```bash
"$P3_PYTHON" -m greek_asr.models.hubert train --config configs/kemal/hubert_ctc_measure.json --output outputs/kemal_hubert_ctc_measure --device cuda
"$P3_PYTHON" -m greek_asr.models.hubert train --config configs/kemal/hubert_ctc.json --output outputs/kemal_hubert_ctc_hparl --device cuda --max-steps <measured>
"$P3_PYTHON" -m greek_asr.models.hubert predict --checkpoint outputs/kemal_hubert_ctc_hparl/checkpoint-best --manifest data/manifests/dev.jsonl --split dev --output outputs/kemal_hubert_ctc_dev --device cuda --batch-size 8
```

## Final test commands (only after the final protocol commit)

Append to each predict/infer command: `--manifest data/manifests/test.jsonl
--split test --allow-test --test-protocol docs/FINAL_PROTOCOL.json` (GMM-HMM also
keeps `--decoder-settings`). Without a committed matching protocol, every entry
point refuses before writing outputs.

## To measure on Serdar's machine

- GMM-HMM: feature extraction time and peak RAM, seconds per iteration
  (`run.json` → `iterations`), tuning and per-utterance decode time; CPU vs `cuda`.
- CTC (`configs/kemal/ctc_measure.json`, 200 updates): `seconds_per_update_excluding_dev`,
  `dev_eval_seconds` and peak VRAM in `run.json`; one epoch is about 2,380 updates at
  batch 32. Choose `max_steps` for `ctc.json` from this.
- HuBERT own training (`configs/kemal/hubert_ctc_measure.json`, 50 full fine-tuning
  updates at batch 8 × 4): the same `run.json` fields.
- wav2vec / published HuBERT: dev inference time and peak VRAM (both large models).

Local indicative timings (macOS CPU, synthetic tensors, not the central machine):
trigram GMM-HMM decoding took 0.15 s for a 470-frame utterance (average train
length); state likelihoods for 100 k frames × 2,784 Gaussians took 0.67 s;
batched forced alignment of 160 utterances × 470 frames took 0.21 s.

## Checks actually run (local)

- `python -m unittest discover -s tests -v`: 164 tests, all passed, 0 skipped
  (local CPU environment above). Without torch installed, 65 pass and 99 skip.
  New tests: `tests/test_models_gmm_hmm.py` (13), `tests/test_models_ctc.py` (20),
  `tests/test_models_wav2vec_hubert.py` (12). They cover CTC blank/repeat collapse,
  valid-length decoding, feasibility policies, padding invariance, finite gradients
  and real parameter updates, nonfinite-loss failure, checkpoint reload, missing
  budget, missing/extra reference IDs, a dropped prediction failing the run, empty
  outputs, failed-run partial files, test-split refusal, measurement-config budgets, exact
  synthetic forced alignment/decoding, LM normalization, mixture EM/splitting,
  backend lock selection and frozen CNN parameters.
- Real published checkpoints through the shared runner on the three train-pilot
  clips (CPU, no WER/CER): `results/runtime_smoke/kemal_wav2vec_train_cpu/` and
  `results/runtime_smoke/kemal_hubert_train_cpu/`.
- GMM-HMM, CTC and HuBERT-CTC train → checkpoint → predict ran on the same three
  pilot clips inside the tests with tiny settings. These only prove the code paths.
