# Greek Parliamentary ASR and Domain Adaptation

COE025 NLP Project 3 — Anonymous group.
Public repository: https://github.com/kaanksn1/greek-parliamentary-asr

**21 November final model status:** all seven training allocations and all
12 final test systems are complete. Every system uses all 8,679 unchanged
original test clips at immutable execution snapshot `0ff3196c`. Saved
predictions, corpus edit counts, WER/CER, speaker-bootstrap intervals and
CSV values passed independent regeneration checks. Original test audio
passed verification before and after execution. Clean-clone data/weight
verification and all 240 frozen-runtime regression tests passed.
The generated final table below is complete. Presentation delivery and
Serhan's own contribution report remain separate member-owned work.

The task is automatic transcription of Greek parliamentary speech. Our main
experiment compares matched Whisper fine-tuning on HParl alone and HParl plus
CSS10 Greek, with evaluation restricted to the original parliamentary test.
The professor approved the two-corpus custom-benchmark and domain-evaluation
plan in principle. This is not a claim that bonus points have been awarded.

## Results and model contribution

[Generated final table](results/benchmark/SUMMARY.md),
[machine-readable results](results/benchmark/results.csv),
[figures](results/benchmark/figures/) and
[presentation brief](results/PRESENTATION_BRIEF_TR.md) are produced by
`scripts/summarize_benchmark.py` from actual complete prediction/score files.
Incomplete systems remain visibly pending. Every final result uses all 8,679
unchanged test clips, identical references and the same frozen scoring policy.
No result-table number is entered by hand.

| Required family | Implemented / selected system | Group training |
|---|---|---|
| Classical | MFCC + GMM-HMM, character HMM / trigram LM | Own implementation and training |
| CTC encoder | Convolutional subsampling + BiLSTM + character CTC | Own implementation and training |
| Attention encoder-decoder | LAS-style hybrid CTC/attention | Own implementation and training |
| Transducer | RNN-T | Own implementation and training |
| wav2vec 2.0 + ASR head | `lighteternal/wav2vec2-large-xlsr-53-greek` | Published Greek fine-tuning; no additional group training |
| HuBERT + ASR head | `utter-project/mHuBERT-147` + our Greek CTC head | Own Greek CTC training; evaluated head is present |
| Whisper | `openai/whisper-small` multilingual | Standard pretrained baseline |
| Canary | `nvidia/canary-1b-v2` | Standard pretrained baseline |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` multilingual | Standard pretrained baseline |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` | Standard pretrained baseline |

The two additional rows are **our Whisper HParl-only and HParl+CSS10
fine-tuning**. They share starting weights, seed 42, effective batch 16,
learning rate, warmup and the 1,600-update allocation. Full-dev loss selected
the step-500 checkpoint in both completed runs. Equal updates imply different
corpus coverage. Other group-trained methods remain standard baseline families
and earn implementation/training credit; a powerful pretrained model alone is
not our model contribution.

Exact model IDs, full 40-character upstream revisions, local selected-weight
hashes, decoding and batch settings are in
[`configs/final_test_protocol.json`](configs/final_test_protocol.json).
The unlicensed published Greek HuBERT candidate was not selected; our evaluated
HuBERT includes a trained Greek CTC head. Do not substitute English-only
checkpoints or a bare encoder.

## Data, cleaning and split limitations

| Actual cleaned cohort | Utterances | Speakers | Hours |
|---|---:|---:|---:|
| Selected HParl training | 23,602 | 208 | 30.883408 |
| Additional CSS10 training | 1,844 | No encoded IDs; producer single reader | 4.137269 |
| Mixed training | 25,446 | 208 identified; CSS10 IDs absent | 35.020678 |
| HParl development | 7,113 | 64 | 9.059189 |
| Original frozen HParl test | 8,679 | 82 | 11.207889 |

These counts come from the committed manifests, selected-training record and
generated [data report](docs/DATA_REPORT_TR.md). CSS10 is literary read speech
and is added only to training. Its distributor declares CC0; HParl remains
CC-BY-NC-4.0. Provenance, attribution and changes are in
[`DATA_LICENSE.md`](DATA_LICENSE.md), `configs/data_source.json` and the data
reports. The root MIT license covers our code, not third-party audio/models.
CSS10 has no encoded speaker/session IDs; its entire source is training-only.
No speaker identifiers or cross-source speaker identities are invented.

The selected raw HParl cohort contains 23,615 original clips from 18 whole
sessions chosen with seed 42. Cleaning removes nine extreme transcription-speed
records and four duplicate-audio records. All original selected raw clips,
including removals, plus dev/test/CSS10 audio are actually stored in this
repository through Git LFS. Clean manifests and reference files select the
clean corpus from those original WAVs; no gain, waveform normalization or
audio rewriting is applied. Format conversion or merging alone is not claimed
as cleaning. The format-aware audit reads original PCM16/PCM32/IEEE-float32
correctly. Original CSS10 float32 audio is retained byte for byte.

HParl's **documented official session split is not speaker-disjoint**.
49 test speakers also occur in selected training, accounting for 4,782 test
clips; recording/session overlap is zero. Results must not be described as
unseen-speaker generalization. Three test sentences appear literally among
actual clean training targets. The frozen final normalization gives nine
matching test utterances/nine unique sentences. These are recomputed by
`scripts/report_benchmark_data.py` in [the selected-cohort integrity report](results/data_integrity.json),
separately from the older diagnostic policy.

The original test was frozen before cleaning and never filtered, relabeled,
resplit or augmented. Its manifest, references and original audio hashes are
rechecked at test startup and completion. Settings/checkpoints use development
only. Audit flags are review candidates, not verified new annotations; human
listening decisions remain zero. Model output is never treated as gold.

## Evaluation

The final scorer is `greek_asr/evaluation.py`. It applies one frozen policy
equally to references and hypotheses **in memory**: remove angle-bracket
annotation tags, NFC/casefold/NFC, replace Unicode punctuation with spaces,
collapse whitespace. Greek accents, digits and nonpunctuation symbols remain.
Source files stay unchanged. All IDs remain, including normalized-empty
references; predictions against them count as insertion errors.

WER measures corpus word substitutions, deletions and insertions divided by
reference words. CER additionally exposes partial spelling/inflection errors;
internal spaces count as characters. Both aggregate edit counts across the
corpus, rather than averaging utterance percentages. The metric choice follows
the [Hugging Face evaluation lesson](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).
WER may exceed 100%. A zero corpus denominator is reported as null.

Scores include S/I/D counts, empty predictions, 95% speaker-cluster bootstrap
intervals (1,000 resamples, seed 42), and a paired bootstrap for the two trained
Whisper conditions. RTFx is audio seconds divided by measured inference
seconds; device, batch, precision and timing scope are recorded. CPU GMM runs
alongside the serial GPU queue, so speed is machine dependent. One training
seed, limited allocations and overlapping voices limit interpretation.

## Reproduce

Use Python 3.12. Install Git LFS before cloning, then materialize repository data
and selected weights with `git lfs pull`. Every notebook/script is committed;
no manual Colab upload or Drive download replaces repository data.

```bash
python -m pip install -r requirements-evaluation.txt
python -m greek_asr.data verify-freeze
python -m greek_asr.data verify --selection configs/training_data_selection.json
python -m greek_asr.data audit --selection configs/training_data_selection.json
python -m greek_asr.data clean --selection configs/training_data_selection.json
```

Cleaning reproduces the committed train/dev outputs and checks locked artifacts;
it does not modify test. The source-only 99-hour train catalog is larger than
the delivered selection; use `--selection` for the actual benchmark cohort.
`docs/DATA_PIPELINE_TR.md` explains the commands and logs.

For the measured AMD RX 7800 XT runtime, use the isolated setup scripts and
exact `requirements-rocm.lock.txt` / `requirements-nemo.lock.txt` versions.
Install the evaluation extras into the Transformers environment; retain NeMo's
separate NumPy 2 runtime. Model training commands and allocations are documented
in `docs/KEMAL_BASELINES.md`, `docs/SEQUENCE_MODELS.md` and
`configs/training_budget.json`. All original training records are in
`results/training/`. Primary jobs ran at `d98abe995`; mixed Whisper at `b760cdcf`.

[The full final-benchmark guide](docs/FINAL_BENCHMARK.md) gives preparation,
dev calibration, protocol freezing and full-test commands. A fresh evaluation
uses the already frozen final settings:

```bash
python scripts/cache_benchmark_models.py
python scripts/cache_benchmark_models.py --check-only
python scripts/run_benchmark_queue.py --phase test --output outputs/final_benchmark --nemo-python NEMO_PYTHON
python scripts/summarize_benchmark.py
```

Cache preparation downloads only exact-revision upstream models, without GPU
inference. Run it with networking available before the queue, whose child
processes deliberately use offline caches. Repository audio and our trained
weights still come from Git LFS, not an external download substitute.

The integrated candidate passed **240 tests, zero skips**; the actual capture is
in `docs/reviews/final_benchmark_2026-11-21/candidate_240_tests.txt`.
Run `python -m unittest discover -s tests -v` in the full Transformers environment.
Model/data tests cover decoding, cleaning, padding/device behavior, scoring,
batch contracts and freeze guards. Resource settings were measured on 65 dev
clips spanning all 64 dev speakers plus the longest clips; raw results and
batch trials are committed. No final test was used for resource selection.
Measured training allocations, model-specific feasible training counts and
selected steps are generated in [the training summary](results/training/SUMMARY.md).
Original run bytes match the source hashes in the checkpoint catalog; see
`results/training/original_record_verification.json`.

Selected inference weights remove optimizer/scheduler/RNG resume state while
preserving every retained tensor exactly; source/export hashes and tensor
fingerprints are recorded. Audio plus selected weights occupy about 8.95 GiB,
within the user's free 10 GiB storage cap. No paid storage or compute is used.

## Repository map and authorship

| Root path | Purpose |
|---|---|
| `greek_asr/` | Original data pipeline, source-preserving audio reader, models, pretrained adapters and scorers |
| `scripts/` | Reproducible setup, selection, training orchestration, benchmark and generated-report commands |
| `configs/` | Exact model/data revisions, training allocations, acceptances and frozen final protocol |
| `data/` | Raw source metadata and LFS WAVs, clean/raw manifests, references and audit/cleaning logs; see `data/README.md` |
| `checkpoints/` | Selected actual inference weights in LFS, licensing and exact export identities |
| `results/` | Original training records, dev calibration/resource trials, historical runtime smoke and generated final benchmark |
| `docs/` | Data/model/reproduction guides, teacher instructions, team handoffs and dated review evidence |
| `tests/` | Data/model/evaluation regression tests and small synthetic fixtures |
| `contributions/` | Each member's own report; its README lists required names and remaining member-owned delivery |
| `LICENSE` | MIT license for our code |
| `DATA_LICENSE.md` | Dataset authors, distribution licenses, attribution and actual changes |
| `requirements.txt` | Pinned common data/evaluation dependencies via the evaluation requirements |
| `requirements-data.txt` | Pinned lightweight data environment |
| `requirements-evaluation.txt` | Pinned final scoring/plotting dependencies for the Transformers runtime |
| `requirements-verification.txt` | Pinned JiWER/RapidFuzz verification dependencies |
| `requirements-sequence.txt` | Model author's pinned isolated development environment |
| `requirements-rocm.lock.txt` | Exact measured Transformers/Qwen/AMD model environment |
| `requirements-nemo.lock.txt` | Exact separate Canary/Parakeet environment |
| `.gitattributes` | Byte-preserving original data rules and LFS patterns for audio/selected weights |
| `.gitignore` | Excludes local environments, caches, generated bytecode and temporary run outputs |
| `AGENTS.md` | Repository editing/ownership instructions for coding assistance; not a model or benchmark method |
| `README.md` | Current entry point and reproduction/interpretation contract |

Ignored local `models/` holds exact-revision upstream caches; ignored `outputs/`
holds intermediate/resume states and execution logs. They are explained local
artifacts, not substitutes for committed data, selected weights or results.
Dated review documents describe their named historical checkouts; the final
protocol and generated final results define the current benchmark.
[A descriptive protocol erratum](docs/FINAL_PROTOCOL_ERRATA.md) corrects the
GMM LM name without changing the frozen model, weights, settings or protocol bytes.

Samed wrote the data pipeline; Kemal wrote GMM-HMM/plain CTC/wav2vec/HuBERT work;
Adnan wrote attention/RNN-T/Whisper fine-tuning. Serdar handled integration,
environments, central full runs, standard pretrained adapters and the explicitly
reassigned final evaluation/reporting. Serhan prepares the presentation and
must commit his own real contribution report from his own account. Central
execution does not transfer model/data code authorship. See `docs/TEAM.md` and
`contributions/README.md`; planned workload percentages are not measured results.
