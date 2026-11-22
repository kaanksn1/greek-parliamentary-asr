# Greek Parliamentary ASR and Domain Adaptation

COE025 NLP Project 3 — Anonymous group.
Public repository: https://github.com/kaanksn1/greek-parliamentary-asr

**Original 21 November benchmark status:** all seven training allocations and all
12 final test systems are complete. Every system uses all 8,679 unchanged
original test clips at immutable execution snapshot `0ff3196c`. Saved
predictions, corpus edit counts, WER/CER, speaker-bootstrap intervals and
CSV values passed independent regeneration checks. Original test audio
passed verification before and after execution. Clean-clone data/weight
verification and all 240 frozen-runtime regression tests passed.
The linked original table is complete. Presentation delivery and
Serhan's own contribution report remain separate member-owned work.

The task is automatic transcription of Greek parliamentary speech. Our main
experiment compares matched Whisper fine-tuning on HParl alone and HParl plus
CSS10 Greek, with evaluation restricted to the original parliamentary test.
The professor approved the two-corpus custom-benchmark and domain-evaluation
plan in principle. This is not a claim that bonus points have been awarded.

**22 November addendum:** the pre-registered longer Plain CTC and HuBERT-CTC
training conditions have completed. Their selected checkpoints, full-DEV
losses and test metrics are in the [extended run record](results/benchmark_extended/run_record.json),
with [training records](results/training_extended/) and [prediction/score files](results/benchmark_extended/).
These are additional training conditions of the existing CTC baseline
families. Original benchmark rows remain available below.

## Results and model contribution

[Original generated final table](results/benchmark/SUMMARY.md),
[machine-readable results](results/benchmark/results.csv),
[figures](results/benchmark/figures/) and
[presentation brief](results/PRESENTATION_BRIEF_TR.md) are produced by
`scripts/summarize_benchmark.py` from actual complete prediction/score files.
Incomplete systems remain visibly pending. Every final result uses all 8,679
unchanged test clips, identical references and the same frozen scoring policy.
No result-table number is entered by hand: the block below is rewritten by
`scripts/summarize_benchmark.py`.

<!-- generated-results:start -->

All 8,679 frozen test clips, one scoring policy; WER/CER in %, lower is better; 95% speaker-bootstrap CI for WER. Full table with speed and batch settings: [results/benchmark/SUMMARY.md](results/benchmark/SUMMARY.md).

| System | WER % | CER % | WER 95% CI | Group training |
|---|---:|---:|---|---|
| **Standard baselines** | | | | |
| gmm_hmm | 89.15 | 49.67 | [88.52, 89.78] | own implementation |
| plain_ctc | 99.95 | 91.37 | [99.93, 99.96] | own implementation |
| attention | 93.77 | 60.96 | [91.25, 96.39] | own implementation |
| transducer | 88.60 | 56.49 | [87.92, 89.31] | own implementation |
| wav2vec | 57.81 | 20.30 | [56.36, 59.35] | none (published Greek fine-tune) |
| hubert_ctc | 99.89 | 82.18 | [99.86, 99.91] | own CTC head training |
| whisper_pretrained | 72.08 | 48.42 | [66.96, 78.21] | none |
| canary | 30.18 | 11.70 | [29.50, 30.87] | none |
| parakeet | 37.30 | 15.04 | [36.46, 38.18] | none |
| qwen | 48.23 | 22.54 | [46.72, 49.86] | none |
| **Registered extended training (follow-up rows)** | | | | |
| hubert_ctc_extended | 30.78 | 9.54 | [29.16, 32.64] | own training, longer registered schedule |
| plain_ctc_extended | 100.69 | 78.74 | [100.26, 101.13] | own training, longer registered schedule |
| **Our contribution: Whisper-small fine-tuning** | | | | |
| whisper_hparl | 23.32 | 10.00 | [21.67, 25.18] | HParl only |
| whisper_mixed | 22.47 | 9.08 | [21.10, 24.11] | HParl + CSS10 |

<!-- generated-results:end -->

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

**The original CTC runs failed to learn usable transcriptions.** Their
recorded outputs, training allocations and full-DEV histories remain in
`results/benchmark/` and `results/training/`. Standard greedy collapse and
passing software tests do not establish successful ASR learning. Training
duration and objective loss alone do not identify the cause of a failure.

The [registered extended-training protocol](configs/extended_ctc_protocol.json)
fixed the longer schedules before either new training or new test pass.
It retained the data, model families, seed, base learning rate and decoder.
Checkpoint selection used the lowest scheduled full-DEV loss. HuBERT
improved substantially; **extended Plain CTC still produces degenerate
repeated-character output and remains a failed ASR training condition**.
Both outcomes are preserved in the extended run record, independently of
the original rows. The exact Plain CTC failure mechanism is unresolved.

The original test results were already known when this addendum was
registered. These are follow-up conditions, each with one recorded pass
on the same frozen test; they are not an independent unseen-test
confirmation. No old prediction, score or selected weight was replaced.

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

For code, saved predictions, result tables, figures or presentation review,
clone without automatically downloading large LFS payloads:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kaanksn1/greek-parliamentary-asr.git
```

In PowerShell, set `$env:GIT_LFS_SKIP_SMUDGE="1"` before `git clone`, then
remove the temporary variable with `Remove-Item Env:GIT_LFS_SKIP_SMUDGE`.
This review checkout contains LFS pointers until explicitly materialized.
All original audio and selected weights remain uploaded in this repository.
From that checkout, fetch the payloads needed for actual execution:

```bash
# Audio only; selected model weights can be fetched separately.
git lfs pull --include="data/raw/**" --exclude=""
# Example: only the selected mixed-corpus Whisper weights.
git lfs pull --include="checkpoints/whisper_mixed/**" --exclude=""
# Or materialize all audio and all selected weights for full reproduction.
git lfs pull --include="" --exclude=""
```

Each fresh LFS download uses the owner's monthly bandwidth; reuse a populated
checkout for repeated runs. A [bounded real anonymous download check](docs/reviews/final_benchmark_2026-11-21/small_public_download.json)
retrieved and verified a training WAV and the small selected GMM checkpoint.
It confirms current payload access, not the remaining monthly allowance.
The private account billing CSV is not published.

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
fingerprints are recorded. The original audio and original selected weights occupy about 8.95 GiB;
the addendum contributes about 0.39 GiB of new selected weights. The
combined payload remains within the free 10 GiB storage cap, as checked
in the [integration review](docs/reviews/pr17_2026-11-22/validation.json).
No paid storage or compute is used.

## Presentation

`presentation/greek_parliamentary_asr.pptx` is the 2-minute deck (speaker notes
included; `greek_parliamentary_asr.pdf` is its PowerPoint PDF export). The
appendix with the full 14-row table, metric definitions, split report and
reproduction commands is `presentation/appendix.pdf`. Both are generated from
the committed result files, so no slide number is typed by hand:

```bash
python presentation/build_presentation.py
```

The deck uses Bricolage Grotesque, Figtree and DM Mono (SIL Open Font License,
`presentation/fonts/`); install them before opening the `.pptx`, or use the PDF.
Greek text uses Arial because these fonts have no Greek glyphs.

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
| `presentation/` | 2-minute deck (`.pptx` with speaker notes, `.pdf` export), appendix PDF and the script that builds both from committed results; vendored OFL fonts |
| `LICENSE` | MIT license for our code |
| `DATA_LICENSE.md` | Dataset authors, distribution licenses, attribution and actual changes |
| `requirements.txt` | Pinned common data/evaluation dependencies via the evaluation requirements |
| `requirements-data.txt` | Pinned lightweight data environment |
| `requirements-evaluation.txt` | Pinned final scoring/plotting dependencies for the Transformers runtime |
| `requirements-verification.txt` | Pinned JiWER/RapidFuzz verification dependencies |
| `requirements-sequence.txt` | Model author's pinned isolated development environment |
| `requirements-rocm.lock.txt` | Exact measured Transformers/Qwen/AMD model environment |
| `requirements-nemo.lock.txt` | Exact separate Canary/Parakeet environment |
| `requirements-presentation.txt` | Pinned isolated environment for `presentation/build_presentation.py` |
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

[The script and document map](docs/REPOSITORY_MAP.md) explains each committed
script and separates current guides from historical planning/pilot records.
[The teacher-instruction audit](docs/TEACHER_FINAL_AUDIT_TR.md) records verified
requirements and remaining member/submission/access items; it is not a grade guarantee.

Samed wrote the data pipeline; Kemal wrote GMM-HMM/plain CTC/wav2vec/HuBERT work;
Adnan wrote attention/RNN-T/Whisper fine-tuning. Serdar handled integration,
environments, central full runs, standard pretrained adapters and the explicitly
reassigned final evaluation/reporting. Serhan prepares the presentation and
must commit his own real contribution report from his own account. Central
execution does not transfer model/data code authorship. See `docs/TEAM.md` and
`contributions/README.md`; planned workload percentages are not measured results.
