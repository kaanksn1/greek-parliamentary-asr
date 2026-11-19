# Greek Parliamentary ASR and Domain Adaptation

COE025 NLP Project 3. Working topic selected on 16 November 2026:
**automatic transcription of Greek parliamentary speech, with a separate
in-domain fine-tuning experiment.**

Public repository: https://github.com/kaanksn1/greek-parliamentary-asr

Owner review, 20 November: actual LFS test/dev/CSS10 bytes have been downloaded.
The frozen 8,679 test WAVs and 7,113 dev WAVs pass the data verifier; all 1,844
CSS10 audio hashes and headers match the committed source inventory. CSS10 is
**IEEE float32**, and its old signal-statistics audit incorrectly read it as
integer PCM32. The shared model input now decodes it correctly; Samed must
regenerate the float-aware audit/clean reports before full training.

The selected training pool is **18 whole HParl sessions / 30.883408 h**, or
**35.020678 h with CSS10** before that corrected audit. Its exact RAW/clean IDs,
references, hashes, overlaps and storage calculation are in
[`configs/training_data_selection.json`](configs/training_data_selection.json).
The 23,615 selected RAW train WAVs (including 13 removed records) still need
upload; the full-source train manifests remain source candidate inventories.
Active training configs point to the selected subset, rather than undelivered
99-hour train audio. All estimated audio plus checkpoint/headroom reserve fits
the user's 10 GiB storage cap; remaining account bandwidth is unverified.

Representative TRAIN measurements, generated compute budgets, package/device
identities and byte-verification evidence are in
[the measured budget report](docs/reviews/budget_2026-11-20/README.md).
Both Whisper conditions use the same measured batch/update budget. The owner
also ran the five published ASR backends on 32 TRAIN clips; these are timing
measurements, without WER/CER or test inference.
[Samed's exact remaining assignment](docs/HANDOFF_SAMED_FINAL_DATA_TR.md)
covers the encoding audit, used-data reproducibility and selected audio upload.
No full-data training, official dev selection, final test protocol or benchmark
result is complete yet. The earlier review records below describe their dated
checkouts.

The two latest professor emails are summarized in
[TEACHER_UPDATES_2026-11-18_TR.md](docs/TEACHER_UPDATES_2026-11-18_TR.md).
New splits must group related recordings by speaker/session. Documented
official splits are permitted with explicit speaker-overlap disclosure.
HParl's official session split has overlapping speakers, so it is not an
unseen-speaker evaluation. Each split needs speaker/utterance/hour counts and
literal test-sentence matches in the actual training targets. All supported
baselines share one dev-selected normalization/scoring policy.

Data review on 18 November: Samed's [PR #2](https://github.com/kaanksn1/greek-parliamentary-asr/pull/2)
at `e5aa488` passes 76 tests; its original source metadata, reporting corrections
and unchanged early test-freeze metadata are verified. Full audio delivery and
independent audio-byte verification remained pending in that reviewed checkout.
The later `4db1a4a` data submission adds LFS audio entries; their actual blobs
were not downloaded/reverified in the model review below.
See [the 18 November report](docs/reviews/PR2_PR5_FOLLOWUP_TR.md).
The [first Samed review](docs/reviews/SAMED_PR2_REVIEW_TR.md) is historical.

Kemal PR #9 review (19 November, `680b875d`): all 164 tests passed independently;
owner integration with the data branch passed 196 tests (LFS audio not downloaded).
GMM-HMM CPU, plain CTC GPU, real mHuBERT-147 + CTC GPU training/checkpoint/predict
and published wav2vec GPU inference passed train-pilot diagnostics. The selected
HuBERT path is our standard CTC training from the documented licensed mHuBERT-147
encoder; the published candidate without a license/model card is not selected.
See [the review, source decision and exact-commit evidence](docs/reviews/KEMAL_PR9_REVIEW_TR.md).
That review used three clips. The later representative measurements above set
the training budgets; all full training/benchmarks remain pending.

Earlier Adnan model review (19 November): Adnan's
[PR #5](https://github.com/kaanksn1/greek-parliamentary-asr/pull/5) at `0c13ce1`
passes 114 tests. PCM32, attention-device, RNN-T loss-backend and Whisper-prefix
blockers are resolved: all three train-pilot GPU checks show finite loss/gradients
and parameter updates. Attention/RNN-T actual train/checkpoint/predict CLIs also
pass GPU checks; a temporary merge with Samed's `4db1a4a` passes 146 tests
(large LFS audio not downloaded for this code integration check).
The user subsequently assigned the final P2 fixes to the owner: nonfinite
Whisper losses now fail with an explicit run record, prediction validates the
source lock, ignored RNN-T padding is safe before gather, and docs/configs match
the code. That model/common suite passed 119 tests; temporary integration with the data branch passed
151 tests. All three final GPU checks and the attention/RNN-T CLI checks pass.
See [the resolution and exact-commit evidence](docs/reviews/SEQUENCE_P2_RESOLVED_TR.md).
The earlier findings remain versioned in
[the review](docs/reviews/ADNAN_0c13ce1_REVIEW_TR.md). Full-data training budgets and
benchmark runs are still pending. The original
assignment remains in [HANDOFF_ADNAN_TR.md](docs/HANDOFF_ADNAN_TR.md).
Full runs remain central on Serdar's machine. The user reported the deadline
as **21 November 2026, Saturday evening (Europe/Istanbul)**; the exact hour is unknown.

Adnan's model implementations and the owner's final review fixes are in
`greek_asr/models/` (see
[SEQUENCE_MODELS.md](docs/SEQUENCE_MODELS.md) and
[the contribution file](contributions/ADNAN_DEMIR_220911784_adnan-demir.md)):
a standard attention encoder-decoder with an optional hybrid CTC head, a
standard RNN-T transducer, and the two-condition Whisper fine-tuning
contribution. Data delivery is under review in PR #2; its selected train upload
and corrected CSS10 audit are pending. The measured `max_steps` allocations are
in `configs/training_budget.json`. Official dev selection and benchmark results
remain pending.

The main source is [HParl](https://inventory.clarin.gr/corpus/1602), a
Greek parliamentary corpus distributed under CC BY-NC 4.0. The producer reports
about 120 hours, with an official session-based train/development/test split.
Samed's local run measured exactly these split durations; the generated
numbers are in [DATA_REPORT_TR.md](docs/DATA_REPORT_TR.md).
The planned additional training source is CSS10 Greek v1, a single-speaker
audiobook corpus. The producer reports `04:08:14`; its distribution metadata
lists CC0. Combined source availability is `123:55:56` (about 124 hours), not
our final cleaned/used duration. CSS10 will not enter the parliamentary test.
See `docs/SECONDARY_SOURCE_TR.md` and `DATA_LICENSE.md`.

Historical startup (18 November 2026): three original training WAVs and transcripts
acquired and audited; common scoring code and baseline inventory created;
Whisper, Qwen3-ASR, Canary and Parakeet completed GPU inference on those WAVs
in two new isolated runtimes. Owner adapters and their common audio-only
runner are implemented and exercised on this training fixture.
No full benchmark result or training improvement is claimed. PR #2 contains
source metadata, cleaning logs, an unchanged early test freeze and LFS
test/dev/CSS10 audio. Independent original-byte verification and the remaining
train/CSS10-audit blockers are described at the top of this README.
The five-member ownership plan is now assigned in `docs/TEAM.md`. Full runs
will be central on Serdar's PC and Ozan will present. The user authorized
owner implementation on 18 November. Write collaborator invitations were sent
to `SmtOrtakaya`, `kemalarif`, `adnan-demir` and `serhandenizhan`; members must
accept and contribute using their own accounts and branch/PR.
Greek was absent from the user's current reservation screenshot; the group's
language/dataset reservation must still be recorded in the class sheet.
Hungarian/VoxPopuli and Icelandic/Samromur are already taken.

Our planned contribution is fine-tuning the documented Whisper-small starting
checkpoint under matched HParl-only and HParl+CSS10 training budgets, with its
unchanged published baseline reported separately. Common settings and updates
are in `configs/training_budget.json`; dev selects checkpoints and scoring.
The architectures remain standard. No bonus or improvement is claimed yet.

The professor reply shared on 17 November approved the meaningful two-corpus
cleaning plan and the use of an additional training source with parliamentary
evaluation. Test data must be frozen in advance and kept unchanged; changes
are permitted on train/dev. No fixed minimum hours were specified: use the
largest statistically useful dataset feasible within measured hardware/time/cost
limits. See `docs/TEACHER_CLARIFICATION_TR.md`. No bonus is claimed as earned.

## Start

The initial scoring code uses Python 3.12's standard library:

```powershell
python -m unittest discover -s tests -v
python -m greek_asr.score --help
```

Clone this repository independently of the older projects:

```bash
git clone https://github.com/kaanksn1/greek-parliamentary-asr.git
cd greek-parliamentary-asr
```

The data owner is Samed. His current copyable task prompt is
[HANDOFF_SAMED_FINAL_DATA_TR.md](docs/HANDOFF_SAMED_FINAL_DATA_TR.md); the initial
assignment is retained in `docs/HANDOFF_SAMED_TR.md`. The exact audio/reference/output
schema is [DATA_CONTRACT.md](docs/DATA_CONTRACT.md). Data/model work remains
with the assigned members; the integration starter is not their completed work.

## Data pipeline (Samed)

`python -m greek_asr.data` acquires HParl 1.0.0 and CSS10 Greek, freezes the
original HParl test **before** any train/dev cleaning, audits train/dev/CSS10,
applies the versioned cleaning policy with a per-record log and writes the
HParl-only and HParl + CSS10 training manifests. Design, rules and the run
order are in [DATA_PIPELINE_TR.md](docs/DATA_PIPELINE_TR.md); the numbers are
generated into `docs/DATA_REPORT_TR.md` after a real run.

```bash
pip install -r requirements-data.txt
python -m greek_asr.data hparl-download   # resumable, ~11.9 GB
python -m greek_asr.data prepare-freeze   # extract, raw manifests, test freeze (commit separately)
python -m greek_asr.data prepare-clean --css10-archive <kaggle zip>
python -m greek_asr.data verify
```

### Split used and speaker overlap

We use the **official HParl 1.0.0 session-based train/dev/test split**. It is
**not speaker-disjoint**: most test speakers also speak in train (other
sessions); session overlap between splits is zero. The professor's email
(How 1) permits a documented official split, so the frozen test is kept and not
resplit. Results are therefore **not** presented as generalization to unseen
speakers. Speakers, utterances, hours and sessions per split and per training
condition, test examples whose speaker occurs in training, and literal
(untransformed) test-sentence matches in each actual training target are
generated in `docs/DATA_REPORT_TR.md` (section 4b) and
`data/clean/cohort_summary.json`; `python -m greek_asr.data split-summary`
recomputes them from the manifests without audio. CSS10 has no per-utterance
speaker/recording IDs (one reader and one book per its producer); those
records are counted as unknown, not invented.

### Data status

| Part | Status |
|---|---|
| Acquisition, test freeze (`d28c97f`), audit and cleaning | Run on Samed's machine; CSS10 float signal audit must be corrected |
| Code, tests, manifests, clean references, cleaning log, audit, source records | In this repository |
| Original Kaldi metadata and CSS10 transcript | In this repository, byte-checked by `verify-source-metadata` |
| Source audio (test/dev/CSS10, ~3.4 GiB) | In this repository via Git LFS (owner approved on 18 Nov 2026) |
| Source audio (selected train) | Pending 23,615 RAW WAVs from 18 sessions, exact list in `configs/training_data_selection.json`; the complete 99-hour source train will not be used |
| Independent audio check | Test/dev verifier passed with audio; CSS10 hashes/RIFF headers passed, but its data verifier on Python 3.12 and signal audit need Samed's float-aware correction |

`docs/DATA_REPORT_TR.md` is the generated data report (sources, freeze, raw
inventory, audit flags, cleaning results with examples, split/speaker and
literal overlap, delivery sizes). It is produced by
`python -m greek_asr.data report` and must not be edited by hand. Its current
CSS10 signal statistics describe the old integer reinterpretation and need
regeneration; source durations/hash identities are separately verified.

The scorer accepts JSONL references (`id`, `reference`) and predictions (`id`,
`text`). Full ID equality is required. It reports corpus WER/CER from edit
counts, not the unweighted mean of sentence error rates. Its initial
normalization preserves accents, punctuation and digits; the final Greek
scoring policy remains a development decision.

```powershell
python -m greek_asr.score --references references.jsonl --predictions predictions.jsonl --split validation --output outputs/validation.json
```

Official-test scoring additionally requires `--allow-test`. That argument is
an accidental-use guard; the scientific requirement is to freeze and commit
the protocol before test inference/scoring.
The test data itself must be frozen earlier and must not be cleaned, filtered,
resplit, relabelled or extended. The draft normalizer is not permission for
additional transformations of the frozen test.

## Standard pretrained inference

The owner provides `python -m greek_asr.infer` for `whisper`, `canary`,
`parakeet`, and `qwen`; Kemal's published CTC checkpoints use the same runner
as `wav2vec` and `hubert` (see the HMM/CTC section below). It loads the exact producer revision in the source lock
once per run, validates every WAV/hash/ID, passes audio only to the model and
keeps genuine empty predictions. Backend failures leave a failed run and
`predictions.partial.jsonl`, not a complete benchmark. Choose a new output
directory for each run. Imports/manifest checks do not require an ASR toolkit.

Export the existing three training clips into the shared interface without
editing their source data or including gold text in model inputs:

```bash
python scripts/prepare_owner_pilot.py
python -m greek_asr.infer --help
```

After installing the appropriate isolated runtime, run a training-only smoke:

```bash
HSA_ENABLE_DXG_DETECTION=1 python -m greek_asr.infer --manifest outputs/owner_pilot_manifest.jsonl --backend whisper --split train --purpose runtime-smoke --output outputs/whisper_train_smoke
```

Use the Transformers runtime for Whisper/Qwen and the separate NeMo runtime
for Canary/Parakeet. Backend sources, dependencies and limitations are in
[PRETRAINED_TRANSFORMERS.md](docs/PRETRAINED_TRANSFORMERS.md) and
[NEMO_RUNTIME.md](docs/NEMO_RUNTIME.md). Only `train` is permitted for runtime
smokes. Full-data benchmark inference must wait for Samed's data delivery,
the early immutable test freeze and later final protocol commit.
The tested Transformers setup/replay and commands are in
[RUNTIME_ENVIRONMENT.md](docs/RUNTIME_ENVIRONMENT.md). Install the pinned
environment in WSL with `bash scripts/setup_transformers_rocm.sh`; use
`bash scripts/setup_nemo_rocm.sh` for the separate NeMo environment. These
assume an existing compatible GPU driver/runtime and do not alter older
project environments.

Actual owner runtime checks on 18 November (all on the same 3 training clips,
6.61 seconds total; no model training, WER/CER or official dev/test access):

| Standard baseline | Complete predictions | Actual run record |
|---|---:|---|
| Whisper-small | 3/3 | [run.json](results/runtime_smoke/owner_whisper_train/run.json) |
| Qwen3-ASR-0.6B | 3/3 | [run.json](results/runtime_smoke/owner_qwen_train_verified/run.json) |
| Canary-1B-v2 | 3/3 | [run.json](results/runtime_smoke/owner_canary_train/run.json) |
| Parakeet-TDT-0.6B-v3 | 3/3 | [run.json](results/runtime_smoke/owner_parakeet_train_verified/run.json) |

On ROCm, Parakeet's NVIDIA CUDA Graph acceleration is disabled before restore;
the standard greedy TDT decoding is retained. Qwen's requested and observed
attention implementations are recorded separately. These are documented runtime
compatibility settings, not our trained-model contribution. Full benchmark
performance and the remaining members' baselines are still pending.

Members start their own assigned branch from `main`, test their changes and
open a PR. See `docs/TEAM.md` for ownership. `contributions/README.md` lists the
required personal filenames; members write their own actual reports. Git LFS
is available locally, but the full-data publication size/quota is not yet
settled. No paid storage or cloud compute is assumed.

## Sequence models: attention, RNN-T, Whisper fine-tuning (Adnan)

Full details, architecture citations, toolkit/loss-backend license
verification and the GPU pilot package are in
[SEQUENCE_MODELS.md](docs/SEQUENCE_MODELS.md). Build the isolated CPU
development environment and run the unit tests (70 tests across
`tests/test_models_common.py`, `tests/test_models_attention.py`,
`tests/test_models_transducer.py`, `tests/test_whisper_finetune.py` and
`tests/test_rnnt_loss.py`; 15 run without any ASR toolkit, the other 55 need
this environment and are skipped, not failed, without it):

```powershell
py -3.12 -m venv .venv-sequence
.venv-sequence\Scripts\python -m pip install -r requirements-sequence.txt
.venv-sequence\Scripts\python -m unittest discover -s tests -v
```

Real training/prediction commands use the measured allocations in
`configs/training_budget.json`. Start only after Samed's selected raw/clean-used
audio verification and corrected CSS10 audit pass. On the isolated ROCm runtime,
use `HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST`:

```bash
python -m greek_asr.models.attention train --config configs/sequence/attention.json --output outputs/attention_run --device cuda
python -m greek_asr.models.transducer train --config configs/sequence/transducer.json --output outputs/transducer_run --device cuda
python -m greek_asr.models.whisper_finetune train --config configs/sequence/whisper_hparl_only.json --output outputs/whisper_hparl_only_run --device cuda
python -m greek_asr.models.whisper_finetune train --config configs/sequence/whisper_hparl_css10.json --output outputs/whisper_hparl_css10_run --device cuda
```

All three `train`/`predict` CLIs now take `--device {cpu,cuda}` explicitly
(default `cpu`; use `cuda` on Serdar's ROCm machine, which reports its
accelerator through the same `"cuda"` device string). Requesting an
unavailable device raises instead of silently training on CPU. Each module
also provides a `predict` subcommand with the same audio-only, no-gold-leak
contract as `greek_asr.infer`, and the same `--split test` refusal without
`--allow-test`/a committed `--test-protocol`.

## HMM and CTC family: GMM-HMM, plain CTC, wav2vec 2.0, HuBERT (Kemal)

Methods, checkpoints, the HuBERT checkpoint search, exact commands and the
measurements still needed are in [KEMAL_BASELINES.md](docs/KEMAL_BASELINES.md).
No full-data training or WER/CER result exists yet; all full runs are central.

| Baseline row | Code | Provenance |
|---|---|---|
| MFCC + GMM-HMM | `greek_asr/models/gmm_hmm.py` (`train`, `tune`, `predict`) | standard method, our implementation and training |
| Plain CTC | `greek_asr/models/ctc.py` (`train`, `predict`) | conv + BiLSTM CTC trained from scratch by us |
| wav2vec 2.0 + CTC | `greek_asr.infer --backend wav2vec` → `greek_asr/models/wav2vec.py` | published `lighteternal/wav2vec2-large-xlsr-53-greek`, no training by us |
| HuBERT + CTC (published) | `greek_asr.infer --backend hubert` → `greek_asr/models/hubert.py` | Historical research adapter; candidate without license/card is not selected for benchmark |
| HuBERT + CTC (our training) | `greek_asr/models/hubert.py` (`train`, `predict`) | Selected standard HuBERT baseline path: our CTC training from CC-BY-NC-SA-4.0 `utter-project/mHuBERT-147`; full training pending |

They use the existing Transformers ROCm runtime; no package or toolkit is added.
`configs/kemal/ctc.json` and `configs/kemal/hubert_ctc.json` now contain measured
owner allocations and the selected cohort paths. Example after data acceptance:

```bash
python -m greek_asr.models.gmm_hmm train --config configs/kemal/gmm_hmm.json --output outputs/kemal_gmm_hparl --device cpu
python -m greek_asr.models.ctc train --config configs/kemal/ctc.json --output outputs/kemal_ctc_hparl --device cuda
python -m greek_asr.infer --manifest data/manifests/dev.jsonl --backend wav2vec --split dev --output outputs/kemal_wav2vec_dev
```

The real wav2vec and published HuBERT checkpoints transcribed the three
training-pilot clips through the shared runner on a CPU
([wav2vec run.json](results/runtime_smoke/kemal_wav2vec_train_cpu/run.json),
[HuBERT run.json](results/runtime_smoke/kemal_hubert_train_cpu/run.json));
no WER/CER and no dev/test inference.

## Files and status

| Path | Purpose |
|---|---|
| `greek_asr/` | Initial common WER/CER scorer and explicit JSONL scoring CLI |
| `greek_asr/infer.py`, `greek_asr/pretrained/` | Owner's pinned standard inference CLI and four backend adapters |
| `greek_asr/audio.py` | Shared original-byte RIFF input contract for PCM16/PCM32/IEEE float32 in Python 3.12; no on-disk audio conversion |
| `greek_asr/models/` | Adnan's attention (hybrid CTC/attention), RNN-T and Whisper fine-tuning code; see `docs/SEQUENCE_MODELS.md` |
| `greek_asr/models/gmm_hmm.py`, `ctc.py`, `wav2vec.py`, `hubert.py`, `ctc_utils.py` | Kemal's GMM-HMM, plain CTC, published wav2vec/HuBERT CTC inference, HuBERT-CTC training and shared CTC helpers; see `docs/KEMAL_BASELINES.md` |
| `configs/sequence/` | Adnan's methods with owner's measured train/dev allocations; matched Whisper conditions |
| `configs/kemal/` | Kemal's GMM-HMM, plain CTC and HuBERT-CTC recipes with owner's measured allocations and separate diagnostic `*_measure.json` configs |
| `configs/training_data_selection.json`, `training_budget.json` | Exact selected RAW/clean training cohort, storage bound, measurements and full-run start conditions |
| `configs/kemal_model_sources.json` | Pins and model-card findings for checkpoints found after the historical lock (Greek HuBERT CTC candidate, mHuBERT-147) and rejected candidates |
| `docs/KEMAL_BASELINES.md` | Kemal's methods, HuBERT checkpoint search, commands, data/test boundary, measurements needed and checks run |
| `tests/` | Scoring, real-pilot lineage, audio/manifest/cohort, adapter-contract, sequence-model and Kemal's GMM-HMM/CTC/wav2vec/HuBERT tests |
| `scripts/` | Scorer/source checks, owner pilot export, isolated runtime bootstrap and Adnan's sequence-model GPU pilot scripts |
| `scripts/prepare_budget_pilot.py`, `measure_train_budget.py` | Original-byte, TRAIN-only representative acquisition and synchronized isolated-process GPU/CPU timing, memory and finite-gradient checks |
| `scripts/select_training_delivery.py`, `finalize_training_plan.py` | Reuse Samed's seeded whole-session selection; generate raw/clean delivery lists, measured run configs and the budget report |
| `scripts/inspect_css10_delivery.py` | Independently verify every delivered CSS10 audio hash/header and demonstrate float-vs-integer signal-statistics error |
| `docs/HANDOFF_SAMED_FINAL_DATA_TR.md`, `docs/reviews/budget_2026-11-20/` | Current data-owner assignment and reproducible measurement/original-byte verification evidence |
| `configs/` | Dataset provenance and pinned checkpoint metadata; no model weights |
| `data/README.md` | Raw/clean data delivery contract and acquisition status |
| `docs/TOPIC_DECISION_TR.md` | Topic rationale, alternatives and class reservation text |
| `docs/TEAM.md` | Five-member task ownership, required deliverables and central-run workflow |
| `docs/SECONDARY_SOURCE_TR.md` | Selected CSS10 Greek source, version, reported hours, license evidence and limits |
| `docs/DATA_CONTRACT.md` | Shared audio/reference/prediction schema and two freeze stages |
| `docs/HANDOFF_SAMED_TR.md` | Detailed data-owner task prompt, deliverables and acceptance checks |
| `docs/HANDOFF_ADNAN_TR.md` | Next model-owner prompt after accepted data delivery; training code, controls, tests and central-run package |
| `docs/SEQUENCE_MODELS.md` | Adnan's architecture/toolkit/loss-backend rationale, environment, GPU pilot package and known limitations |
| `docs/ADNAN_SLIDES_DRAFT_TR.md` | Adnan's 1-2 slide draft and speaker notes for Ozan; numbers pending real runs |
| `requirements-sequence.txt` | Adnan's isolated CPU dev environment pins for `greek_asr/models/` (torch/torchaudio/transformers) |
| `docs/PRETRAINED_TRANSFORMERS.md` | Whisper/Qwen APIs, standard settings and primary sources |
| `docs/NEMO_RUNTIME.md` | Canary/Parakeet APIs, separate dependencies and actual runtime status |
| `docs/RUNTIME_ENVIRONMENT.md` | Tested isolated Whisper/Qwen installation, lock replay and training-pilot commands |
| `docs/BASELINES.md` | Required family inventory, verified checkpoint candidates and pending work |
| `docs/PROTOCOL_DRAFT.md` | Draft experimental protocol; not a frozen final-test protocol |
| `docs/INITIAL_WORK_TR.md` | Completed startup work, observed reference issue and remaining work |
| `docs/TEACHER_CLARIFICATION_TR.md` | Professor approval, immutable test requirement and data-budget rule |
| `docs/TEACHER_UPDATES_2026-11-18_TR.md` | New speaker/group split rule, official-split exception and exact baseline/scoring requirements |
| `docs/reviews/` | Versioned PR #2/#5/#9 reviews, source/licensing decisions and metadata/GPU/CLI/math evidence; no model benchmark |
| `results/runtime_smoke/` | Real train-pilot predictions and runtime record; no benchmark scores |
| `greek_asr/data/` | Samed's data pipeline: acquisition, test freeze, audit, cleaning, report, delivery shards |
| `configs/data_cleaning.json` | Versioned train/dev/CSS10 cleaning policy and flag thresholds |
| `docs/DATA_PIPELINE_TR.md` | Data pipeline design, rule justifications and run order |
| `docs/DATA_REPORT_TR.md` | Generated data report (numbers only from pipeline outputs; regenerate, never hand-edit) |
| `configs/data_delivery.json` | Decision record for how/whether each data group is delivered in the repository |
| `data/` | Raw source records, frozen test, raw/clean manifests and references, audit and cleaning outputs (see `data/README.md`) |
| `requirements-data.txt` | Exact data-pipeline dependency (`numpy`) |
| `tests/test_data_pipeline.py`, `tests/data_fixtures.py` | Synthetic end-to-end data tests with known problems |
| `contributions/` | Actual individual reports and required filename/status inventory |
| `AGENTS.md` | Scope and test instructions for coding assistants |
| `requirements.txt` | Dependency scope of the initial implementation |
| `scripts/review_data_submission.py` | Read-only integration review of submitted manifests, literal repeated test sentences and freeze history |
| `scripts/probe_sequence_submission.py` | Independent PCM32/token-prefix probes and one-step submitted model checks on three train clips; no dev/test inference |
| `scripts/probe_sequence_entrypoints.py` | Real attention/RNN-T train/checkpoint/predict CLI diagnostics using temporary train-pilot fixtures; no official dev/test |
| `scripts/verify_rnnt_submission.py` | CPU exhaustive-alignment oracle for submitted RNN-T values/gradients and ignored-padding contract |
| `scripts/probe_kemal_submission.py` | Owner's independent GMM-HMM CPU, CTC/mHuBERT GPU training and wav2vec GPU diagnostics on three train clips; no official dev/test |
| `requirements-verification.txt` | Exact optional JiWER oracle-check dependencies |
| `requirements-rocm.lock.txt` | Actual isolated Whisper/Qwen environment pins, including AMD wheel identities |
| `requirements-nemo.lock.txt` | Actual separate NeMo environment pins; runtime validation status in its documentation |
| `LICENSE` | MIT license for our own code |
| `DATA_LICENSE.md` | HParl and CSS10 Greek attribution and separate dataset licenses |
| `.gitignore` | Local environments, caches and generated experiment outputs; raw/clean data are not ignored |
| `.gitattributes` | Consistent source line endings and binary audio/document handling |

The original GPU preflight passed on the existing WSL ROCm installation:
AMD Radeon RX 7800 XT, Python 3.12.3, PyTorch 2.9.1+ROCm 7.2.1, finite
small-matrix forward/backward and three-clip Whisper inference. That historical
record remains in `results/runtime_smoke/whisper_small_train/run.json`.
The owner now has separate new Transformers and NeMo inference environments;
their actual pins and current runtime evidence are recorded separately.
Adnan's submitted training implementations still need the corrections documented
in the latest review; all full training and benchmark runs remain pending.

Before full runs: obtain the official data release, freeze the original test,
inspect real train/dev quality, acquire and audit the selected CSS10 release, measure
the feasible data budget, confirm class reservation and the exact deadline hour, and reconcile
the full Week 9 slides with the baseline inventory. Do not substitute a mirror's
random split for the official session split.

The official archive is reachable by public bounded HTTP GET: its ZIP index
contains 92,133 WAV entries. This verifies access and source structure, not
the duration or quality of every recording. The train pilot deliberately
fetches only a small reproducible training sample; it is not the final dataset.

```powershell
python scripts/fetch_train_pilot.py --count 3 --seed 42
python scripts/audit_train_pilot.py --pilot data/raw/pilot --output outputs/train_pilot_audit.json
```

The three source clips total 6.61 seconds. One original reference contains
`<spoken_noise>`; it is retained and flagged for reference-policy review. No
development/test transcript or audio was read. The full data release and a
justified final subset remain pending.

GPU runtime command in the existing compatible ROCm environment:

```bash
HSA_ENABLE_DXG_DETECTION=1 /opt/banking77-venv/bin/python scripts/smoke_whisper_train.py
```

This is a record of the local preflight, not a portable environment-install
recipe. The script uses only train-pilot audio as model input, writes Greek
transcriptions and an environment record, and performs no model training or
WER/CER benchmark. To repeat it, choose a new `--output` directory.

Optional independent scorer check in an isolated verification environment:

```powershell
pip install -r requirements-verification.txt
python scripts/verify_scoring.py
```

Historical startup verification: 44 unit tests passed (including the original
15 startup tests); the independent startup JiWER check
matched WER/CER on 500 deterministic fixture corpora. The latter are synthetic
verification cases, not audio data or model results.

Adnan's submitted suite reached 114 tests; the owner's final P2 regressions
brought it to 119. Kemal added 45 tests, bringing the reviewed model/common
suite to **164 tests, all passed without skips in the isolated ROCm runtime**.
They include real-train-pilot code paths, synthetic model/decoder mathematics,
checkpoint reload, finite gradients, CTC lengths/padding and failure handling.
Missing optional ASR dependencies cause explicit skips; a partial environment
is not evidence that the complete suite passed. See `docs/SEQUENCE_MODELS.md`
and `docs/KEMAL_BASELINES.md` for implementation and environment details.
