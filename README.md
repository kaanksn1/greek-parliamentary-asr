# Greek Parliamentary ASR and Domain Adaptation

COE025 NLP Project 3. Working topic selected on 16 November 2026:
**automatic transcription of Greek parliamentary speech, with a separate
in-domain fine-tuning experiment.**

Public repository: https://github.com/kaanksn1/greek-parliamentary-asr

The main source is [HParl](https://inventory.clarin.gr/corpus/1602), a
Greek parliamentary corpus distributed under CC BY-NC 4.0. The producer reports
about 120 hours, with an official session-based train/development/test split.
Those are source-reported sizes, not measurements from a completed local download.
The planned additional training source is CSS10 Greek v1, a single-speaker
audiobook corpus. The producer reports `04:08:14`; its distribution metadata
lists CC0. Combined source availability is `123:55:56` (about 124 hours), not
our final cleaned/used duration. CSS10 will not enter the parliamentary test.
See `docs/SECONDARY_SOURCE_TR.md` and `DATA_LICENSE.md`.

Current stage (18 November 2026): three original training WAVs and transcripts
acquired and audited; common scoring code and baseline inventory created;
Whisper, Qwen3-ASR, Canary and Parakeet completed GPU inference on those WAVs
in two new isolated runtimes. Owner adapters and their common audio-only
runner are implemented and exercised on this training fixture.
No full benchmark result or
training improvement is claimed. The full corpus has not yet been acquired,
cleaned or frozen for evaluation.
The five-member ownership plan is now assigned in `docs/TEAM.md`. Full runs
will be central on Serdar's PC and Ozan will present. The user authorized
owner implementation on 18 November. Write collaborator invitations were sent
to `SmtOrtakaya`, `kemalarif`, `adnan-demir` and `serhandenizhan`; members must
accept and contribute using their own accounts and branch/PR.
Greek was absent from the user's current reservation screenshot; the group's
language/dataset reservation must still be recorded in the class sheet.
Hungarian/VoxPopuli and Icelandic/Samromur are already taken.

Our planned contribution is our own fine-tuning of an unchanged, documented
pretrained starting model on parliamentary training audio. Standard model
results remain separate. Candidate comparison: the same model before fine-tuning,
after a smaller training budget, and after the selected larger training budget.
Budgets are selected using training/development data and available compute.
This is not a claim of a new architecture or guaranteed bonus points.

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

The next data owner is Samed. His detailed, copyable task prompt is
[HANDOFF_SAMED_TR.md](docs/HANDOFF_SAMED_TR.md); the exact audio/reference/output
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

Status: code and synthetic end-to-end tests are implemented (the suite now has
73 tests: the earlier 44 plus 29 data tests); the full corpus has not yet been
downloaded, frozen or cleaned.

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
`parakeet`, and `qwen`. It loads the exact producer revision in the source lock
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

## Files and status

| Path | Purpose |
|---|---|
| `greek_asr/` | Initial common WER/CER scorer and explicit JSONL scoring CLI |
| `greek_asr/infer.py`, `greek_asr/pretrained/` | Owner's pinned standard inference CLI and four backend adapters |
| `tests/` | Scoring, real-pilot lineage, audio/manifest/cohort and adapter-contract tests |
| `scripts/` | Scorer/source checks, owner pilot export and isolated runtime bootstrap scripts |
| `configs/` | Dataset provenance and pinned checkpoint metadata; no model weights |
| `data/README.md` | Raw/clean data delivery contract and acquisition status |
| `docs/TOPIC_DECISION_TR.md` | Topic rationale, alternatives and class reservation text |
| `docs/TEAM.md` | Five-member task ownership, required deliverables and central-run workflow |
| `docs/SECONDARY_SOURCE_TR.md` | Selected CSS10 Greek source, version, reported hours, license evidence and limits |
| `docs/DATA_CONTRACT.md` | Shared audio/reference/prediction schema and two freeze stages |
| `docs/HANDOFF_SAMED_TR.md` | Detailed data-owner task prompt, deliverables and acceptance checks |
| `docs/PRETRAINED_TRANSFORMERS.md` | Whisper/Qwen APIs, standard settings and primary sources |
| `docs/NEMO_RUNTIME.md` | Canary/Parakeet APIs, separate dependencies and actual runtime status |
| `docs/RUNTIME_ENVIRONMENT.md` | Tested isolated Whisper/Qwen installation, lock replay and training-pilot commands |
| `docs/BASELINES.md` | Required family inventory, verified checkpoint candidates and pending work |
| `docs/PROTOCOL_DRAFT.md` | Draft experimental protocol; not a frozen final-test protocol |
| `docs/INITIAL_WORK_TR.md` | Completed startup work, observed reference issue and remaining work |
| `docs/TEACHER_CLARIFICATION_TR.md` | Professor approval, immutable test requirement and data-budget rule |
| `results/runtime_smoke/` | Real train-pilot predictions and runtime record; no benchmark scores |
| `greek_asr/data/` | Samed's data pipeline: acquisition, test freeze, audit, cleaning, report, delivery shards |
| `configs/data_cleaning.json` | Versioned train/dev/CSS10 cleaning policy and flag thresholds |
| `docs/DATA_PIPELINE_TR.md` | Data pipeline design, rule justifications and run order |
| `requirements-data.txt` | Exact data-pipeline dependency (`numpy`) |
| `tests/test_data_pipeline.py`, `tests/data_fixtures.py` | Synthetic end-to-end data tests with known problems |
| `contributions/` | Actual individual reports and required filename/status inventory |
| `AGENTS.md` | Scope and test instructions for coding assistants |
| `requirements.txt` | Dependency scope of the initial implementation |
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
The teammates' full training implementations/environments remain pending.

Before full runs: obtain the official data release, freeze the original test,
inspect real train/dev quality, acquire and audit the selected CSS10 release, measure
the feasible data budget, confirm class reservation/deadline, and reconcile
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

Current lightweight verification: 44 unit tests passed (including the original
15 startup tests); the independent startup JiWER check
matched WER/CER on 500 deterministic fixture corpora. The latter are synthetic
verification cases, not audio data or model results.
