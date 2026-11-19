# Greek Parliamentary ASR and Domain Adaptation

COE025 NLP Project 3. Working topic selected on 16 November 2026:
**automatic transcription of Greek parliamentary speech, with a separate
in-domain fine-tuning experiment.**

Public repository: https://github.com/kaanksn1/greek-parliamentary-asr

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

Latest model review (19 November): Adnan's
[PR #5](https://github.com/kaanksn1/greek-parliamentary-asr/pull/5) at `0c13ce1`
passes 114 tests. PCM32, attention-device, RNN-T loss-backend and Whisper-prefix
blockers are resolved: all three train-pilot GPU checks show finite loss/gradients
and parameter updates. Attention/RNN-T actual train/checkpoint/predict CLIs also
pass GPU checks; a temporary merge with Samed's `4db1a4a` passes 146 tests
(large LFS audio not downloaded for this code integration check).
The user subsequently assigned the final P2 fixes to the owner: nonfinite
Whisper losses now fail with an explicit run record, prediction validates the
source lock, ignored RNN-T padding is safe before gather, and docs/configs match
the code. All 119 tests pass; temporary integration with the data branch passes
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
contribution. Data PR #2 is still unmerged, so its full manifests/audio are
not yet integrated into main. Model verification uses synthetic tensors and
three real HParl train-pilot clips, including the recorded GPU checks above.
No full training budget (`max_steps`), official dev selection or benchmark
result exists yet; representative real-data measurements remain necessary.

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
No full benchmark result or training improvement is claimed. PR #2 contains
full-corpus metadata, cleaning logs and an early test freeze, but the referenced
full audio is absent from its checkout. Owner verification therefore covers
metadata, code and the existing three training WAVs, not the full audio corpus.
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

Real training/prediction commands (pending Samed's accepted data delivery and
Serdar's GPU pilot measurement for `max_steps`; `configs/sequence/*.json`
currently refuse to run with `max_steps: null`):

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

## Files and status

| Path | Purpose |
|---|---|
| `greek_asr/` | Initial common WER/CER scorer and explicit JSONL scoring CLI |
| `greek_asr/infer.py`, `greek_asr/pretrained/` | Owner's pinned standard inference CLI and four backend adapters |
| `greek_asr/models/` | Adnan's attention (hybrid CTC/attention), RNN-T and Whisper fine-tuning code; see `docs/SEQUENCE_MODELS.md` |
| `configs/sequence/` | Adnan's method/Whisper-condition configs; `max_steps` is `null` pending Serdar's GPU pilot measurement |
| `tests/` | Scoring, real-pilot lineage, audio/manifest/cohort, adapter-contract and sequence-model tests |
| `scripts/` | Scorer/source checks, owner pilot export, isolated runtime bootstrap and Adnan's sequence-model GPU pilot scripts |
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
| `docs/reviews/` | Versioned PR #2/#5 reviews, metadata/GPU/CLI/RNN-T-math JSON evidence and correction prompts; no model benchmark |
| `results/runtime_smoke/` | Real train-pilot predictions and runtime record; no benchmark scores |
| `contributions/` | Actual individual reports and required filename/status inventory |
| `AGENTS.md` | Scope and test instructions for coding assistants |
| `requirements.txt` | Dependency scope of the initial implementation |
| `scripts/review_data_submission.py` | Read-only integration review of submitted manifests, literal repeated test sentences and freeze history |
| `scripts/probe_sequence_submission.py` | Independent PCM32/token-prefix probes and one-step submitted model checks on three train clips; no dev/test inference |
| `scripts/probe_sequence_entrypoints.py` | Real attention/RNN-T train/checkpoint/predict CLI diagnostics using temporary train-pilot fixtures; no official dev/test |
| `scripts/verify_rnnt_submission.py` | CPU exhaustive-alignment oracle for submitted RNN-T values/gradients and ignored-padding contract |
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

Current lightweight verification: 44 unit tests passed (including the original
15 startup tests); the independent startup JiWER check
matched WER/CER on 500 deterministic fixture corpora. The latter are synthetic
verification cases, not audio data or model results.

Adnan's sequence-model tests add 70 more (114 total): 15 run with no ASR
toolkit installed, the other 55 (tiny synthetic + real-train-pilot-audio
forward/backward/checkpoint/predict checks for all three models, the RNN-T
loss backend's value/gradient correctness, PCM16/32 decoding, device
selection, and the Whisper decoder-start-token/long-audio regressions) need
`requirements-sequence.txt` and are skipped, not failed, without it. All 114
pass inside that environment. See `docs/SEQUENCE_MODELS.md`.
