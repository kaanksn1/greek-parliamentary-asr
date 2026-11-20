# Project 3 — Greek parliamentary ASR

The user explicitly started Project 3 on 16 November 2026. This directory is
independent of BANKING77 and RAGTruth; do not change their code, data or reports.

The five-member ownership plan is in `docs/TEAM.md`, assigned on 17 November.
On 18 November the user explicitly authorized completing Serdar's own tasks,
creating the public repository, inviting all four teammates and preparing a
detailed Samed handoff. Owner integration/standard inference/environment work
is now active. Previous startup evidence remains valid. Data cleaning and other
teammates' model implementations remain with their assigned owners.

Serdar owns GitHub/integration, environments, all central full runs and standard
Whisper/Canary/Parakeet/Qwen inference adapters. Samed owns data preparation and
cleaning. Kemal owns MFCC+GMM-HMM, plain CTC, wav2vec/HuBERT ASR. Adnan owns
attention encoder-decoder, transducer and Whisper fine-tuning. Ozan owns common
evaluation, comparisons and presentation. Keep implementation within the
assisted owner's scope unless the user explicitly reassigns work. Shared setup
does not authorize completing teammates' implementations or personal reports.
Do not fabricate contribution files, claim other members' work, or invent
completed runs. Retain genuine member authorship when work is later integrated.

On 19 November the user explicitly assigned the remaining PR #5 review fixes
to the owner: Whisper nonfinite-loss handling and prediction source-lock check,
RNN-T ignored-padding handling, and related config/documentation corrections.
Implement these limited fixes under the owner's own commits, retaining Adnan's
original commits and personal contribution report. Other model/data ownership
is unchanged. This extension does not authorize full-data training or final
test inference before the existing data/budget/protocol requirements are met.

Use the professor announcements in this conversation. All standard Week 9
methods are baselines, including advanced pretrained ASR. Own training or
fine-tuning is reported separately from standard baseline results.

Preserve source audio/transcripts and official split/session identities. Model
outputs are not verified gold. The professor reply shared on 17 November is
recorded in `docs/TEACHER_CLARIFICATION_TR.md`: freeze the original test data
early, before train/dev cleaning or mixed-corpus experiments. Do not clean,
filter, add/drop, resplit or relabel test data. Keep the second corpus out of
the parliamentary test. Samed submitted freeze metadata in PR #2; the review
verified that it has not changed since its initial commit. Full source audio
delivery/independent audio-byte verification are still pending. Review at
`e5aa488` verified the committed original Kaldi/CSS10 text metadata and the
earlier reporting corrections. PR #5's former PCM32, GPU and Whisper-prefix
blockers at `0437631` were fixed in `0c13ce1`: 114 tests and all three real
train-pilot GPU checks passed; attention/RNN-T actual CLI checks also passed.
The remaining P2 corrections in `docs/reviews/ADNAN_0c13ce1_REVIEW_TR.md` were
subsequently assigned to and implemented by the owner as scoped above: fail-fast
nonfinite loss, safe ignored padding, predict source-lock and matching docs/configs.
The final local suite passes 119 tests. Original model authorship remains Adnan's.
Passing these fixtures does not establish full-data training feasibility.

Kemal submitted PR #9 at `680b875d`: 164 tests passed independently; GMM-HMM
CPU, plain CTC GPU, real mHuBERT-147 + CTC GPU training/checkpoint/predict and
published wav2vec GPU inference passed train-pilot diagnostics. Model authorship
remains Kemal's. The owner selected `utter-project/mHuBERT-147` + our Greek CTC
training, with its documented CC-BY-NC-SA-4.0 model license. The published
`ntinouldinho/hubert_greek_6gram` candidate has no license/card/training-data
statement and is not selected for benchmark. Do not label a bare encoder or
the diagnostic head updates as completed Greek ASR training/benchmark.
See `docs/reviews/KEMAL_PR9_REVIEW_TR.md`. Full representative measure runs,
selected train audio delivery and training budgets remain pending.

Freeze and commit the final model/settings/scoring protocol separately, before
test inference/scoring. Do not tune from test. The two further professor
emails shared on 18 November are summarized in
`docs/TEACHER_UPDATES_2026-11-18_TR.md`. One common versioned scoring-time
normalization policy must be selected on train/dev for every reference and
hypothesis. This does not authorize rewriting frozen source files.

Split by speaker/group when designing a new split. The professor explicitly
permits a documented official split if the README identifies it and reports
whether speakers are separate. HParl's official session split is NOT speaker
disjoint; disclose overlaps, do not call it unseen-speaker evaluation, and do
not silently resplit the previously frozen original test. Report speakers,
utterances, hours and literal test-sentence matches in training separately
from normalized unique-text overlaps.

The two-corpus cleaning plan and domain-focused evaluation were approved in
principle; that is not a completed bonus award. There is no professor-specified
fixed minimum hour count. Justify the largest feasible, statistically useful
dataset using actual hardware/time/cost measurements, not the three-clip smoke.
The planned second source is CSS10 Greek v1, only for training. Its producer
reports 04:08:14; the data submission records its audit, but owner-independent
verification of the full audio bytes is pending.
See `docs/SECONDARY_SOURCE_TR.md` for provenance and limitations.

The user's 18 November reply sets the deadline to 21 November 2026, Saturday
evening, Europe/Istanbul. Exact hour remains unknown; do not invent one or
choose an unmeasured training budget from it.

The draft normalization is not the final Greek scoring rule. Keep references
and predictions under one common, versioned policy. Report corpus edit counts,
full cohort identity and source hashes.
Run every baseline supported for Greek. An unsupported checkpoint needs a
model-card-backed explanation, not a wrong-language run. Bare encoders are
not ASR. Published fine-tuned checkpoints remain baselines; our own training
is explicitly attributed and reported as our training contribution.

Verification for common scoring changes:
`python -m unittest discover -s tests -v` from this directory.
Model runs and data cleaning require their own appropriate checks once implemented.

20 November owner integration update: all test/dev LFS WAVs passed independent
audio verification; CSS10's 1844 original audio hashes/RIFF headers also match.
Actual CSS10 is IEEE float32 (format tag 3), not integer PCM32. The owner repaired
the shared input contract in greek_asr/audio.py, infer.py and the thin
models/common.py decoder adapter; no model architecture was changed. All 203
tests passed after that fix. Samed still owns correcting data/common.py and the
format-aware signal audit/clean reports, plus selected RAW train audio delivery.
Current handoff: docs/HANDOFF_SAMED_FINAL_DATA_TR.md. Do not claim that the old
CSS10 integer-interpreted audit is valid or that full-data training is complete.

The user explicitly selected the 10 GiB free storage cap on 20 November. The
owner's measured session-cohort decision, raw/clean delivery hashes and training
allocations are configs/training_data_selection.json and training_budget.json.
Do not buy storage, alter the frozen test or train on unavailable source-catalog
paths. Full runs still wait for the handoff's real-audio acceptance checks.
Qwen's owner adapter supports bounded batch inference; record its batch setting
and choose it on train/dev before the separate final test protocol.

Later on 20 November the user explicitly requested starting training without
waiting for sleeping Samed. Owner may acquire the SAME locked RAW HParl cohort,
verify it and reproduce Samed's unchanged audit/clean pipeline in isolation.
After exact train/dev hashes and unchanged frozen test are verified and the
HParl acceptance is committed, HParl-only training may run while repository
audio upload completes. This dataset-specific rule supersedes the earlier
global CSS10 wait: mixed-corpus training still requires Samed's corrected
float-aware CSS10 audit. The owner does not rewrite his cleaning methods or
personal report. RAW/clean-used data must actually be in the public repository
before final submission. No paid storage or final test inference is authorized.

Later 20 November review: Samed's CSS10 data correction at 28930653 passed all
229 tests independently, with zero skips in the owner's Python 3.12 runtime.
An isolated real-audio audit/clean/verify reproduction passed using 41,251
original WAV paths; the selected HParl/dev, CSS10/mixture, cleaning log and
review queue match the submission. configs/css10_training_acceptance.json
records the owner acceptance for matched mixed Whisper training. All six
HParl-only jobs at immutable execution snapshot d98abe995 completed. Keep that
snapshot and its original run records; run mixed Whisper from a separate
committed snapshot with the same optimizer/update budget and starting model.
Frozen test is still untouched and final test inference remains separately locked.
