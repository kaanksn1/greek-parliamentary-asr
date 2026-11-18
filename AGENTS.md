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
earlier reporting corrections. PR #5 at `0437631` has verified PCM32, GPU and
Whisper-prefix defects; do not treat passing CPU tests as full-training readiness.
See `docs/reviews/PR2_PR5_FOLLOWUP_TR.md`. Model corrections remain Adnan's work.

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
