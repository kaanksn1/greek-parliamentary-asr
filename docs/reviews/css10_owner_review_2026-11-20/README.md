# Independent CSS10 data acceptance

Reviewed Samed's PR #2 at `28930653a6af253541b6750cea221877d7bce012`.
No blocking data finding remains. Original Samed commits and his contribution
report are retained. Model code, locked selection, HParl/dev/test manifests
and references did not change from `45bb2b4e` to the reviewed submission.

All 229 tests passed, zero skipped, in the owner's existing Python 3.12.3,
NumPy 1.26.4, torch/torchaudio/Transformers runtime with GPU disabled for tests.
`unittest.txt` is captured output with trailing terminal whitespace removed.
Samed's original data-only
environment passed 130 and skipped 99 model tests; its historical log remains
unchanged under `../css10_reaudit_2026-11-20/`.

`scripts/verify_css10_training.py` reproduced the unchanged submitted data
commands in isolated staging. It reused 39,407 original HParl WAVs through
hard links and copied/hash-checked the 1,844 already-cached original CSS10 WAVs;
no additional LFS audio download was made. The fresh source-metadata check,
selected audit, clean and audio-enabled verify all exited zero. The selected
verify checks 32,572 train/dev/CSS10 WAVs plus all 8,679 frozen test WAVs.

The locked HParl/dev outputs, CSS10 and mixture paths/hashes, real CSS10 signal
statistics/flags, cleaning log and review queue all match the submission. The
test freeze and every locked input remained unchanged before/after. Current
CSS10 remains 1,844 clips / 4.137269 h; the mixture is 25,446 / 35.020678 h.
The additional long-edge-silence flag remains in the review queue; no human
listening decision or new ground-truth annotation is claimed.

`configs/css10_training_acceptance.json` is the independent owner acceptance.
The four `css10_check_*.log` files are generated command outputs;
`lineage_reproduced_linux.json` is the unedited independently regenerated
lineage. Platform/environment fields and audit-summary hashes can differ
between Windows and Linux; actual cleaning decisions and protected data match.

This accepts the matched mixed-corpus Whisper training condition with the same
checkpoint revision, seed, microbatch, accumulation, learning rate, warmup,
updates and dev set as the completed HParl-only condition. It does not authorize
test inference or establish a WER/CER improvement. Final evaluation remains a
separate committed protocol.
