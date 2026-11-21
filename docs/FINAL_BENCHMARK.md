# Final benchmark and reproduction

All seven full training allocations finished before final test selection.
The six HParl jobs ran at immutable snapshot `d98abe995f85ff634704ec9f60b26ca671a21a29`;
the matched mixed-corpus Whisper job ran at
`b760cdcf2662dbe6a40ee3267e59cb7f41dbe0c5`. Original training `run.json`
records are copied unchanged into `results/training/`. Their full-development
losses select the saved checkpoints; fixed GMM EM completion selects its model.

## Selection before test

`scripts/prepare_benchmark.py` creates a seed-42 calibration sample containing
one recording per development speaker and the two longest development clips.
The committed `configs/evaluation_dev_selection.json` records membership,
references, hashes and normalization examples. This short calibration checks
resource settings/decoders; it does not replace the full dev losses already
used during training. GMM's nine predeclared LM/insertion settings are compared
only on this development sample. Other systems retain their standard decoding.

All 12 systems must finish this identical dev cohort before
`scripts/freeze_benchmark_protocol.py` can create the final protocol. The
protocol must be committed before test inference. It freezes system settings,
exact upstream model IDs/revisions, selected local weight identities, scorer
source, references and test membership. The test runner checks these identities
and original audio hashes before launching inference, and checks the audio
freeze again after completion. Each system uses every original test recording.
No test reference enters a decoder. Infrastructure retries retain failure logs
and use the identical frozen settings; partial cohorts cannot produce a score.

## Common scoring

The final implementation is `greek_asr/evaluation.py`; the older
`greek_asr/scoring.py` remains the historical draft/oracle utility. The frozen
policy is `greek-asr-nfc-casefold-punctuation-v1`, applied in memory equally to
references and hypotheses: remove angle-bracket annotation tags, normalize
Unicode to NFC, casefold, restore NFC, replace Unicode punctuation with spaces
and collapse whitespace. Greek accents, digits and nonpunctuation symbols are
retained. There is no transliteration or conversion between numbers and words.
Frozen reference/audio files are never rewritten.

Corpus WER is `(S + D + I) / reference words`; it measures transcription word
errors and can exceed 100%. CER applies the same calculation to characters,
including internal spaces, making partial errors visible in Greek's inflected
words. Both use exact edit counts from pinned JiWER/RapidFuzz, aggregated over
the corpus rather than averaging utterance rates. Their use and normalization
follow the [Hugging Face ASR evaluation lesson](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

All IDs remain, including references that become empty after normalization.
Words predicted for an empty reference count as insertions. A zero corpus
denominator yields JSON `null`, never an invented zero. The report includes
raw S/I/D counts and empty-hypothesis counts. Word/character accuracy is not
used as a substitute. A seed-42, 1,000-resample speaker-cluster bootstrap gives
95% percentile WER/CER intervals. Paired resampling of the same test speakers
estimates the mixed-minus-HParl-only Whisper WER difference. These intervals
do not measure variation over training seeds.

RTFx is test audio seconds divided by inference wall seconds. Model loading is
excluded when the adapter has a separate inference timer; the output records
the actual scope. It includes decoding/output I/O and depends on CPU/GPU,
precision, batch size and software. It is not a hardware-independent ranking.
Native CPU GMM decoding overlaps the serial GPU queue; keep that contention in
mind when interpreting speed. All run records retain their runtime identities.

## Execute and regenerate

Use a fresh checkout at the committed final protocol, with Git LFS installed:

```bash
git lfs pull
python -m greek_asr.data verify-freeze
python -m greek_asr.data verify --selection configs/training_data_selection.json
```

The original WAVs and selected local checkpoints are actual repository LFS
objects. `checkpoints/README.md` explains optimizer-free exports and licenses.
Published upstream models are separately cached at their exact locked revisions
by the following command; do not substitute another checkpoint. The queue is
deliberately offline, so run this network-enabled preparation before launching it:

```bash
python scripts/cache_benchmark_models.py
python scripts/cache_benchmark_models.py --check-only
```

To reproduce HuBERT training as well, add `--include-training-encoder` to the
cache preparation command before the offline primary training queue. This bare
encoder is training initialization, not an ASR baseline result.

The cache preparation uses the pinned Hugging Face Hub dependency in the
Transformers environment; it neither loads a GPU model nor transcribes test
audio. It fetches the exact files/caches used by both standard and trained
Whisper loaders, wav2vec and NeMo. The selected
Greek HuBERT system has our trained CTC head, rather than a bare encoder.

Reproduce training using the commands/configs in `docs/KEMAL_BASELINES.md`,
`docs/SEQUENCE_MODELS.md` and `configs/training_budget.json`. The full execution
snapshots above preserve the original training code and run records. In a
separate calibration checkout at preparation candidate `220eb85` (before the
selected-weight export directories exist), export selected models and run dev first:

```bash
python scripts/prepare_benchmark.py --primary-run PRIMARY_OUTPUT --mixed-run MIXED_OUTPUT
python scripts/run_benchmark_queue.py --phase dev --output outputs/dev_calibration --nemo-python NEMO_PYTHON
python scripts/freeze_benchmark_protocol.py
# Commit the generated protocol, code/configs and selected weights before test.
```

For the frozen final evaluation, use the Transformers Python as `python` and
the separate NeMo Python for the two NVIDIA checkpoints:

```bash
python scripts/run_benchmark_queue.py --phase test --output outputs/final_benchmark --nemo-python NEMO_PYTHON
python scripts/summarize_benchmark.py
```

`requirements-rocm.lock.txt` and `requirements-nemo.lock.txt` pin the isolated
model runtimes. Add `requirements-evaluation.txt` to the Transformers runtime
for compiled scoring and plotting. NeMo retains its own NumPy 2 runtime; do
not install the evaluation extras into it. On the measured AMD WSL machine,
use `HSA_ENABLE_DXG_DETECTION=1`, `MIOPEN_FIND_MODE=FAST`, two OMP/MKL threads
and one OpenBLAS thread. Environment variables are recorded by the queue.

Results are generated under `results/benchmark/`: each system has untouched
predictions, runtime metadata, corpus metrics, per-utterance errors and selected
error examples. `SUMMARY.md`, `results.csv`, `summary.json` and PNG/SVG figures
are generated from these files. `results/PRESENTATION_BRIEF_TR.md` supplies
Serhan's slides. No table number should be edited manually.

## Interpretation limits

HParl uses its documented official session split; it is not speaker-disjoint.
The selected training data share 49 speakers with test, covering 4,782 test
clips; session overlap is zero. This is not an unseen-speaker benchmark.
Literal/normalized sentence overlap and split counts are in the generated data
report and selected-training record. `scripts/report_benchmark_data.py`
recomputes the final-policy overlap in `results/data_integrity.json`: nine test
utterances / nine unique sentences for both actual training conditions. Keep
this separate from the historical diagnostic, which uses an older rule.

The main contribution compares matched HParl-only and HParl+CSS10 Whisper
fine-tuning with the same starting weights, seed, effective batch, learning
rate, warmup and 1,600 optimizer updates. Equal updates imply different corpus
coverage. Full-dev loss selected step 500 in both completed Whisper allocations;
the benchmark uses these best-dev checkpoints, not automatically the last step.
CSS10 is read literary speech, while test is parliamentary speech;
the mixture is a training-only data-composition experiment. Poor or negative
results remain in the report. Classical/CTC/attention/transducer/HuBERT systems
trained by the group are standard baseline families and also credited as
implementation/training work. Running pretrained Qwen/Whisper/Canary/Parakeet
alone is baseline work. Limited training and one seed do not establish model
family superiority or convergence. Human audit-listening decisions remain zero.
The professor approved the proposed data/domain bonus dimensions in principle;
no bonus points are claimed as awarded.
