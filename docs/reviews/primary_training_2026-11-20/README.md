# Primary HParl training evidence

The execution checkout is frozen at
`d98abe995f85ff634704ec9f60b26ca671a21a29`; dataset-specific acceptance is
`configs/hparl_training_acceptance.json`. Its raw-to-clean proof passed before
training began. Repository audio upload is a separate delivery task tracked by
`configs/hparl_delivery_status.json`.

`ctc/run.json` and `ctc/ctc.live.json` are unedited records produced by the
committed CTC training code and owner run wrapper. They record all completed
optimizer updates, dev-loss checkpoint selection, actual runtime/environment,
input/code hashes and the model-specific CTC loss exclusions. Excluded train
or dev-loss examples are logged; this does not remove any clip from the final
shared evaluation cohort. `ctc/checkpoint_sha256.txt` identifies the selected
local inference checkpoint. It does not claim the weights are uploaded yet.

`integration_tests.txt` is the full unittest output from the frozen execution
checkout, run on CPU during primary training. `integration_validation.json`
records the command and successful exit status. These are code/data/scoring
checks, not test-set ASR results.

No WER/CER benchmark or trained-model improvement is claimed here. Other model
families continue through the independent CPU/sequential GPU queue; completed
CTC training does not imply the whole queue has finished. Final model settings,
normalization and scoring protocol must be committed before test inference.
