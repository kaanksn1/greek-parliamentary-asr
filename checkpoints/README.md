# Selected inference checkpoints

These are our actual selected model weights, stored in Git LFS inside this
repository. `configs/selected_checkpoints.json` records source/export hashes,
tensor fingerprints, selection and storage totals. `results/training/` retains
the original training run records. Do not replace the selected weights when
regenerating a published benchmark. The two extended exports and their
original-training-record hashes are separately recorded in
[the extended run record](../results/benchmark_extended/run_record.json)
under the [registered addendum](../configs/extended_ctc_protocol.json).

For six PyTorch checkpoints, only optimizer/scheduler/RNG resume state was
removed to fit the free storage allowance. Every retained tensor's path, dtype,
shape and bytes have identical fingerprints before/after export. The export performs no
quantization, precision conversion, additional training or test-based selection. Files
are inference artifacts; full optimizer resume states remain outside Git.
The selected HuBERT model directory is copied unchanged, including its trained
Greek CTC head and preprocessing/configuration metadata.

| Directory | Contents / origin |
|---|---|
| `gmm_hmm/` | Our trained MFCC + GMM-HMM |
| `plain_ctc/` | Our trained convolutional/BiLSTM CTC model |
| `attention/` | Our trained hybrid CTC/attention encoder-decoder |
| `transducer/` | Our trained RNN-T model |
| `hubert_ctc/` | Our Greek CTC fine-tuning from `utter-project/mHuBERT-147` |
| `whisper_hparl/` | Our HParl-only fine-tuning of `openai/whisper-small` |
| `whisper_mixed/` | Matched fine-tuning with additional CSS10 Greek training data |
| `hubert_ctc_extended/` | Registered longer HuBERT-CTC training; full-DEV-selected checkpoint |
| `plain_ctc_extended/` | Registered longer Plain CTC training; its failed ASR outcome is retained |

Our from-scratch model artifacts accompany the root MIT code license. The
HuBERT-derived artifacts retain the upstream **CC-BY-NC-SA-4.0** terms:
[mHuBERT-147 card](https://huggingface.co/utter-project/mHuBERT-147).
Whisper-derived weights retain the selected model card's **Apache-2.0** terms:
[Whisper-small card](https://huggingface.co/openai/whisper-small).
The exact upstream revisions and license metadata are pinned in
`configs/model_sources.lock.json` / `configs/kemal_model_sources.json` and the
final protocol. `DATA_LICENSE.md` separately covers dataset provenance and
terms; the root code license does not relicense third-party models or data.
