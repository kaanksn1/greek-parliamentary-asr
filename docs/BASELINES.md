# Final baseline inventory

Updated after the 21 November final queue: every required Greek-applicable
family was evaluated on every original test clip. This replaces the earlier
pilot-only inventory. Scores are generated in
[the final benchmark](../results/benchmark/SUMMARY.md); this file contains no
manually copied result table.

| Required family | Exact selected system / implementation | Training provenance |
|---|---|---|
| MFCC + GMM-HMM | `greek_asr/models/gmm_hmm.py`, `checkpoints/gmm_hmm/model.pt`; character HMM and order-3 LM | Kemal's implementation, group training |
| CTC encoder | Conv + BiLSTM CTC, `greek_asr/models/ctc.py`, `checkpoints/plain_ctc/model.pt` | Kemal's implementation, group training |
| Attention encoder-decoder | LAS-style hybrid CTC/attention, `greek_asr/models/attention.py`, `checkpoints/attention/model.pt` | Adnan's implementation, group training |
| Transducer | LSTM RNN-T, `greek_asr/models/transducer.py`, `checkpoints/transducer/model.pt` | Adnan's implementation, group training |
| wav2vec 2.0 + Greek head | `lighteternal/wav2vec2-large-xlsr-53-greek`, `greek_asr/models/wav2vec.py` | Published Greek fine-tuning; no group training |
| HuBERT + Greek head | `utter-project/mHuBERT-147` plus our trained CTC head, `greek_asr/models/hubert.py`, `checkpoints/hubert_ctc/` | Group Greek CTC training; evaluated checkpoint has an ASR head |
| Whisper | `openai/whisper-small`, `greek_asr/pretrained/whisper.py` | Published multilingual checkpoint |
| Canary | `nvidia/canary-1b-v2`, `greek_asr/pretrained/nemo.py` | Published Greek-capable checkpoint |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3`, same NeMo adapter | Published multilingual v3; standard TDT, CUDA graphs disabled for ROCm |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B`, `greek_asr/pretrained/qwen.py` | Published Greek-capable checkpoint |

The two separate contribution rows are our matched `openai/whisper-small`
fine-tuning on HParl only and HParl + CSS10. Training implementation:
`greek_asr/models/whisper_finetune.py`; actual allocations and dev-selected
steps: [training summary](../results/training/SUMMARY.md). All baseline training
code is committed; central execution does not transfer its authorship.

Exact immutable upstream revisions, model-card links, local checkpoint hashes,
device/batch/decoder settings and development acceptance are in
[`configs/final_test_protocol.json`](../configs/final_test_protocol.json).
The GMM's descriptive bigram typo in that immutable protocol is corrected in
[the erratum](FINAL_PROTOCOL_ERRATA.md); the actual frozen model always used
its trained trigram LM. No weights or decoding were changed.

The published candidate `ntinouldinho/hubert_greek_6gram` has no license/model
card and was not selected. Its historical diagnostic is not our HuBERT result.
The final HuBERT row uses the licensed encoder plus our actual trained Greek
head, never a bare encoder or emotion classifier. The CTC models' very poor
results remain in the table; completed execution is not a claim of good ASR.

Standard families remain baselines even when the group implements/trains them.
Credit implementation/training separately. Running advanced pretrained Qwen,
Canary or Parakeet alone is not a new model contribution. All systems use
identical original test IDs/references and the same scoring policy. Settings
were chosen on development before test; no result was omitted.

Current reproduction: [FINAL_BENCHMARK.md](FINAL_BENCHMARK.md).
Teacher clarification: [TEACHER_UPDATES_2026-11-18_TR.md](TEACHER_UPDATES_2026-11-18_TR.md).
