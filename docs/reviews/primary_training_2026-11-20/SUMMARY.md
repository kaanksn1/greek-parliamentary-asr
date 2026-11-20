# Completed primary HParl training

Generated from the retained queue and live run records. These are training progress counts, not ASR benchmark scores.

| Job | Status | Optimizer updates | Elapsed hours |
|---|---|---:|---:|
| gmm | complete | not applicable (EM) | 3.160 |
| ctc | complete | 2300/2300 | 0.193 |
| whisper_hparl | complete | 1600/1600 | 3.780 |
| attention | complete | 3000/3000 | 0.937 |
| hubert | complete | 1500/1500 | 0.825 |
| transducer | complete | 3000/3000 | 4.577 |

Execution commit: d98abe995f85ff634704ec9f60b26ca671a21a29. CPU GMM ran concurrently with the sequential GPU lane; row durations must not be summed as wall-clock duration.
No final test predictions were generated. Final inference-weight delivery and WER/CER evaluation are separate.
