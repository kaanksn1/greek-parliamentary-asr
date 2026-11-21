# Completed training and DEV-selected checkpoints

Generated from unmodified original run.json records; do not edit numbers manually.

| System | Training records | Updates / EM | Selected update | Effective batch | Hours |
|---|---:|---:|---:|---:|---:|
| attention | 23602 | 3000 | 3000 | 16 | 0.925 |
| gmm_hmm | 23600 | 25 EM | fixed EM allocation | n/a | 3.154 |
| hubert_ctc | 23601 | 1500 | 1500 | 32 | 0.817 |
| plain_ctc | 23532 | 2300 | 2300 | 32 | 0.184 |
| transducer | 23602 | 3000 | 3000 | 8 | 4.574 |
| whisper_hparl | 23602 | 1600 | 500 | 16 | 3.774 |
| whisper_mixed | 25446 | 1600 | 500 | 16 | 3.167 |

Every seed is 42. Checkpoint selection uses full-development loss; both Whisper conditions select update 500 after completing 1,600 updates.
Model-specific training exclusions are recorded explicitly in each original run/config. Test inference retains all clips. GMM iteration 0 is initialization, followed by 25 EM iterations; its character LM has order 3.
Per-process training wall times; CPU GMM overlapped GPU jobs. The sum is not project elapsed wall time.
Dev losses across architectures are different objectives and should not be compared as model accuracy.
