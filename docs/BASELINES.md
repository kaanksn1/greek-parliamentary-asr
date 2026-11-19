# Baseline inventory — draft, not completed runs

Every standard Week 9 method is a baseline. This inventory now follows the
professor's exact baseline email shared on 18 November; the required family
list is clear. Slides can further specify standard recipes. Run every listed
baseline that supports Greek. Unsupported language/head cases need a producer
model-card link and explanation in the README; never use a wrong language.
No row is silently satisfied by merely naming an architecture.
Owners are assigned in `TEAM.md`; assignment is not evidence of completed work.

| Family | Candidate / implementation | Owner | Current status |
|---|---|---|---|
| MFCC (or log-mel) + GMM-HMM | Own PyTorch grapheme GMM-HMM: MFCC+Δ+ΔΔ, Viterbi training, char n-gram LM decoder (`greek_asr/models/gmm_hmm.py`) | Kemal | Code, dev tuning and tests ready; train-pilot code path checked; budget measurement and full training pending |
| Plain CTC system | Conv + BiLSTM encoder + CTC trained from scratch (`greek_asr/models/ctc.py`) | Kemal | Code and tests ready; train-pilot code path checked; measured `max_steps` and full training pending |
| Attention encoder-decoder, including hybrid CTC/attention | Standard supervised attention ASR baseline | Adnan | Code and train-pilot GPU/CLI checks verified; owner completed final review fixes; full training pending |
| RNN-T transducer | Standard supervised RNN-T; keep distinct from pretrained Parakeet TDT | Adnan | Pure-PyTorch loss mathematically and GPU/CLI verified; owner corrected padding/config; full training pending |
| wav2vec 2.0 ASR | `lighteternal/wav2vec2-large-xlsr-53-greek` (`greek_asr.infer --backend wav2vec`) | Kemal | Real checkpoint transcribed three train-pilot clips on CPU; GPU dev/test inference pending |
| HuBERT ASR | Published `ntinouldinho/hubert_greek_6gram` (Greek CTC head, no license/model card) via `--backend hubert`; own CTC fine-tuning of `utter-project/mHuBERT-147` (`greek_asr/models/hubert.py`) | Kemal | Published candidate transcribed three train-pilot clips on CPU, licensing decision pending; own-training recipe and tests ready, budget and full training pending. See `KEMAL_BASELINES.md` |
| Whisper | `openai/whisper-small` multilingual baseline | Serdar | New isolated ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Canary | `nvidia/canary-1b-v2` | Serdar | New isolated NeMo/ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | Serdar | New isolated NeMo/ROCm runtime verified on three train-pilot clips; NVIDIA CUDA Graph acceleration disabled, standard greedy TDT retained; full benchmark pending |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` | Serdar | New isolated ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Own contribution | Same selected starting Whisper checkpoint with our fine-tuning; HParl-only/mixed-source controls | Adnan | PR #5 code and corrected train-pilot GPU step verified; full-data protocol/budget/training pending; report separately |

Serdar runs all full training/inference centrally; model owners provide and fix
their own code/configs. Ozan owns the common evaluation/comparison, Samed the
shared data. A standard toolkit recipe is acceptable with attribution, exact
versions and reproducible committed run/training code; a bespoke architecture
is not an additional requirement.

Checkpoint metadata, when fetched, are recorded in `configs/model_sources.lock.json`.
That file is the immutable metadata-fetch snapshot: its status fields describe
the fetch time, before the later Whisper runtime check. Current runtime evidence
is in the table above and `results/runtime_smoke/` (including the verified
owner Whisper/Qwen/Canary/Parakeet records). See `RUNTIME_ENVIRONMENT.md` and
`NEMO_RUNTIME.md` for the tested environments and execution settings.
Pinning a revision does not certify inference, training compatibility or absence
of pretraining overlap. Model cards' declared training data must be reviewed.

Whisper must be multilingual (`openai/whisper-small`, not a `.en` model);
Parakeet must be multilingual v3, not English-only v2. The professor permits
Qwen3-ASR 0.6B or 1.7B; our selected 0.6B is one of those exact choices.
Kemal's 19 November search found one published HuBERT checkpoint with a Greek
ASR head, without license or model card, and prepared both its standard
inference and an own HuBERT-CTC training recipe; candidates and reasons are in
`KEMAL_BASELINES.md` and `configs/kemal_model_sources.json`. A bare
HuBERT/wav2vec encoder cannot supply a transcription.

A checkpoint fine-tuned and published by someone else is still an off-the-shelf
baseline. If we train GMM-HMM/CTC or an ASR head ourselves, the standard method
remains a baseline while the group receives credit for its actual training
work. Label this provenance separately; do not claim a new architecture.
Any own fine-tuned version and its off-the-shelf starting model use the same
frozen test/reference/scoring policy and clearly separate result rows.

Primary sources:

- https://huggingface.co/lighteternal/wav2vec2-large-xlsr-53-greek
- https://huggingface.co/openai/whisper-small
- https://huggingface.co/nvidia/canary-1b-v2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://github.com/QwenLM/Qwen3-ASR

Advanced pretrained models remain baselines. Plain CTC and pretrained wav2vec,
or a trained transducer and pretrained Parakeet, are not automatically merged
into one fulfilled requirement. Applicability questions must be settled against
the slides and professor's instructions, not by selecting only easy rows.
