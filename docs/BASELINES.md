# Baseline inventory — draft, not completed runs

Every standard Week 9 method is a baseline. This inventory comes from the
professor's announcement; the full Week 9 slides still need reconciliation.
No row is silently satisfied by merely naming an architecture.
Owners are assigned in `TEAM.md`; assignment is not evidence of completed work.

| Family | Candidate / implementation | Owner | Current status |
|---|---|---|---|
| MFCC + GMM-HMM | Train a conventional system with a documented lexicon/decoder recipe | Kemal | Recipe and environment pending |
| Plain CTC system | Small supervised encoder + CTC trained on the selected train split | Kemal | Implementation/training pending |
| Attention encoder-decoder | Standard supervised attention ASR baseline | Adnan | Architecture/recipe pending |
| Transducer | Standard supervised transducer; keep distinct from pretrained Parakeet | Adnan | Architecture/recipe pending |
| wav2vec 2.0 ASR | `lighteternal/wav2vec2-large-xlsr-53-greek` | Kemal | Greek CTC ASR card verified; runtime pending |
| HuBERT ASR | Must have a Greek-compatible ASR head, not emotion classification or a bare encoder | Kemal | Suitable checkpoint or own standard HuBERT-CTC training pending |
| Whisper | `openai/whisper-small` multilingual baseline | Serdar | New isolated ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Canary | `nvidia/canary-1b-v2` | Serdar | New isolated NeMo/ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | Serdar | New isolated NeMo/ROCm runtime verified on three train-pilot clips; NVIDIA CUDA Graph acceleration disabled, standard greedy TDT retained; full benchmark pending |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` | Serdar | New isolated ROCm runtime verified on three train-pilot clips; full benchmark pending |
| Own contribution | Same selected starting Whisper checkpoint with our fine-tuning; HParl-only/mixed-source controls | Adnan | Protocol/budget/training pending; report separately |

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
