# Ozan Serhan Denizhan — 220911758 — serhandenizhan

Role in `docs/TEAM.md`: common evaluation, comparisons, figures and the
presentation. On 21 November the central final evaluation run and its first
reports were reassigned to Serdar (`AGENTS.md`), so the frozen scorer
(`greek_asr/evaluation.py`), the final 12-system test run and the original
benchmark tables are **not** my work. This report lists only what I did,
with the commits from my own account. It does not claim Samed's data
pipeline, Kemal's or Adnan's models, or Serdar Kaan's integration, central
runs and execution of the extended training on his machine.

## Requirement and code-quality review (21 November)

- Reviewed the repository against the professor's Project 3 announcements
  (baseline list, contribution files, raw/clean data, split rule, metric
  justification, reproducibility checklist) and reported the open items to
  the team: missing contribution report/commits, non-converged CTC and
  HuBERT-CTC baselines, team-internal prompt files, light cleaning evidence,
  Git LFS bandwidth risk and an incorrect confidence-interval sentence.

## Reporting fixes (`621d3720`, `54c753f7`)

- `scripts/summarize_benchmark.py`: the paired Whisper bootstrap reading was
  a fixed "CI spanning zero" sentence although the interval
  [-1.93, -0.06] lies below zero. The generator now derives the sentence from
  the actual bounds; SUMMARY was regenerated with unchanged scores.
- README: documented that the original plain CTC and HuBERT-CTC runs did not
  learn usable transcriptions (WER ≈ 100%, mean 1.0 / 1.5 predicted words,
  full-DEV loss 3.03 / 2.52).
- Removed two superseded team-internal handoff prompts and fixed their
  links; kept the documents that members' own reports cite. Added
  `.DS_Store` to `.gitignore`.

## Pre-registered extended CTC / HuBERT-CTC training (`0a008d11`)

- Designed a follow-up for the two non-converged baselines that changes only
  training length (max_steps, 10% warmup, DEV-evaluation frequency); data,
  model, seed, learning rate and decoder stay the same. The motivation is
  DEV-only (full-DEV loss had not converged).
- `configs/extended_ctc_protocol.json`: frozen before any training or new
  test pass; fixes the schedules, best-full-DEV-loss checkpoint selection, a
  single test pass per system, reporting regardless of outcome, and hashes of
  the code, configs, frozen test inputs and scorer. Budgets were set from the
  measured seconds per update of the original runs to fit the available
  ~6 hours of GPU time.
- `configs/kemal/ctc_extended.json`, `configs/kemal/hubert_ctc_extended.json`
  and two job names in `scripts/run_training_job.py`. Kemal's model code is
  unchanged.
- `scripts/run_extended_ctc.py`: one command that verifies the test freeze,
  trains, exports the DEV-selected checkpoint, decodes the frozen test once,
  scores it with the unchanged frozen scorer, re-verifies the freeze and
  commits; it refuses to run if code or configs differ from the protocol.
- `tests/test_extended_ctc_protocol.py`: only training-length fields may
  change, warmup stays 10%, protocol hashes match, new outputs never overlap
  original rows.
- Serdar Kaan ran the protocol on his GPU and merged it (PR #17). Outcome:
  HuBERT-CTC WER 99.89% → 30.78% (CER 82.18% → 9.54%); plain CTC still
  fails (WER 100.69%), so its problem is not training length.

## Checking and reporting the extended results (`7e4b3592`, PR #19)

- Independently recomputed both extended scores from the saved predictions
  with the frozen scorer (identical), checked the protocol hash, unchanged
  original files and before/after test freeze.
- `scripts/summarize_benchmark.py`: the extended conditions are separate
  rows in `results/benchmark/SUMMARY.md`,
  `results/benchmark_extended/results.csv`, the new
  `results/benchmark/figures/extended_ctc.{png,svg}` figure and the
  presentation brief. The same script rewrites a generated results block in
  the README, so README result numbers come from committed code. The
  original 12 rows, `results/benchmark/results.csv` and existing figures are
  unchanged; `validate_final_benchmark.py` still recomputes all 12 rows.

## Presentation

- Slides and the speaker narrative for the group presentation.

## Tools

I used an AI coding assistant (Claude Code) for parts of the code and text
above; I reviewed, ran and committed every change myself.
