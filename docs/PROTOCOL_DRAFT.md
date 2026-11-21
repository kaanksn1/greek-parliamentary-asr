# Experimental protocol — draft

> **Superseded planning record:** pending/acquisition/scoring statements below are historical. Current data: [selected report](DATA_REPORT_TR.md); protocol: [final protocol](../configs/final_test_protocol.json); completed results: [final table](../results/benchmark/SUMMARY.md).

This file is not a frozen final-test protocol. No official-test evaluation is
authorized by the existence of this draft.

1. Acquire and identify the original HParl release, source license and official
   train/development/test sessions. Record original bytes/hashes. Do not use
   a mirror's random split as the official split. Freeze the original test
   data early: source version, ID/session list and audio/reference hashes.
   Do not clean, filter, add/drop, resplit or relabel that test data. Keep
   the second corpus out of the parliamentary test. This data freeze is
   separate from the later model/settings freeze. PR #2 supplies its metadata;
   independent verification against the complete source audio is still pending.
2. Inspect train first, verify audio decoding, duration, silence/clipping,
   duplicates and transcript alignment. Record real corrections/removals.
   Preserve human transcript provenance; model hypotheses are not verified gold.
   Apply the professor's latest split rule: a new split groups related clips by
   speaker/recording/session before seeded assignment. A documented official
   split is permitted with explicit README disclosure of speaker separation.
   HParl's official session split has speaker overlap: do not describe it as
   speaker-disjoint or silently change the already-frozen original test.
   Report per-split speakers/utterances/hours and literal test-sentence matches
   in each actual training condition, apart from normalized unique-text overlaps.
3. Use the largest statistically useful dataset feasible within actual
   hardware/time/cost limits. The professor specified no fixed minimum hours.
   Measure train/dev throughput, VRAM and epoch cost before justifying a subset.
   Keep the frozen test sessions; any train/dev changes are made without
   consulting test labels or scores. The three-clip runtime check is not
   evidence of statistical sufficiency or full-training feasibility.
4. Use the exact family list in the professor's second email shared on 18 November.
   Implement/train every language-applicable standard family and retain ordinary
   results separately. Unsupported language/head cases need model-card evidence
   and an explanation; never report a bare encoder or wrong-language ASR result.
5. Choose the final Greek normalization using train/development examples. The
   initial scorer only performs Unicode NFC, case folding and whitespace
   normalization; accents, digits and punctuation are preserved. CER counts
   internal spaces. This policy is provisional. The second new email explicitly
   requires one common scoring normalization for all systems. Apply that chosen
   policy consistently in memory to references and hypotheses, preserving frozen
   source test audio/reference files. Annotation tags are a dev-only policy
   decision, not grounds for hand-editing test references or tuning from test.
6. Primary candidate metric: corpus WER. Secondary candidate: corpus CER,
   substitution/deletion/insertion counts and clearly defined error slices.
   Both metrics use the same IDs/reference policy for every model. Empty
   reference handling is explicit; WER may exceed 100%.
7. Compare the same pretrained checkpoint before and after our supervised
   in-domain fine-tuning. A smaller/larger training-budget experiment is a
   candidate contribution analysis. The approved two-corpus plan permits an
   additional training source with parliamentary evaluation kept fixed. Keep
   source attribution and real cleaning logs. A target-only versus mixed-source
   training comparison should distinguish data composition from extra hours
   or optimization steps. Select epochs/settings with development.
   The selected additional training source is CSS10 Greek (Kaggle v1);
   acquisition and local quality/duration audit remain pending. It is not
   added to HParl dev/test. See `SECONDARY_SOURCE_TR.md`.
8. Record exact model revision, source/split IDs, seeds, effective batch,
   optimizer, training budget, machine/library versions and selected checkpoint.
9. Freeze the final settings, selection decision, already-locked cohort and scoring
   policy in a committed protocol before test inference/scoring. Archive
   references, predictions and run code; retain selected weights/config where
   practicable. Do not retune from test results.
10. Generate result tables from committed code. Explain every root file and
    directory, data/code licenses, real member contributions and actual limits.

The professor reply shared on 17 November is in `TEACHER_CLARIFICATION_TR.md`.
The user reported the deadline as 21 November 2026, Saturday evening,
Europe/Istanbul; the exact hour is unknown.
Open before full runs: full data acquisition/delivery; class reservation;
working standard recipes; full audio delivery/independent freeze verification;
measured data budget;
Greek scoring policy; each baseline's working recipe. These are pending facts,
not completed tasks. The minimum-hour question is resolved: no fixed number.
The required baseline list and official-split exception are clarified in
`TEACHER_UPDATES_2026-11-18_TR.md`; PR #2 already contains source/clean metadata
and an early freeze, while its full audio delivery remains incomplete. PR #2
reporting/source-metadata corrections were verified at `e5aa488`. Adnan's
PR #5's former GPU/audio/label blockers at `0437631` are fixed at `0c13ce1`;
114 tests, three train-pilot GPU checks and attention/RNN-T CLI checks passed.
The owner completed the subsequent P2 loss/padding/source-lock and documentation
corrections after the user's explicit assignment; all 119 local tests pass.
These checks do not establish full-data training feasibility or benchmark readiness.
