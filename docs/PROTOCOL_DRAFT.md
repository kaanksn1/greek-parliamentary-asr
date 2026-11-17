# Experimental protocol — draft

This file is not a frozen final-test protocol. No official-test evaluation is
authorized by the existence of this draft.

1. Acquire and identify the original HParl release, source license and official
   train/development/test sessions. Record original bytes/hashes. Do not use
   a mirror's random split as the official split. Freeze the original test
   data early: source version, ID/session list and audio/reference hashes.
   Do not clean, filter, add/drop, resplit or relabel that test data. Keep
   the second corpus out of the parliamentary test. This data freeze is
   separate from the later model/settings freeze and is not yet completed.
2. Inspect train first, verify audio decoding, duration, silence/clipping,
   duplicates and transcript alignment. Record real corrections/removals.
   Preserve human transcript provenance; model hypotheses are not verified gold.
3. Use the largest statistically useful dataset feasible within actual
   hardware/time/cost limits. The professor specified no fixed minimum hours.
   Measure train/dev throughput, VRAM and epoch cost before justifying a subset.
   Keep the frozen test sessions; any train/dev changes are made without
   consulting test labels or scores. The three-clip runtime check is not
   evidence of statistical sufficiency or full-training feasibility.
4. Reconcile the baseline matrix with all Week 9 slides. Implement/train the
   applicable standard families and retain their ordinary results separately.
5. Choose the final Greek normalization using train/development examples. The
   initial scorer only performs Unicode NFC, case folding and whitespace
   normalization; accents, digits and punctuation are preserved. CER counts
   internal spaces. This provisional policy must not be mistaken for an
   established Greek benchmark normalization or permission to transform the
   frozen test. Keep source test audio/references unchanged; do not assume an
   exception to the professor's no-test-processing instruction.
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
Open before full runs: full data acquisition; class reservation; deadline;
full slides; actual test-data freeze; selected CSS10 release acquisition/audit; measured data budget;
Greek scoring policy; each baseline's working recipe. These are pending facts,
not completed tasks. The minimum-hour question is resolved: no fixed number.
