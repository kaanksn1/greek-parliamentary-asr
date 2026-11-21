# Descriptive erratum, 21 November

The immutable pre-test protocol `configs/final_test_protocol.json` at
`0ff3196c` describes the GMM system as having a **bigram** LM. That descriptive
word is incorrect: the selected GMM checkpoint actually contains a **trigram
(order-3) character LM**, trained solely on training references with Witten-Bell
interpolation and add-one unigram smoothing. This is explicit in the original
training `run.json` (`lm.order=3`, `training.lm_order=3`), selected model state,
training code and configuration.

No decoder, setting, weight, input, source lock, scorer or frozen protocol byte
was changed. The original protocol hash and selected checkpoint hash still
identify the actual same trigram system used for development and test. This
note corrects the description rather than rewriting a pre-test freeze.

The first preparation script also serialized training `run.json` objects into
canonical JSON, preserving their values but changing their byte hashes. The
final archive now copies each of the seven original files byte for byte and
verifies its SHA-256 against the already frozen selected-checkpoint catalog.
`results/training/original_record_verification.json` records both semantic
identity with the earlier copy and byte identity with the original source.
`scripts/prepare_benchmark.py` now preserves those original bytes. This change
does not affect model selection, exported tensors, weights or evaluation.
