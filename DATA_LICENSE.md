# Dataset provenance and license

Main source: **HParl: Hellenic Parliamentary Speech Corpus**, version 1.0.0,
by Georgios Paraskevopoulos, Georgios Rouvalis and Theodoros Kouzelis,
Institute for Language and Speech Processing, Athena Research Center.

- Official record: https://inventory.clarin.gr/corpus/1602
- Persistent identifier: http://hdl.handle.net/11500/CLARIN-EL-0000-0000-7665-A
- Producer-declared license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Resource citation: Paraskevopoulos, Georgios; Rouvalis, Georgios; Kouzelis,
  Theodoros (2023, September 29). HParl: Hellenic Parliamentary Speech Corpus.
  Version 1.0.0. CLARIN:EL.

Keep the authors, source, license link and a description of our actual changes
with any redistributed subset. Non-commercial license conditions remain in
force. Our MIT code license does not relicense the recordings or transcripts.

The selected 23,615 original RAW training clips, all original dev/test clips
and their source metadata are delivered through Git LFS. The cleaned training
manifest keeps 23,602 of those clips, removing nine extreme transcription-speed
cases and four duplicate-audio records; original audio bytes remain intact.
The selected training identities and changes are recorded in
`configs/training_data_selection.json` and the data cleaning logs. The small
startup pilot remains historical provenance in `data/raw/pilot/`.
No new human transcription is claimed. The non-commercial condition is
preserved for both the redistributed subset and its derived manifest.

## Additional training source: CSS10 Greek v1

CSS10 Greek / Greek Single Speaker Speech Dataset, by Kyubyong Park and
Thomas (Tommy) Mulc. All 1,844 original IEEE-float32 audio clips were acquired,
hash-verified and audited using their actual format tag. All are retained in
the training-only mixture. Their bytes and transcripts remain unchanged.
See `configs/css10_mixture_lineage.json` and the corrected audit reports.

- Producer repository: https://github.com/Kyubyong/css10
- Producer distribution: https://www.kaggle.com/datasets/bryanpark/greek-single-speaker-speech-dataset
- Public distribution metadata checked on 17 November 2026:
  https://www.kaggle.com/api/v1/datasets/view/bryanpark/greek-single-speaker-speech-dataset
  lists version `1` and `licenseName: CC0: Public Domain`.
- Paper: Park and Mulc (2019), *CSS10: A Collection of Single Speaker Speech
  Datasets for 10 Languages*, https://arxiv.org/abs/1903.11269.
- Original audiobook: Penelope Delta, *Παραμύθι χωρίς όνομα*, read by
  Rapunzelina: https://librivox.org/paramythi-horis-onoma-by-penelope-delta/.

Use the distributor's CC0 declaration for the dataset; the producer GitHub
repository's Apache-2.0 license is not the evidence for the audio license.
Retain the acquired release's license/source records and attribution. The
LibriVox page independently describes US public-domain status and asks users
elsewhere to check local status; do not turn that statement into a blanket
worldwide legal claim. Any combined redistribution must still preserve HParl's
CC BY-NC 4.0 obligations. Our MIT license covers our own code only.
