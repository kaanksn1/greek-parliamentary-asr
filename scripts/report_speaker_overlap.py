"""Re-score the saved final test predictions separately for test speakers that do and do not occur in the training data.

HParl's official split is not speaker-disjoint. This post-hoc breakdown uses only the already saved per-utterance
scores of every final system; it decodes nothing, tunes nothing and leaves the frozen test files untouched.
Writes results/speaker_overlap_scores.json and the generated README block between the markers below.
Run: python scripts/report_speaker_overlap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greek_asr.evaluation import speaker_intervals  # noqa: E402

START, END = '<!-- generated-speaker-overlap:start -->', '<!-- generated-speaker-overlap:end -->'
NOTE = ('Computed after the test results were known and never used for any selection or tuning; '
        'it is an additional breakdown of the same single test pass, not a new test.')


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def counts(details):
    totals = dict(we=0, wn=0, ce=0, cn=0)
    for row in details:
        totals['we'] += row['word_counts']['errors']
        totals['wn'] += row['word_counts']['reference_units']
        totals['ce'] += row['character_counts']['errors']
        totals['cn'] += row['character_counts']['reference_units']
    return totals


def subset_score(details, manifest_rows):
    ids = {row['id'] for row in manifest_rows}
    chosen = [row for row in details if row['id'] in ids]
    total = counts(chosen)
    interval = speaker_intervals(chosen, manifest_rows)
    return {'clips': len(chosen), 'wer_percent': 100 * total['we'] / total['wn'], 'cer_percent': 100 * total['ce'] / total['cn'],
            'wer_ci95_percent': [100 * value for value in interval['wer']]}


def compute(root=ROOT):
    root = Path(root)
    summary = json.loads((root / 'results/benchmark/summary.json').read_text(encoding='utf-8'))
    integrity = json.loads((root / 'results/data_integrity.json').read_text(encoding='utf-8'))
    test = read_lines(root / integrity['cohorts']['test']['manifest'])
    train_speakers = {row['speaker_id'] for row in read_lines(root / integrity['cohorts']['hparl_train']['manifest'])}
    seen = [row for row in test if row['speaker_id'] in train_speakers]
    unseen = [row for row in test if row['speaker_id'] not in train_speakers]
    subsets = {}
    for name, rows in (('seen_in_training', seen), ('not_in_training', unseen)):
        subsets[name] = {'speakers': len({row['speaker_id'] for row in rows}), 'clips': len(rows),
                         'hours': sum(row['duration_seconds'] for row in rows) / 3600}
    systems = []
    for row in summary['rows'] + summary.get('extended_rows', []):
        if row.get('status') != 'complete':
            continue
        folder = 'benchmark_extended' if row['role'] == 'baseline_extended_training' else 'benchmark'
        details = read_lines(root / 'results' / folder / row['system'] / 'utterances.jsonl')
        systems.append({'system': row['system'], 'role': row['role'], 'all_wer_percent': row['wer_percent'],
                        'seen_in_training': subset_score(details, seen), 'not_in_training': subset_score(details, unseen)})
    return {'schema': 'p3-speaker-overlap-scores-v1', 'note': NOTE, 'bootstrap': 'speaker clusters, 1000 resamples, seed 42, 95%',
            'test_speakers': subsets['seen_in_training']['speakers'] + subsets['not_in_training']['speakers'],
            'subsets': subsets, 'systems': systems}


def render_block(data):
    seen, unseen = data['subsets']['seen_in_training'], data['subsets']['not_in_training']
    lines = [START, '',
             f"Test speakers that also occur in the selected training data: {seen['speakers']} ({seen['clips']:,} clips, {seen['hours']:.2f} h); "
             f"test speakers absent from training: {unseen['speakers']} ({unseen['clips']:,} clips, {unseen['hours']:.2f} h). "
             f"{data['note']} Intervals are {data['bootstrap']}.", '',
             '| System | WER % seen speakers | WER % unseen speakers | Gap (pts) |', '|---|---:|---:|---:|']
    for item in data['systems']:
        a, b = item['seen_in_training'], item['not_in_training']
        lines.append(f"| {item['system']} | {a['wer_percent']:.2f} [{a['wer_ci95_percent'][0]:.2f}, {a['wer_ci95_percent'][1]:.2f}] | "
                     f"{b['wer_percent']:.2f} [{b['wer_ci95_percent'][0]:.2f}, {b['wer_ci95_percent'][1]:.2f}] | "
                     f"{b['wer_percent'] - a['wer_percent']:+.2f} |")
    lines += ['', END]
    return '\n'.join(lines)


def update_readme(root, data):
    readme = Path(root) / 'README.md'
    content = readme.read_text(encoding='utf-8')
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError('README speaker-overlap markers missing')
    start, end = content.index(START), content.index(END) + len(END)
    readme.write_text(content[:start] + render_block(data) + content[end:], encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    root = parser.parse_args().root.resolve()
    data = compute(root)
    (root / 'results/speaker_overlap_scores.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    update_readme(root, data)
    print(json.dumps({'systems': len(data['systems']), 'seen_clips': data['subsets']['seen_in_training']['clips'],
                      'unseen_clips': data['subsets']['not_in_training']['clips']}))


if __name__ == '__main__':
    main()
