"""Report actual selected cohorts and literal/final-policy overlap; never edit data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greek_asr.evaluation import POLICY, normalize_asr


def rows(relative):
    return [json.loads(line) for line in (ROOT/relative).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(relative):
    return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()


def main():
    selection = json.loads((ROOT/'configs/training_data_selection.json').read_text())
    artifacts = selection['artifacts']
    files = {'raw_hparl_train': artifacts['raw_manifest']['path'], 'hparl_train': artifacts['clean_manifest']['path'],
             'mixed_train': artifacts['mixed_manifest']['path'], 'css10_train':'data/manifests/css10_el_train.jsonl',
             'dev':'data/manifests/dev.jsonl', 'test':'data/manifests/test.jsonl'}
    splits = {name: rows(path) for name,path in files.items()}
    report = {'schema':'p3-benchmark-data-report-v1', 'generator_sha256':sha('scripts/report_benchmark_data.py'),
              'normalization_policy':POLICY['name'], 'computed_after_selection_frozen':True, 'cohorts':{}}
    def identities(records, key):
        return {(r['source'],r[key]) for r in records if r.get(key)}
    for name, records in splits.items():
        report['cohorts'][name] = {'manifest':files[name], 'sha256':sha(files[name]), 'utterances':len(records),
            'hours':sum(r['duration_seconds'] for r in records)/3600,
            'speakers_with_ids':len(identities(records,'speaker_id')),
            'recordings_with_ids':len(identities(records,'source_recording_id')),
            'utterances_without_speaker_id':sum(not r.get('speaker_id') for r in records),
            'utterances_without_recording_id':sum(not r.get('source_recording_id') for r in records)}
    test = splits['test']
    report['training_to_test_overlap'] = {}
    test_refs = rows('data/manifests/references/test.jsonl')
    for name,key in [('hparl_train','clean_references'),('mixed_train','mixed_references')]:
        train = splits[name]
        speaker_overlap = identities(train,'speaker_id') & identities(test,'speaker_id')
        recording_overlap = identities(train,'source_recording_id') & identities(test,'source_recording_id')
        audio_overlap = {r['audio_sha256'] for r in train} & {r['audio_sha256'] for r in test}
        text = {r['reference'] for r in rows(artifacts[key]['path'])}
        normalized = {normalize_asr(t) for t in text}
        literal_matches = [r for r in test_refs if r['reference'] in text]
        norm_matches = [r for r in test_refs if normalize_asr(r['reference']) in normalized]
        report['training_to_test_overlap'][name] = {
            'speakers':len(speaker_overlap), 'recordings':len(recording_overlap),
            'shared_audio_sha256_values':len(audio_overlap),
            'test_clips_with_shared_audio_sha256':sum(r['audio_sha256'] in audio_overlap for r in test),
            'test_clips_with_shared_speaker':sum((r['source'],r.get('speaker_id')) in speaker_overlap for r in test),
            'literal_matching_test_utterances':len(literal_matches),
            'literal_matching_unique_sentences':len({r['reference'] for r in literal_matches}),
            'final_policy_matching_test_utterances':len(norm_matches),
            'final_policy_matching_unique_sentences':len({normalize_asr(r['reference']) for r in norm_matches})}
    report['css10_speaker_note'] = 'Producer single-reader corpus; original manifests encode no speaker/session IDs. All CSS10 is training-only; no invented IDs or cross-source identity claims.'
    report['normalized_empty_test_references_retained'] = sum(not normalize_asr(r['reference']) for r in test_refs)
    report['split_pair_group_overlap'] = {}
    for left,right in [('hparl_train','dev'),('hparl_train','test'),('dev','test')]:
        report['split_pair_group_overlap'][left+'__'+right] = {
            'speakers':len(identities(splits[left],'speaker_id') & identities(splits[right],'speaker_id')),
            'recordings':len(identities(splits[left],'source_recording_id') & identities(splits[right],'source_recording_id'))}
    output = ROOT/'results/data_integrity.json'
    output.write_bytes((json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':
    main()
