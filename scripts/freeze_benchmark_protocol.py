"""Freeze all systems and scoring only after complete DEV calibration acceptance."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from greek_asr.evaluation import POLICY, source_sha256
    read = lambda relative: json.loads((root/relative).read_text(encoding='utf-8-sig'))
    plan, selection = read('configs/benchmark_plan.json'), read('configs/evaluation_dev_selection.json')
    evidence = {}
    for job in plan['jobs']:
        folder = root/'results/dev_calibration'/job['name']
        score, run = read(folder/'score.json'), read(folder/'run.json')
        if run['status'] != 'complete' or score['examples'] != selection['records']:
            raise ValueError('every planned system must complete the identical DEV cohort')
        if run['job'] != job or run['manifest_sha256'] != selection['manifest_sha256']:
            raise ValueError('selected resource/model settings differ from accepted DEV run')
        if score['reference_sha256'] != selection['references_sha256'] or score['scorer_code_sha256'] != source_sha256():
            raise ValueError('DEV scoring/input identity differs')
        evidence[job['name']] = {'score_sha256': sha(folder/'score.json'), 'run_sha256': sha(folder/'run.json'),
                                 'wer': score['wer'], 'cer': score['cer'], 'records': score['examples']}
    freeze = read('data/manifests/test.freeze.json')
    manifest = root/plan['test_manifest']
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if sha(manifest) != freeze['test_manifest_sha256'] or sha(root/plan['test_references']) != freeze['references_sha256']:
        raise ValueError('original frozen test identity changed')
    paths = [*sorted((root/'greek_asr').rglob('*.py')), root/'scripts/benchmark_worker.py',
             root/'scripts/run_benchmark_queue.py']
    locks = read('configs/model_sources.lock.json')['models'] + read('configs/kemal_model_sources.json')['models']
    models = {model['model_id']: model for model in locks}
    ids = {'wav2vec': 'lighteternal/wav2vec2-large-xlsr-53-greek',
           'hubert_ctc': 'utter-project/mHuBERT-147', 'whisper_pretrained': 'openai/whisper-small',
           'whisper_hparl': 'openai/whisper-small', 'whisper_mixed': 'openai/whisper-small',
           'canary': 'nvidia/canary-1b-v2', 'parakeet': 'nvidia/parakeet-tdt-0.6b-v3',
           'qwen': 'Qwen/Qwen3-ASR-0.6B'}
    architectures = {'gmm_hmm': 'MFCC + diagonal-mixture GMM, character HMM and bigram LM',
        'plain_ctc': 'from-scratch log-mel BiLSTM encoder + character CTC',
        'attention': read('configs/sequence/attention.json')['architecture'],
        'transducer': read('configs/sequence/transducer.json')['architecture'],
        'wav2vec': 'wav2vec 2.0 XLSR-53 + published Greek CTC head',
        'hubert_ctc': 'mHuBERT-147 encoder + our trained Greek CTC head',
        'whisper_pretrained': 'multilingual Whisper-small encoder-decoder',
        'whisper_hparl': 'multilingual Whisper-small, our HParl fine-tuning',
        'whisper_mixed': 'multilingual Whisper-small, our HParl + CSS10 fine-tuning',
        'canary': 'multilingual Canary FastConformer encoder-decoder',
        'parakeet': 'multilingual Parakeet FastConformer TDT transducer',
        'qwen': 'Qwen3-ASR-0.6B audio encoder and autoregressive decoder'}
    catalog = read('configs/selected_checkpoints.json')['exports']
    details = {}
    for job in plan['jobs']:
        name = job['name']
        item = {'architecture': architectures[name], 'trained_by_group': name in catalog,
                'decoder': 'fixed standard greedy decoding; no external LM' if name != 'gmm_hmm'
                           else read('configs/selected_decoders.json')['gmm_hmm']}
        if name in ids:
            model = models[ids[name]]
            item.update(model_id=model['model_id'], revision=model['revision'],
                        upstream_license=model['license'], model_card=model['source'])
        else:
            item.update(model_id='group-implemented/' + name, revision='selected local checkpoint SHA-256')
        if name in ('canary', 'parakeet'):
            item['decoder'] = 'pinned NeMo checkpoint defaults; Parakeet CUDA graphs disabled for ROCm'
        if name in catalog:
            item['selected_weight_identity'] = catalog[name]
        details[name] = item
    protocol = {'schema': 'p3-final-test-protocol-v1', 'selection_fixed': True,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'selection_uses': 'full-DEV checkpoint losses and DEV calibration only',
        'test_manifest_sha256': sha(manifest), 'test_references_sha256': sha(root/plan['test_references']),
        'test_freeze_sha256': sha(root/'data/manifests/test.freeze.json'), 'test_records': len(records),
        'test_audio_seconds': sum(r['duration_seconds'] for r in records), 'test_cohort_policy': 'every original frozen test clip, no exclusions',
        'scoring_policy': POLICY['name'], 'normalization': POLICY, 'scorer_code_sha256': source_sha256(),
        'metrics': {'primary': 'corpus WER', 'secondary': ['corpus CER including spaces', 'S/I/D counts', 'RTFx', 'empty-hypothesis rate'],
                    'confidence_intervals': '95% percentile speaker-cluster bootstrap, 1000 resamples, seed 42'},
        'systems': plan['jobs'], 'system_details': details, 'expected_systems': len(plan['jobs']), 'seed': 42,
        'benchmark_plan_sha256': sha(root/'configs/benchmark_plan.json'),
        'selected_checkpoints_sha256': sha(root/'configs/selected_checkpoints.json'),
        'selected_decoders_sha256': sha(root/'configs/selected_decoders.json'),
        'dev_batch_resources_sha256': sha(root/'configs/dev_batch_resources.json'),
        'source_locks': {relative: sha(root/relative) for relative in ('configs/model_sources.lock.json', 'configs/kemal_model_sources.json')},
        'inference_code_sha256': {p.relative_to(root).as_posix(): sha(p) for p in paths},
        'dev_acceptance': evidence, 'dev_selection_sha256': sha(root/'configs/evaluation_dev_selection.json'),
        'evaluation_rule': 'one successful complete test prediction pass per frozen system; no test-driven changes',
        'runtime_failure_policy': 'retain failures/partial files; retry infrastructure faults with identical settings; never score partial cohorts',
        'authorizations': 'user assigned all remaining benchmark/report work on 21 November after all seven trainings completed',
        'runtime': {'HSA_ENABLE_DXG_DETECTION': '1', 'MIOPEN_FIND_MODE': 'FAST', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
        'user_working_cutoff': '2026-11-21T21:00:00+03:00', 'professor_exact_cutoff_unknown': True,
        'test_predictions_before_freeze': False}
    destination = root/plan['protocol']
    if destination.exists():
        raise FileExistsError('never replace an existing frozen final protocol')
    destination.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'ready_to_commit', 'systems': len(plan['jobs']), 'test_records': len(records),
                      'protocol_sha256': sha(destination)}))


if __name__ == '__main__':
    main()
