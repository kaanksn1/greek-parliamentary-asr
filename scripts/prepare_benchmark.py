"""Export exact inference weights and select a seeded DEV-only calibration cohort."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_rows(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in value), encoding='utf-8')


def tensor_fingerprint(value):
    import torch
    result = hashlib.sha256()
    def visit(item, location=''):
        if isinstance(item, torch.Tensor):
            result.update((location + str(item.dtype) + str(tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], location + '/' + str(key))
        elif isinstance(item, (list, tuple)):
            for index, member in enumerate(item):
                visit(member, location + '/' + str(index))
    visit(value)
    return result.hexdigest()


def export_pt(source, destination):
    import torch
    source_sha = sha(source)
    value = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    removed = []
    for key in ('optimizer_state', 'optimizer', 'scheduler_state', 'scheduler', 'rng_state'):
        if key in value:
            value.pop(key)
            removed.append(key)
    fingerprint = tensor_fingerprint(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(value, destination)
    exported = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
    if tensor_fingerprint(exported) != fingerprint:
        raise ValueError('inference export changed tensor values')
    metadata = {'source_sha256': source_sha, 'export_sha256': sha(destination), 'removed_training_state_keys': removed,
                'inference_tensors_sha256': fingerprint, 'tensor_values_identical': True, 'step': value.get('step')}
    del value, exported
    gc.collect()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-run', type=Path, required=True)
    parser.add_argument('--mixed-run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    import torch
    torch.set_num_threads(2)
    dev = rows(root / 'data/manifests/dev.jsonl')
    references = {row['id']: row['reference'] for row in rows(root / 'data/manifests/references/dev.jsonl')}
    groups = {}
    for row in dev:
        groups.setdefault(row.get('speaker_id') or row.get('source_recording_id') or row['id'], []).append(row)
    generator = random.Random(42)
    chosen = {generator.choice(sorted(group, key=lambda r: r['id']))['id'] for name, group in sorted(groups.items())}
    chosen.update(row['id'] for row in sorted(dev, key=lambda r: (-r['duration_seconds'], r['id']))[:2])
    subset = [row for row in dev if row['id'] in chosen]
    manifest = 'data/manifests/evaluation/dev_calibration_seed42.jsonl'
    refs = 'data/manifests/evaluation/references/dev_calibration_seed42.jsonl'
    write_rows(root / manifest, subset)
    write_rows(root / refs, [{'id': row['id'], 'reference': references[row['id']]} for row in subset])
    from greek_asr.evaluation import normalize_asr, POLICY
    write(root / 'configs/evaluation_dev_selection.json', {
        'schema': 'p3-dev-calibration-v1', 'seed': 42, 'rule': 'one seeded clip per DEV speaker plus the two longest DEV clips',
        'selection_uses': 'DEV metadata only, before any test ASR', 'manifest': manifest, 'references': refs,
        'manifest_sha256': sha(root / manifest), 'references_sha256': sha(root / refs), 'records': len(subset),
        'audio_seconds': sum(row['duration_seconds'] for row in subset), 'speakers': len(groups),
        'full_dev_manifest_sha256': sha(root / 'data/manifests/dev.jsonl'), 'normalization': POLICY,
        'normalization_examples': [{'id': row['id'], 'before': references[row['id']],
                                    'after': normalize_asr(references[row['id']])} for row in subset[:12]]})
    sources = {
        'gmm_hmm': args.primary_run / 'gmm/model.pt',
        'plain_ctc': args.primary_run / 'ctc/checkpoint.best.pt',
        'attention': args.primary_run / 'attention/checkpoint.best.pt',
        'transducer': args.primary_run / 'transducer/checkpoint.best.pt',
        'whisper_hparl': args.primary_run / 'whisper_hparl/checkpoint.best.pt',
        'whisper_mixed': args.mixed_run / 'checkpoint.best.pt',
    }
    exports = {}
    for name, source in sources.items():
        destination = root / 'checkpoints' / name / 'model.pt'
        exports[name] = {'path': destination.relative_to(root).as_posix(), **export_pt(source, destination)}
        run = read(source.parent / 'run.json')
        exports[name]['training_code_commit'] = run.get('code_commit')
        exports[name]['selection'] = 'best full-DEV loss' if name != 'gmm_hmm' else 'completed fixed EM allocation'
        exports[name]['training_run_sha256'] = sha(source.parent / 'run.json')
        run_target = root / f'results/training/{name}/run.json'
        run_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / 'run.json', run_target)
        print(json.dumps({'exported': name, 'bytes': destination.stat().st_size}), flush=True)
    hubert_source = args.primary_run / 'hubert/checkpoint-best'
    hubert_target = root / 'checkpoints/hubert_ctc'
    shutil.copytree(hubert_source, hubert_target)
    exports['hubert_ctc'] = {'path': 'checkpoints/hubert_ctc', 'files': {
        p.relative_to(hubert_target).as_posix(): sha(p) for p in sorted(hubert_target.rglob('*')) if p.is_file()},
        'selection': 'best full-DEV loss', 'training_run_sha256': sha(args.primary_run / 'hubert/run.json')}
    run_target = root / 'results/training/hubert_ctc/run.json'
    run_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.primary_run / 'hubert/run.json', run_target)
    weights_bytes = sum(p.stat().st_size for p in (root / 'checkpoints').rglob('*') if p.is_file())
    audio_gib = read(root / 'configs/training_data_selection.json')['storage']['all_audio_unique_lfs_gib']
    if audio_gib + weights_bytes / 1024**3 > 10:
        raise ValueError('original audio plus selected inference weights exceed the authorized 10 GiB storage cap')
    write(root / 'configs/selected_checkpoints.json', {'schema': 'p3-selected-inference-checkpoints-v1',
        'selection_uses': 'training full-DEV losses only; no test predictions', 'exports': exports,
        'inference_weight_bytes': weights_bytes, 'audio_plus_weights_gib': audio_gib + weights_bytes / 1024**3,
        'paid_storage_authorized': False})
    jobs = [
        {'name': 'gmm_hmm', 'kind': 'gmm', 'batch_size': 1, 'role': 'baseline', 'device': 'cpu'},
        {'name': 'plain_ctc', 'kind': 'ctc', 'batch_size': 16, 'role': 'baseline'},
        {'name': 'attention', 'kind': 'attention', 'batch_size': 1, 'role': 'baseline'},
        {'name': 'transducer', 'kind': 'transducer', 'batch_size': 1, 'role': 'baseline'},
        {'name': 'wav2vec', 'kind': 'pretrained', 'backend': 'wav2vec', 'batch_size': 1, 'role': 'baseline'},
        {'name': 'hubert_ctc', 'kind': 'hubert', 'batch_size': 4, 'role': 'baseline'},
        {'name': 'whisper_pretrained', 'kind': 'pretrained', 'backend': 'whisper', 'batch_size': 4, 'role': 'baseline'},
        {'name': 'canary', 'kind': 'pretrained', 'backend': 'canary', 'batch_size': 2, 'role': 'baseline'},
        {'name': 'parakeet', 'kind': 'pretrained', 'backend': 'parakeet', 'batch_size': 4, 'role': 'baseline'},
        {'name': 'qwen', 'kind': 'pretrained', 'backend': 'qwen', 'batch_size': 4, 'role': 'baseline'},
        {'name': 'whisper_hparl', 'kind': 'whisper', 'batch_size': 4, 'role': 'contribution'},
        {'name': 'whisper_mixed', 'kind': 'whisper', 'batch_size': 4, 'role': 'contribution'},
    ]
    for job in jobs:
        job.setdefault('device', 'cuda')
        if job['name'] in exports:
            job['checkpoint'] = exports[job['name']]['path']
    write(root / 'configs/benchmark_plan.json', {'schema': 'p3-complete-benchmark-v1', 'seed': 42,
        'dev_manifest': manifest, 'dev_references': refs, 'test_manifest': 'data/manifests/test.jsonl',
        'test_references': 'data/manifests/references/test.jsonl', 'protocol': 'configs/final_test_protocol.json',
        'jobs': jobs, 'user_working_cutoff': '2026-11-21T21:00:00+03:00', 'professor_exact_deadline_unknown': True,
        'batch_selection': 'resource settings verified on seeded DEV, same frozen decoding at test'})
    print(json.dumps({'status': 'prepared', 'dev_records': len(subset), 'inference_weight_bytes': weights_bytes,
                      'audio_plus_weights_gib': audio_gib + weights_bytes / 1024**3, 'systems': len(jobs)}), flush=True)


if __name__ == '__main__':
    main()
