"""Execute exactly one planned ASR system; references never enter test decoders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_test(plan):
    """Check immutable inputs/code without passing references to a decoder."""
    from greek_asr.infer import _test_protocol
    protocol = ROOT / plan['protocol']
    _test_protocol(protocol, ROOT / plan['test_manifest'], ROOT)
    fixed = read(protocol)
    paths = {'configs/benchmark_plan.json': fixed['benchmark_plan_sha256'],
             'configs/selected_checkpoints.json': fixed['selected_checkpoints_sha256'],
             'configs/selected_decoders.json': fixed['selected_decoders_sha256'],
             'configs/dev_batch_resources.json': fixed['dev_batch_resources_sha256'],
             'data/manifests/test.freeze.json': fixed['test_freeze_sha256'],
             plan['test_references']: fixed['test_references_sha256'],
             **fixed['source_locks'], **fixed['inference_code_sha256']}
    for relative, expected in paths.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('frozen input/code changed: ' + relative)
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'greek_asr', 'scripts', 'configs'], cwd=ROOT).returncode:
        raise ValueError('runtime code/configs changed after protocol commit')
    return fixed


def verify_checkpoint(job):
    if 'checkpoint' not in job:
        return
    selected = read(ROOT / 'configs/selected_checkpoints.json')['exports'][job['name']]
    if 'files' in selected:
        expected = {str(Path(selected['path']) / name): value for name, value in selected['files'].items()}
    else:
        expected = {selected['path']: selected['export_sha256']}
    for relative, wanted in expected.items():
        if digest(ROOT / relative) != wanted:
            raise ValueError('selected inference weights changed: ' + relative)


def tune_gmm(checkpoint, manifest, references, output):
    import torch
    from greek_asr.models import gmm_hmm, common
    from greek_asr.infer import load_manifest
    from greek_asr.evaluation import score
    from greek_asr.scoring import load_jsonl

    records = load_manifest(manifest, ROOT, 'dev')
    refs, _ = load_jsonl(references, 'reference')
    model = gmm_hmm.GmmHmmModel.load(checkpoint, torch.device('cpu'))
    frontend = gmm_hmm.MfccFrontend(model.features)
    started = time.perf_counter()
    with torch.inference_mode():
        emissions = [model.gmm.state_loglik(frontend(common.load_waveform(row['audio'],
                     target_sample_rate=model.features.sample_rate))) for row in records]
    config = read(ROOT / 'configs/kemal/gmm_hmm.json')['tuning']
    trials, best_predictions = [], None
    for lm_weight in config['lm_weights']:
        for penalty in config['insertion_penalties']:
            settings = gmm_hmm.DecoderSettings(lm_weight=lm_weight, insertion_penalty=penalty,
                                               space_skip_logprob=config.get('space_skip_logprob', 0.))
            decoder = model.decoder(settings)
            predictions = {row['id']: decoder.decode(loglik) for row, loglik in zip(records, emissions)}
            metrics, _ = score(refs, predictions)
            candidate = {'settings': asdict(settings), 'wer': metrics['wer'], 'cer': metrics['cer']}
            trials.append(candidate)
            if best_predictions is None or (candidate['wer'], candidate['cer']) < (best['wer'], best['cer']):
                best, best_predictions = candidate, predictions
            print(json.dumps({'gmm_dev_candidate': candidate}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps({'id': row['id'], 'text': best_predictions[row['id']]}, ensure_ascii=False) + '\n')
    selection = {'gmm_hmm': best['settings'], 'source': 'fixed seeded DEV calibration only',
                 'dev_manifest_sha256': digest(manifest), 'scoring_policy': 'greek-asr-nfc-casefold-punctuation-v1',
                 'trials': trials}
    (ROOT / 'configs/selected_decoders.json').write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    result = {'status': 'complete', 'split': 'dev', 'records': len(records), 'completed_records': len(records),
              'inference_seconds': time.perf_counter() - started, 'tuning': selection,
              'test_used': False, 'emissions_cached_once': True}
    (output / 'run.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--phase', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read(ROOT / 'configs/benchmark_plan.json')
    job = next(row for row in plan['jobs'] if row['name'] == args.job)
    manifest = ROOT / plan[args.phase + '_manifest']
    references = ROOT / plan[args.phase + '_references']
    protocol = ROOT / plan['protocol'] if args.phase == 'test' else None
    if protocol is not None:
        fixed = validate_test(plan)
        if job not in fixed['systems'] or digest(ROOT / 'configs/benchmark_plan.json') != fixed['benchmark_plan_sha256']:
            raise ValueError('system settings differ from final committed protocol')
        verify_checkpoint(job)
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if job['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required, no CPU fallback')
        torch.cuda.set_per_process_memory_fraction(.85)
    checkpoint = ROOT / job['checkpoint'] if 'checkpoint' in job else None
    started = time.perf_counter()
    common = dict(project_root=ROOT, split=args.phase, output_dir=args.output,
                  device=job['device'], allow_test=args.phase == 'test', test_protocol=protocol)
    if job['kind'] == 'pretrained':
        from greek_asr.infer import run
        result = run(manifest=manifest, backend_name=job['backend'], split=args.phase, output=args.output,
                     project_root=ROOT, device=job['device'], seed=42, allow_test=args.phase == 'test',
                     test_protocol=protocol, batch_size=job['batch_size'])
    elif job['kind'] == 'gmm' and args.phase == 'dev':
        result = tune_gmm(checkpoint, manifest, references, args.output)
    else:
        module_name = {'gmm': 'gmm_hmm', 'whisper': 'whisper_finetune'}.get(job['kind'], job['kind'])
        module = importlib.import_module('greek_asr.models.' + module_name)
        options = dict(common)
        if job['kind'] in ('ctc', 'hubert', 'whisper'):
            options['batch_size'] = job['batch_size']
        if job['kind'] == 'attention':
            options.update(decode='greedy', beam_size=1)
        if job['kind'] == 'gmm':
            options['settings'] = module.DecoderSettings(**read(ROOT / 'configs/selected_decoders.json')['gmm_hmm'])
        result = module.predict(checkpoint, manifest, **options)
    from greek_asr.infer import _environment
    result.update(job=job, worker_seconds_including_load=time.perf_counter()-started,
                  worker_code_sha256=digest(Path(__file__)), seed=42, environment=_environment(),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  manifest_sha256=digest(manifest), predictions_sha256=digest(args.output / 'predictions.jsonl'))
    (args.output / 'run.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'completed': args.job, 'records': result.get('completed_records')}), flush=True)


if __name__ == '__main__':
    main()
