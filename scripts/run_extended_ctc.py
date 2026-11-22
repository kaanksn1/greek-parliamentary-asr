"""Pre-registered extended-training addendum for plain CTC and mHuBERT-147 + CTC.

The original plain CTC and HuBERT-CTC runs had not converged on full DEV loss
(results/training/*/run.json). This addendum gives the SAME models, data, seed
and optimizer longer training only. Everything is fixed in the committed
configs/extended_ctc_protocol.json BEFORE training and before any new test
inference:

  for each system: train (scripts/run_training_job.py)
                   -> export the best full-DEV-loss checkpoint (no test use)
                   -> one full-test inference with the dev-measured batch size
                   -> score with the unchanged frozen scorer/protocol

Original runs, checkpoints and benchmark rows are never modified; new outputs
go to *_extended paths and are reported as separate rows whatever the outcome.

Usage on the training machine (Transformers/ROCm environment):
    python scripts/run_extended_ctc.py --output outputs/extended_ctc --commit --push
Protocol (re)generation, before committing, is done once with --freeze-protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'configs/extended_ctc_protocol.json'
FINAL_PROTOCOL = 'configs/final_test_protocol.json'
TEST_MANIFEST = 'data/manifests/test.jsonl'
TEST_REFERENCES = 'data/manifests/references/test.jsonl'
# Order fixed in advance: HuBERT first, because it was still improving on DEV.
SYSTEMS = {
    'hubert_ctc_extended': {'job': 'hubert_extended', 'kind': 'hubert', 'config': 'configs/kemal/hubert_ctc_extended.json',
                            'original': 'hubert_ctc', 'checkpoint': 'checkpoints/hubert_ctc_extended', 'batch_size': 4},
    'plain_ctc_extended': {'job': 'ctc_extended', 'kind': 'ctc', 'config': 'configs/kemal/ctc_extended.json',
                           'original': 'plain_ctc', 'checkpoint': 'checkpoints/plain_ctc_extended/model.pt', 'batch_size': 16},
}
LOCKED_CODE = ['greek_asr/audio.py', 'greek_asr/infer.py', 'greek_asr/evaluation.py', 'greek_asr/scoring.py',
               'greek_asr/models/common.py', 'greek_asr/models/ctc.py', 'greek_asr/models/ctc_utils.py',
               'greek_asr/models/hubert.py', 'scripts/run_training_job.py', 'scripts/run_extended_ctc.py',
               'scripts/prepare_benchmark.py']
TRAINING_LENGTH_FIELDS = {'max_steps', 'warmup_steps', 'dev_eval_every', '_comment'}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def original_config(spec: dict) -> Path:
    return ROOT / spec['config'].replace('_extended', '')


def config_changes(spec: dict) -> dict:
    """Fields that differ from the original config; only training-length fields are allowed."""
    old, new = read(original_config(spec)), read(ROOT / spec['config'])
    keys = set(old) | set(new)
    changed = {key: {'original': old.get(key), 'extended': new.get(key)} for key in sorted(keys) if old.get(key) != new.get(key)}
    illegal = set(changed) - TRAINING_LENGTH_FIELDS
    if illegal:
        raise ValueError(f"{spec['config']}: only training-length fields may change, not {sorted(illegal)}")
    return {key: value for key, value in changed.items() if key != '_comment'}


def build_protocol() -> dict:
    final = read(ROOT / FINAL_PROTOCOL)
    systems = {}
    for name, spec in SYSTEMS.items():
        original_run = read(ROOT / 'results/training' / spec['original'] / 'run.json')
        systems[name] = {
            **spec, 'config_sha256': sha(ROOT / spec['config']), 'changes_from_original': config_changes(spec),
            'original_run_dev_history': [{k: row[k] for k in ('optimizer_update', 'dev_loss')}
                                         for row in original_run['dev_history']],
            'device': 'cuda', 'decode': 'greedy CTC best path (unchanged)',
        }
    return {
        'schema': 'p3-extended-ctc-protocol-v1', 'status': 'frozen_before_training_and_test',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': ('Longer training for the two non-converged group-trained CTC baselines. Model, data, seed, '
                    'optimizer, learning rate and decoding are unchanged; only training length differs.'),
        'motivation_source': 'DEV only: full-DEV CTC loss of the original runs had not converged (original_run_dev_history).',
        'system_order': list(SYSTEMS), 'systems': systems,
        'selection_rule': 'checkpoint with the lowest full-DEV CTC loss among the scheduled DEV evaluations (same rule as originals)',
        'test_rule': 'each new system is decoded exactly once on all 8,679 frozen test clips after its own training finishes',
        'reporting_rule': ('new rows are reported as separate *_extended rows regardless of outcome; original rows, runs, '
                           'checkpoints and scores stay byte-identical and remain in every table'),
        'scoring': {'protocol': FINAL_PROTOCOL, 'protocol_sha256': sha(ROOT / FINAL_PROTOCOL),
                    'scoring_policy': final['scoring_policy'], 'scorer_code_sha256': final['scorer_code_sha256']},
        'test_manifest_sha256': sha(ROOT / TEST_MANIFEST), 'test_references_sha256': sha(ROOT / TEST_REFERENCES),
        'test_freeze_sha256': sha(ROOT / 'data/manifests/test.freeze.json'),
        'locked_code_sha256': {relative: sha(ROOT / relative) for relative in LOCKED_CODE},
        'measured_cost_basis': {
            'hubert_seconds_per_update': 1.2967, 'hubert_dev_eval_seconds': 300,
            'ctc_seconds_per_update': 0.2553, 'ctc_dev_eval_seconds': 25,
            'source': 'original run.json records on the same AMD RX 7800 XT machine',
            'expected_hours': {'hubert_ctc_extended': 3.6, 'plain_ctc_extended': 1.2, 'test_inference_and_checks': 0.3},
        },
        'storage': 'new selected weights add about 0.39 GiB of Git LFS, within the free 10 GiB cap',
    }


def check_protocol() -> dict:
    relative = PROTOCOL.relative_to(ROOT).as_posix()
    committed = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT, capture_output=True).stdout
    if committed != PROTOCOL.read_bytes():
        raise ValueError('commit configs/extended_ctc_protocol.json before running')
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'greek_asr', 'scripts', 'configs'], cwd=ROOT).returncode:
        raise ValueError('runtime code/configs differ from the committed protocol state')
    protocol = read(PROTOCOL)
    for relative, expected in protocol['locked_code_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('locked code changed after protocol: ' + relative)
    for name, spec in protocol['systems'].items():
        if sha(ROOT / spec['config']) != spec['config_sha256']:
            raise ValueError('extended config changed after protocol: ' + spec['config'])
    for key, relative in (('test_manifest_sha256', TEST_MANIFEST), ('test_references_sha256', TEST_REFERENCES),
                          ('test_freeze_sha256', 'data/manifests/test.freeze.json')):
        if sha(ROOT / relative) != protocol[key]:
            raise ValueError('frozen test input changed: ' + relative)
    if sha(ROOT / FINAL_PROTOCOL) != protocol['scoring']['protocol_sha256']:
        raise ValueError('final scoring protocol changed')
    return protocol


def runtime_env(cpu: bool = False) -> dict:
    return dict(os.environ, HSA_ENABLE_DXG_DETECTION='0' if cpu else '1', MIOPEN_FIND_MODE='FAST',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')


def run_logged(command: list[str], log: Path, cpu: bool = False) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'running': command[1:4], 'log': str(log)}), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=runtime_env(cpu), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if '"optimizer_updates"' in line or 'dev ' in line or 'completed' in line:
                print(line.rstrip(), flush=True)
        if process.wait():
            raise RuntimeError(f'command failed ({process.returncode}); see {log}')


def export_checkpoint(name: str, spec: dict, train_dir: Path) -> dict:
    destination = ROOT / spec['checkpoint']
    if destination.exists():
        raise FileExistsError(destination)
    if spec['kind'] == 'ctc':
        sys.path.insert(0, str(ROOT / 'scripts'))
        from prepare_benchmark import export_pt
        metadata = export_pt(train_dir / 'checkpoint.best.pt', destination)
    else:
        source = train_dir / 'checkpoint-best'
        shutil.copytree(source, destination)
        metadata = {'files': {path.name: sha(path) for path in sorted(destination.iterdir())},
                    'copied_unchanged_from': 'checkpoint-best',
                    'step': read(destination / 'greek_asr_ctc.json')['step']}
    metadata.update(path=spec['checkpoint'], selection='best full-DEV loss', training_run_sha256=sha(train_dir / 'run.json'))
    return metadata


def predict(name: str) -> None:
    """Child process: one full-test decode of an exported extended checkpoint."""
    import importlib
    import random

    import numpy as np
    import torch
    from greek_asr.infer import _environment

    protocol = check_protocol()
    spec = protocol['systems'][name]
    output = Path(os.environ['EXTENDED_PREDICT_OUTPUT'])
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    torch.cuda.set_per_process_memory_fraction(.85)
    module = importlib.import_module('greek_asr.models.' + spec['kind'])
    started = time.perf_counter()
    result = module.predict(ROOT / spec['checkpoint'], ROOT / TEST_MANIFEST, project_root=ROOT, split='test',
                            output_dir=output, device='cuda', batch_size=spec['batch_size'], allow_test=True,
                            test_protocol=ROOT / FINAL_PROTOCOL)
    result.update(job={'name': name, 'kind': spec['kind'], 'batch_size': spec['batch_size'], 'device': 'cuda',
                       'checkpoint': spec['checkpoint'], 'role': 'baseline_extended_training'},
                  worker_seconds_including_load=time.perf_counter() - started, seed=42, environment=_environment(),
                  extended_protocol_sha256=sha(PROTOCOL),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  predictions_sha256=sha(output / 'predictions.jsonl'))
    write(output / 'run.json', result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/extended_ctc')
    parser.add_argument('--freeze-protocol', action='store_true', help='write the protocol file (before committing)')
    parser.add_argument('--commit', action='store_true', help='git-commit the new results/weights at the end')
    parser.add_argument('--push', action='store_true', help='push after committing')
    parser.add_argument('--predict-child')
    args = parser.parse_args()
    if args.freeze_protocol:
        write(PROTOCOL, build_protocol())
        print(json.dumps({'written': str(PROTOCOL.relative_to(ROOT))}))
        return
    if args.predict_child:
        predict(args.predict_child)
        return

    protocol = check_protocol()
    from greek_asr.data.freeze import verify_freeze
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'p3-extended-ctc-run-v1', 'status': 'running', 'protocol_sha256': sha(PROTOCOL),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'systems': {}}
    record_path = ROOT / 'results/benchmark_extended/run_record.json'
    write(output / 'test_freeze_before.json', verify_freeze(ROOT, check_audio=True))
    for name in protocol['system_order']:
        spec = protocol['systems'][name]
        entry = record['systems'][name] = {'status': 'training', 'started_utc': datetime.now(timezone.utc).isoformat()}
        write(output / 'record.json', record)
        try:
            train_dir = output / 'train' / name
            started = time.perf_counter()
            run_logged([sys.executable, str(ROOT / 'scripts/run_training_job.py'), '--job', spec['job'],
                        '--output', str(train_dir)], output / f'{name}.train.log')
            entry['training_wall_seconds'] = time.perf_counter() - started
            run = read(train_dir / 'run.json')
            if run.get('status') != 'complete' or run.get('best_optimizer_update') is None:
                raise RuntimeError('training did not complete with a DEV-selected checkpoint')
            entry.update(best_optimizer_update=run['best_optimizer_update'], best_dev_loss=run['best_dev_loss'])
            training_record = ROOT / 'results/training_extended' / name
            training_record.mkdir(parents=True, exist_ok=False)
            shutil.copy2(train_dir / 'run.json', training_record / 'run.json')
            live = train_dir.with_name(train_dir.name + '.live.json')
            if live.exists():
                shutil.copy2(live, training_record / 'live.json')
            entry['export'] = export_checkpoint(name, spec, train_dir)
            entry['status'] = 'test_inference'
            write(output / 'record.json', record)
            test_dir = output / 'test' / name
            child_env = dict(runtime_env(), EXTENDED_PREDICT_OUTPUT=str(test_dir))
            with (output / f'{name}.test.log').open('w', encoding='utf-8') as log:
                if subprocess.run([sys.executable, str(Path(__file__)), '--predict-child', name], cwd=ROOT,
                                  env=child_env, stdout=log, stderr=subprocess.STDOUT).returncode:
                    raise RuntimeError(f'test inference failed; see {output / (name + ".test.log")}')
            with (output / f'{name}.score.log').open('w', encoding='utf-8') as log:
                if subprocess.run([sys.executable, '-m', 'greek_asr.evaluation', '--references', str(ROOT / TEST_REFERENCES),
                                   '--predictions', str(test_dir / 'predictions.jsonl'), '--manifest', str(ROOT / TEST_MANIFEST),
                                   '--output', str(test_dir), '--split', 'test', '--protocol', str(ROOT / FINAL_PROTOCOL)],
                                  cwd=ROOT, env=runtime_env(True), stdout=log, stderr=subprocess.STDOUT).returncode:
                    raise RuntimeError('scoring failed')
            shutil.copytree(test_dir, ROOT / 'results/benchmark_extended' / name)
            score = read(test_dir / 'score.json')
            entry.update(status='complete', test_wer=score['wer'], test_cer=score['cer'])
            print(json.dumps({'completed': name, 'wer': score['wer'], 'cer': score['cer']}), flush=True)
        except Exception as error:  # record and continue with the next pre-registered system
            entry.update(status='failed', error_type=type(error).__name__, error=str(error))
            print(json.dumps({'failed': name, 'error': str(error)}), flush=True)
        write(output / 'record.json', record)
    write(output / 'test_freeze_after.json', verify_freeze(ROOT, check_audio=True))
    record['test_freeze_verified_before_and_after'] = True
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    record['status'] = 'complete' if all(e['status'] == 'complete' for e in record['systems'].values()) else 'has_failures'
    write(record_path, record)
    write(output / 'record.json', record)
    if args.commit:
        paths = ['results/benchmark_extended', 'results/training_extended'] + [
            protocol['systems'][name]['checkpoint'].split('/model.pt')[0] for name, e in record['systems'].items()
            if e.get('export')]
        subprocess.run(['git', 'add', '--'] + [p for p in paths if (ROOT / p).exists()], cwd=ROOT, check=True)
        summary = ', '.join(f"{n}: {e['status']}" + (f" WER {100 * e['test_wer']:.2f}%" if 'test_wer' in e else '')
                            for n, e in record['systems'].items())
        subprocess.run(['git', 'commit', '-m', 'Run pre-registered extended CTC/HuBERT training and single test pass\n\n' + summary],
                       cwd=ROOT, check=True)
        if args.push:
            subprocess.run(['git', 'push', 'origin', 'HEAD'], cwd=ROOT, check=True)
    print(json.dumps({'status': record['status'], 'record': str(record_path.relative_to(ROOT))}), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
