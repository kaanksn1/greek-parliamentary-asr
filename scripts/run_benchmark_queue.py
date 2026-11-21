"""Run all planned ASR systems, score complete cohorts, and produce reproducible reports."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nemo-python', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT / 'configs/benchmark_plan.json').read_text())
    if args.phase == 'test':
        sys.path.insert(0, str(ROOT))
        from benchmark_worker import validate_test
        from greek_asr.data.freeze import verify_freeze
        validate_test(plan)
        verification = verify_freeze(ROOT, check_audio=True)
        (output / 'test_freeze_before.json').write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8')
    state = {'schema': 'p3-benchmark-queue-v1', 'phase': args.phase, 'status': 'running', 'pid': os.getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'expected_systems': len(plan['jobs']), 'jobs': {}}
    destination = ROOT / 'results' / ('dev_calibration' if args.phase == 'dev' else 'benchmark')
    destination.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output / 'queue.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary, output / 'queue.json')

    def environment(cpu=False):
        return dict(os.environ, HSA_ENABLE_DXG_DETECTION='0' if cpu else '1', MIOPEN_FIND_MODE='FAST',
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')

    def launch(job):
        python = str(args.nemo_python) if job.get('backend') in ('canary', 'parakeet') else sys.executable
        log = (output / (job['name']+'.log')).open('w', encoding='utf-8')
        process = subprocess.Popen([python, str(ROOT / 'scripts/benchmark_worker.py'), '--job', job['name'],
                                    '--phase', args.phase, '--output', str(output / job['name'])], cwd=ROOT,
                                   env=environment(job['device']=='cpu'), stdout=log, stderr=subprocess.STDOUT)
        state['jobs'][job['name']] = {'status': 'running', 'pid': process.pid,
            'started_utc': datetime.now(timezone.utc).isoformat(), 'log': str(output/(job['name']+'.log'))}
        save()
        print(json.dumps({'started': job['name'], 'phase': args.phase}), flush=True)
        return process, log

    def finish(job, process, log):
        log.close()
        entry = state['jobs'][job['name']]
        entry.update(exit_code=process.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        if process.returncode != 0:
            entry['status']='failed'
            save()
            return
        command = [sys.executable, '-m', 'greek_asr.evaluation', '--references', str(ROOT/plan[args.phase+'_references']),
                   '--predictions', str(output/job['name']/'predictions.jsonl'), '--manifest', str(ROOT/plan[args.phase+'_manifest']),
                   '--output', str(output/job['name']), '--split', args.phase]
        if args.phase == 'test':
            command += ['--protocol', str(ROOT/plan['protocol'])]
        score_log = (output/(job['name']+'.score.log')).open('w', encoding='utf-8')
        scored = subprocess.run(command, cwd=ROOT, env=environment(True), stdout=score_log, stderr=subprocess.STDOUT)
        score_log.close()
        entry['score_exit_code'] = scored.returncode
        entry['status'] = 'complete' if scored.returncode == 0 else 'scoring_failed'
        if scored.returncode == 0:
            target = destination/job['name']
            shutil.copytree(output/job['name'], target, dirs_exist_ok=False)
            if args.phase == 'test':
                with (output/'reports.log').open('a', encoding='utf-8') as report_log:
                    reported = subprocess.run([sys.executable, str(ROOT/'scripts/summarize_benchmark.py')], cwd=ROOT,
                                              env=environment(True), stdout=report_log, stderr=subprocess.STDOUT)
                entry['report_exit_code'] = reported.returncode
        save()
        print(json.dumps({'finished': job['name'], 'status': entry['status']}), flush=True)

    cpu_job = next(job for job in plan['jobs'] if job['device']=='cpu')
    cpu, cpu_log = launch(cpu_job)
    cpu_finished = False
    order = ['whisper_pretrained', 'whisper_hparl', 'whisper_mixed', 'plain_ctc', 'hubert_ctc',
             'attention', 'transducer', 'wav2vec', 'canary', 'parakeet', 'qwen']
    for name in order:
        job = next(job for job in plan['jobs'] if job['name']==name)
        process, log = launch(job)
        while process.poll() is None:
            if not cpu_finished and cpu.poll() is not None:
                finish(cpu_job, cpu, cpu_log)
                cpu_finished = True
            save()
            time.sleep(5)
        finish(job, process, log)
    if not cpu_finished:
        while cpu.poll() is None:
            save()
            time.sleep(5)
        finish(cpu_job, cpu, cpu_log)
    state['status'] = 'complete' if all(row['status']=='complete' for row in state['jobs'].values()) else 'has_failures'
    if args.phase == 'test':
        verification = verify_freeze(ROOT, check_audio=True)
        (output/'test_freeze_after.json').write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8')
        with (output/'reports.log').open('a', encoding='utf-8') as report_log:
            final_report = subprocess.run([sys.executable, str(ROOT/'scripts/summarize_benchmark.py')], cwd=ROOT,
                                          env=environment(True), stdout=report_log, stderr=subprocess.STDOUT)
        state['final_report_exit_code'] = final_report.returncode
        if final_report.returncode:
            state['status'] = 'report_failed'
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
