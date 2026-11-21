"""Compare bounded batch 4/8 on DEV, selecting by time/memory, never test WER."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT/'outputs/dev_batch_resources'
    output.mkdir(parents=True, exist_ok=False)
    path = ROOT/'configs/benchmark_plan.json'
    original = path.read_bytes()
    plan = read(path)
    (output/'original_plan.json').write_bytes(original)
    env = dict(os.environ, HSA_ENABLE_DXG_DETECTION='1', MIOPEN_FIND_MODE='FAST',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    evidence = {'selection_uses': 'DEV inference time and observed GPU memory only', 'trials': []}
    for name in ('whisper_pretrained', 'whisper_hparl', 'whisper_mixed', 'qwen'):
        job = next(row for row in plan['jobs'] if row['name']==name)
        before = read(ROOT/'results/dev_calibration'/name/'run.json')
        old_batch = job['batch_size']
        job['batch_size'] = 8
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
        with (output/(name+'.log')).open('w') as log:
            process = subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_worker.py'), '--job', name,
                '--phase', 'dev', '--output', str(output/name)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {'name': name, 'candidate_batch':8, 'previous_batch':old_batch, 'exit_code':process.returncode}
        old_seconds = before.get('inference_seconds_including_output_io', before.get('inference_seconds'))
        result['previous_seconds'] = old_seconds
        selected = False
        if process.returncode==0:
            after = read(output/name/'run.json')
            seconds = after.get('inference_seconds_including_output_io', after.get('inference_seconds'))
            result.update(candidate_seconds=seconds, peak_gpu_bytes=after['environment'].get('peak_gpu_memory_allocated_bytes'))
            selected = seconds < old_seconds
            with (output/(name+'.score.log')).open('w') as log:
                scored = subprocess.run([sys.executable, '-m', 'greek_asr.evaluation', '--references', plan['dev_references'],
                    '--predictions', str(output/name/'predictions.jsonl'), '--manifest', plan['dev_manifest'],
                    '--output', str(output/name), '--split', 'dev'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if scored.returncode:
                raise RuntimeError('DEV resource trial scoring failed: '+name)
        if selected:
            archived = ROOT/'results/dev_resource_trials/batch4'/name
            archived.parent.mkdir(parents=True, exist_ok=True)
            (ROOT/'results/dev_calibration'/name).rename(archived)
            shutil.copytree(output/name, ROOT/'results/dev_calibration'/name)
        else:
            job['batch_size'] = old_batch
        result['selected_batch'] = job['batch_size']
        evidence['trials'].append(result)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
        (ROOT/'configs/dev_batch_resources.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
