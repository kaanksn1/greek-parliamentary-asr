"""Re-score saved predictions and check final cohort/provenance; never run ASR."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--allow-partial',action='store_true',help='Inspect completed systems; not final acceptance')
    args=parser.parse_args()
    root=args.root.resolve()
    sys.path.insert(0,str(root))
    from greek_asr.evaluation import score, speaker_intervals, source_sha256
    from greek_asr.scoring import load_jsonl
    def read(relative):
        return json.loads((root/relative).read_text(encoding='utf-8-sig'))
    def sha(relative):
        h=hashlib.sha256()
        with (root/relative).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    plan=read('configs/benchmark_plan.json')
    protocol=read(plan['protocol'])
    assert sha(plan['protocol'])=='b322570ca6376e7524466534b758155d6a28cb6ef0f3f92bf458969710742db2'
    assert source_sha256()==protocol['scorer_code_sha256']
    for relative,expected in {**protocol['source_locks'],**protocol['inference_code_sha256'],
        plan['test_manifest']:protocol['test_manifest_sha256'],plan['test_references']:protocol['test_references_sha256'],
        'configs/selected_checkpoints.json':protocol['selected_checkpoints_sha256'],
        'configs/selected_decoders.json':protocol['selected_decoders_sha256'],
        'configs/benchmark_plan.json':protocol['benchmark_plan_sha256']}.items():
        assert sha(relative)==expected,relative
    references,_=load_jsonl(root/plan['test_references'],'reference')
    manifest=[json.loads(line) for line in (root/plan['test_manifest']).read_text().splitlines()]
    summary=read('results/benchmark/summary.json')
    table={r['system']:r for r in summary['rows']}
    with (root/'results/benchmark/results.csv').open(encoding='utf-8',newline='') as stream:
        csv_rows={r['system']:r for r in csv.DictReader(stream)}
    checked=[]
    for job in plan['jobs']:
        folder=Path('results/benchmark')/job['name']
        if not (root/folder/'score.json').exists():
            assert args.allow_partial, 'missing final system: '+job['name']
            continue
        saved,run=read(folder/'score.json'),read(folder/'run.json')
        predictions,pred_hash=load_jsonl(root/folder/'predictions.jsonl','text')
        assert len(predictions)==len(references)==protocol['test_records']==8679
        assert pred_hash==saved['predictions_sha256']==run['predictions_sha256']
        assert run['status']=='complete' and run['job']==job
        assert run['code_commit']=='0ff3196c9870317be89d11a7fd09277bd0445dde'
        assert saved['reference_sha256']==protocol['test_references_sha256']
        actual,details=score(references,predictions)
        for key,value in actual.items(): assert saved[key]==value,(job['name'],key)
        assert read(folder/'score.json')['speaker_bootstrap_95_ci']==speaker_intervals(details,manifest)
        for metric in ('wer','cer'):
            value=100*actual[metric]
            assert math.isfinite(value) and abs(table[job['name']][metric+'_percent']-value)<1e-10
            assert abs(float(csv_rows[job['name']][metric+'_percent'])-value)<1e-10
        assert table[job['name']]['status']=='complete'
        checked.append({'system':job['name'],'records':len(predictions),'counts_and_intervals_recomputed':True,'prediction_sha256':pred_hash})
        print(json.dumps({'verified':job['name']}),flush=True)
    assert summary['completed_systems']==len(checked)
    catalog=read('configs/selected_checkpoints.json')
    for name,entry in catalog['exports'].items():
        assert sha(Path('results/training')/name/'run.json')==entry['training_run_sha256']
        files={str(Path(entry['path'])/n):h for n,h in entry['files'].items()} if 'files' in entry else {entry['path']:entry['export_sha256']}
        for relative,expected in files.items(): assert sha(relative)==expected,relative
    if not args.allow_partial:
        assert len(checked)==len(plan['jobs'])==12 and summary['status']=='complete'
        review=Path('docs/reviews/final_benchmark_2026-11-21')
        queue=read(review/'final_queue.json')
        assert queue['status']=='complete' and queue['final_report_exit_code']==0
        for suffix in ('before','after'):
            freeze=read(review/('final_test_freeze_'+suffix+'.json'))
            assert freeze['status']=='ok' and freeze['audio_checked'] is True
            assert freeze['freeze_sha256']==protocol['test_freeze_sha256']
        for plot in ('wer_percent','cer_percent','speed_accuracy','whisper_conditions'):
            assert (root/'results/benchmark/figures'/(plot+'.png')).stat().st_size>1000
            assert (root/'results/benchmark/figures'/(plot+'.svg')).stat().st_size>1000
    result={'schema':'p3-final-result-acceptance-v1','status':'partial' if args.allow_partial else 'passed',
        'checked_utc':datetime.now(timezone.utc).isoformat(),'systems':checked,'complete_systems_checked':len(checked),
        'expected_systems':12,'original_training_run_hashes_match':True,'asr_inference_repeated':False,
        'generator_sha256':sha('scripts/validate_final_benchmark.py')}
    target=root/'docs/reviews/final_benchmark_2026-11-21/results_validation.json'
    target.write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':result['status'],'verified_systems':len(checked)}),flush=True)


if __name__=='__main__': main()
