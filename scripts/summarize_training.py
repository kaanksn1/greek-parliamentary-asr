"""Generate actual training-budget/selection tables from original run records."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[]
    for path in sorted((ROOT/'results/training').glob('*/run.json')):
        run=json.loads(path.read_text(encoding='utf-8'))
        iterations=run.get('iterations',[])
        updates=run.get('optimizer_updates_completed')
        allocation=run.get('max_steps')
        status=run.get('status') or ('complete' if updates==allocation and allocation is not None else 'unknown')
        row={'system':path.parent.name,'status':status,'seed':run['seed'],'device':run['device_used'],
             'optimizer_updates':updates,'allocated_updates':allocation,'selected_update':run.get('best_optimizer_update'),
             'effective_batch':run.get('effective_batch_size'),
             'em_iterations':max(item['iteration'] for item in iterations) if isinstance(iterations,list) and iterations else None,
             'train_records':run.get('train_utterances',run.get('train_items',run.get('train_utterances_used'))),
             'training_hours':run['training_seconds']/3600,'precision':run.get('precision','CPU model-specific'),
             'code_commit':run['code_commit'],'run_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if row['status']!='complete':
            raise ValueError('incomplete training record: '+path.parent.name)
        rows.append(row)
    if len(rows)!=7:
        raise ValueError('expected all seven original training runs')
    destination=ROOT/'results/training'
    with (destination/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={'schema':'p3-original-training-summary-v1','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':rows,'note':'Per-process training wall times; CPU GMM overlapped GPU jobs. The sum is not project elapsed wall time.'}
    (destination/'summary.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    text=['# Completed training and DEV-selected checkpoints','','Generated from unmodified original run.json records; do not edit numbers manually.','',
          '| System | Training records | Updates / EM | Selected update | Effective batch | Hours |',
          '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        budget=f"{row['em_iterations']} EM" if row['system']=='gmm_hmm' else str(row['optimizer_updates'])
        text.append(f"| {row['system']} | {row['train_records']} | {budget} | {row['selected_update'] or 'fixed EM allocation'} | {row['effective_batch'] or 'n/a'} | {row['training_hours']:.3f} |")
    text+=['','Every seed is 42. Checkpoint selection uses full-development loss; both Whisper conditions select update 500 after completing 1,600 updates.',
           'Model-specific training exclusions are recorded explicitly in each original run/config. Test inference retains all clips. GMM iteration 0 is initialization, followed by 25 EM iterations; its character LM has order 3.',
           report['note'],'Dev losses across architectures are different objectives and should not be compared as model accuracy.','']
    (destination/'SUMMARY.md').write_bytes(('\n'.join(text)).encode('utf-8'))
    print(json.dumps({'complete_training_runs':len(rows),'rows':rows}))


if __name__=='__main__':
    main()
