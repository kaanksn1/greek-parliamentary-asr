"""Generate tables, figures, paired comparison and the presentation brief from actual outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def extended_section(root, protocol, rows, text):
    """Separate rows for the registered extended-training follow-up (configs/extended_ctc_protocol.json).

    The original 12 rows, results.csv and their validation stay unchanged; these rows are written to
    results/benchmark_extended/results.csv and a separate SUMMARY.md section.
    """
    extended_dir = root/'results/benchmark_extended'
    record_path, protocol_path = extended_dir/'run_record.json', root/'configs/extended_ctc_protocol.json'
    if not record_path.exists():
        return []
    record, registered = read(record_path), read(protocol_path)
    if record['protocol_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError('extended run record does not belong to the committed extended protocol')
    if record.get('test_freeze_verified_before_and_after') is not True:
        raise ValueError('extended run lacks before/after test-freeze verification')
    original = {row['system']: row for row in rows}
    extended_rows = []
    for name in registered['system_order']:
        entry, spec = record['systems'][name], registered['systems'][name]
        if entry['status'] != 'complete':
            extended_rows.append({'system': name, 'status': entry['status']})
            continue
        score, run = read(extended_dir/name/'score.json'), read(extended_dir/name/'run.json')
        if (run['status'] != 'complete' or score['examples'] != protocol['test_records']
                or score['reference_sha256'] != protocol['test_references_sha256']
                or score['scorer_code_sha256'] != protocol['scorer_code_sha256']
                or score['predictions_sha256'] != run['predictions_sha256']):
            raise ValueError('refusing a partial/inconsistent extended row: '+name)
        inference = run.get('inference_seconds', run.get('worker_seconds_including_load'))
        intervals = score['speaker_bootstrap_95_ci']
        before = original[spec['original']]
        extended_rows.append({
            'system': name, 'role': 'baseline_extended_training', 'status': 'complete', 'original_system': spec['original'],
            'original_wer_percent': before['wer_percent'], 'original_cer_percent': before['cer_percent'],
            'records': score['examples'], 'wer_percent': 100*score['wer'], 'cer_percent': 100*score['cer'],
            'wer_ci95_low_percent': 100*intervals['wer'][0], 'wer_ci95_high_percent': 100*intervals['wer'][1],
            'cer_ci95_low_percent': 100*intervals['cer'][0], 'cer_ci95_high_percent': 100*intervals['cer'][1],
            'rtfx': score['audio_seconds']/inference, 'inference_seconds': inference,
            'batch_size': spec['batch_size'], 'device': spec['device'],
            'original_updates': spec['changes_from_original']['max_steps']['original'],
            'extended_updates': spec['changes_from_original']['max_steps']['extended'],
            'selected_update': entry['best_optimizer_update'], 'best_dev_loss': entry['best_dev_loss'],
            'word_substitutions': score['word_counts']['substitutions'], 'word_insertions': score['word_counts']['insertions'],
            'word_deletions': score['word_counts']['deletions'], 'empty_predictions': score['empty_predictions']})
    fields = ['system','role','status','original_system','original_updates','extended_updates','selected_update','best_dev_loss',
              'records','original_wer_percent','wer_percent','original_cer_percent','cer_percent','rtfx','inference_seconds',
              'batch_size','device','word_substitutions','word_insertions','word_deletions','empty_predictions',
              'wer_ci95_low_percent','wer_ci95_high_percent','cer_ci95_low_percent','cer_ci95_high_percent']
    with (extended_dir/'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(extended_rows)
    text += ['## Registered extended-training follow-up (separate rows)', '',
             'Longer training of the two group-trained CTC baselines whose original runs had not converged on full-DEV loss. '
             '[The registered protocol](../../configs/extended_ctc_protocol.json) fixed the schedules, DEV-loss checkpoint selection and '
             'one test pass before training; data, model, seed, learning rate and decoder are unchanged. '
             'The original rows above stay unchanged.', '',
             '| System | Updates (original → extended) | Selected update | WER % (original → extended) | CER % (original → extended) | WER 95% CI | RTFx | Status |',
             '|---|---:|---:|---:|---:|---|---:|---|']
    for row in extended_rows:
        if row['status'] != 'complete':
            text.append(f"| {row['system']} | | | | | | | {row['status']} |")
            continue
        text.append(f"| {row['system']} | {row['original_updates']:,} → {row['extended_updates']:,} | {row['selected_update']:,} | "
                    f"{row['original_wer_percent']:.2f} → " + (f"**{row['wer_percent']:.2f}**" if row['wer_percent'] < min(100, row['original_wer_percent']) else f"{row['wer_percent']:.2f}") + " | " + f"{row['original_cer_percent']:.2f} → {row['cer_percent']:.2f} | "
                    f"[{row['wer_ci95_low_percent']:.2f}, {row['wer_ci95_high_percent']:.2f}] | {row['rtfx']:.2f} | {row['status']} |")
    text += ['', 'The original test results were known when this follow-up was registered, so these rows are not an independent '
             'unseen-test confirmation. A WER near or above 100% means the condition still did not learn usable transcriptions.', '']
    return extended_rows


README_START, README_END = '<!-- generated-results:start -->', '<!-- generated-results:end -->'


def readme_block(root, rows, extended_rows):
    """Rewrite the generated results block in README.md from the same rows as SUMMARY.md."""
    readme = root/'README.md'
    if not readme.exists():
        return
    content = readme.read_text(encoding='utf-8')
    if content.count(README_START) != 1 or content.count(README_END) != 1:
        return
    def line(name, row, note):
        if row.get('status') != 'complete':
            return f"| {name} | {row.get('status', 'pending')} | | | {note} |"
        return (f"| {name} | {row['wer_percent']:.2f} | {row['cer_percent']:.2f} | "
                f"[{row['wer_ci95_low_percent']:.2f}, {row['wer_ci95_high_percent']:.2f}] | {note} |")
    by_name = {row['system']: row for row in rows}
    block = [README_START, '', 'All 8,679 frozen test clips, one scoring policy; WER/CER in %, lower is better; '
             '95% speaker-bootstrap CI for WER. Full table with speed and batch settings: '
             '[results/benchmark/SUMMARY.md](results/benchmark/SUMMARY.md).', '',
             '| System | WER % | CER % | WER 95% CI | Group training |', '|---|---:|---:|---|---|']
    notes = {'gmm_hmm': 'own implementation', 'plain_ctc': 'own implementation', 'attention': 'own implementation',
             'transducer': 'own implementation', 'hubert_ctc': 'own CTC head training', 'wav2vec': 'none (published Greek fine-tune)',
             'whisper_pretrained': 'none', 'canary': 'none', 'parakeet': 'none', 'qwen': 'none'}
    block.append('| **Standard baselines** | | | | |')
    for row in rows:
        if row['role'] == 'baseline':
            block.append(line(row['system'], row, notes.get(row['system'], '')))
    if extended_rows:
        block.append('| **Registered extended training (follow-up rows)** | | | | |')
        for row in extended_rows:
            block.append(line(row['system'], row, 'own training, longer registered schedule'))
    block.append('| **Our contribution: Whisper-small fine-tuning** | | | | |')
    for name, note in (('whisper_hparl', 'HParl only'), ('whisper_mixed', 'HParl + CSS10')):
        if name in by_name:
            block.append(line(name, by_name[name], note))
    block += ['', README_END]
    start, end = content.index(README_START), content.index(README_END) + len(README_END)
    readme.write_text(content[:start] + '\n'.join(block) + content[end:], encoding='utf-8')


DATA_START, DATA_END = '<!-- generated-data:start -->', '<!-- generated-data:end -->'


def data_block(root):
    """Rewrite the generated data/overlap block in README.md from results/data_integrity.json."""
    readme, source = root/'README.md', root/'results/data_integrity.json'
    if not readme.exists() or not source.exists():
        return
    content = readme.read_text(encoding='utf-8')
    if content.count(DATA_START) != 1 or content.count(DATA_END) != 1:
        return
    data = json.loads(source.read_text(encoding='utf-8'))
    cohorts, cleaning, overlap = data['cohorts'], data['cleaning'], data['split_pair_group_overlap']
    to_test = data['training_to_test_overlap']['hparl_train']
    def speakers(cohort):
        if not cohort['speakers_with_ids']:
            return 'No encoded IDs; producer single reader'
        missing = cohort['utterances_without_speaker_id']
        return f"{cohort['speakers_with_ids']} identified; {missing:,} utterances without IDs" if missing else f"{cohort['speakers_with_ids']}"
    rows = (('Selected HParl training', 'hparl_train'), ('Additional CSS10 training', 'css10_train'),
            ('Mixed training', 'mixed_train'), ('HParl development', 'dev'), ('Original frozen HParl test', 'test'))
    block = [DATA_START, '', '| Actual cleaned cohort | Utterances | Speakers | Hours |', '|---|---:|---:|---:|']
    for label, key in rows:
        cohort = cohorts[key]
        block.append(f"| {label} | {cohort['utterances']:,} | {speakers(cohort)} | {cohort['hours']:.6f} |")
    raw, dropped = cohorts['raw_hparl_train'], cleaning['dropped_by_rule']
    block += ['',
              f"- Selected raw HParl training: {raw['utterances']:,} original clips from {raw['recordings_with_ids']} whole sessions "
              f"({raw['hours']:.2f} h). Cleaning removed {cleaning['removed_records']} records "
              f"({dropped['cps_extreme']} extreme transcription speed, {dropped['duplicate_audio']} duplicate audio); "
              f"{cleaning['review_queue_records']} review-queue flags, {cleaning['human_review_decisions']} human review decisions.",
              f"- Speaker overlap (the official split is not speaker-disjoint): {to_test['speakers']} of "
              f"{cohorts['test']['speakers_with_ids']} test speakers also occur in selected training, covering "
              f"{to_test['test_clips_with_shared_speaker']:,} test clips; shared recordings/sessions {to_test['recordings']}, "
              f"shared audio hashes {to_test['shared_audio_sha256_values']}. Dev-test shared speakers: "
              f"{overlap['dev__test']['speakers']}; train-dev: {overlap['hparl_train__dev']['speakers']}.",
              f"- Test sentences found word for word in the clean training targets: {to_test['literal_matching_test_utterances']} "
              f"utterances ({to_test['literal_matching_unique_sentences']} unique); under the frozen final normalization: "
              f"{to_test['final_policy_matching_test_utterances']} utterances ({to_test['final_policy_matching_unique_sentences']} unique).",
              '', DATA_END]
    start, end = content.index(DATA_START), content.index(DATA_END) + len(DATA_END)
    readme.write_text(content[:start] + '\n'.join(block) + content[end:], encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root/'results/benchmark'
    destination.mkdir(parents=True, exist_ok=True)
    plan = read(root/'configs/benchmark_plan.json')
    protocol = read(root/plan['protocol'])
    rows, completed = [], {}
    for job in plan['jobs']:
        folder = destination/job['name']
        row = {'system': job['name'], 'role': job['role'], 'status': 'pending', 'wer_percent': None,
               'cer_percent': None, 'rtfx': None, 'inference_seconds': None, 'batch_size': job['batch_size'], 'device': job['device']}
        metadata = protocol['system_details'][job['name']]
        row.update(model_id=metadata['model_id'], revision=metadata['revision'], trained_by_group=metadata['trained_by_group'])
        if (folder/'score.json').exists() and (folder/'run.json').exists():
            score, run = read(folder/'score.json'), read(folder/'run.json')
            if (run['status'] != 'complete' or score['examples'] != protocol['test_records']
                    or score['reference_sha256'] != protocol['test_references_sha256']
                    or score['scorer_code_sha256'] != protocol['scorer_code_sha256']):
                raise ValueError('refusing a partial/failed final benchmark row')
            inference = run.get('inference_seconds_including_output_io', run.get('inference_seconds'))
            if inference is None:
                inference = run['worker_seconds_including_load']
            row.update(status='complete', wer_percent=100*score['wer'], cer_percent=100*score['cer'],
                       inference_seconds=inference, rtfx=score['audio_seconds']/inference,
                       word_substitutions=score['word_counts']['substitutions'],
                       word_insertions=score['word_counts']['insertions'], word_deletions=score['word_counts']['deletions'],
                       empty_predictions=score['empty_predictions'], records=score['examples'],
                       peak_gpu_bytes=run.get('environment', {}).get('peak_gpu_memory_allocated_bytes'))
            intervals = score['speaker_bootstrap_95_ci']
            for metric in ('wer','cer'):
                row[metric+'_ci95_low_percent'] = 100*intervals[metric][0]
                row[metric+'_ci95_high_percent'] = 100*intervals[metric][1]
            completed[job['name']] = (score, run)
            details = lines(folder/'utterances.jsonl')
            ranked = sorted(details, key=lambda r: (-r['word_counts']['errors']/max(r['word_counts']['reference_units'], 1), r['id']))
            random_sample = random.Random(42).sample(details, min(10, len(details)))
            (folder/'error_examples.json').write_text(json.dumps({'selection': 'highest utterance error rate plus seed-42 random sample',
                'highest_error': ranked[:10], 'random_sample': random_sample}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        rows.append(row)
    fields = ['system','role','status','model_id','revision','trained_by_group','records','wer_percent','cer_percent','rtfx','inference_seconds','batch_size','device',
              'word_substitutions','word_insertions','word_deletions','empty_predictions','peak_gpu_bytes',
              'wer_ci95_low_percent','wer_ci95_high_percent','cer_ci95_low_percent','cer_ci95_high_percent']
    with (destination/'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = len(completed)==len(plan['jobs'])
    report = {'schema':'p3-benchmark-summary-v1','status':'complete' if complete else 'partial',
              'generated_utc':datetime.now(timezone.utc).isoformat(), 'expected_systems':len(rows),'completed_systems':len(completed),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows':rows}
    def fmt(value):
        return 'pending' if value is None else f'{value:.2f}'
    text = ['# ASR benchmark results', '', f"Status: **{report['status']} — {len(completed)}/{len(rows)} systems complete**.",
            '', 'Generated by `scripts/summarize_benchmark.py`; never edit table values manually.',
            'All completed systems use all 8,679 original frozen test clips, the same references and common frozen scoring policy.',
            'WER/CER are percentages (lower is better); RTFx is audio duration divided by inference time (higher is better).',
            'Inference includes output I/O, excludes model loading unless a run explicitly lacks a separate timer. Batch/device are shown.', '']
    for role, title in [('baseline','Standard baselines'), ('contribution','Our matched Whisper fine-tuning conditions')]:
        text += ['## '+title, '', '| System | WER % | CER % | RTFx | Batch / device | Status |', '|---|---:|---:|---:|---|---|']
        for row in rows:
            if row['role']==role:
                text.append(f"| {row['system']} | {fmt(row['wer_percent'])} | {fmt(row['cer_percent'])} | {fmt(row['rtfx'])} | {row['batch_size']} / {row['device']} | {row['status']} |")
        text.append('')
    pair = ('whisper_hparl','whisper_mixed')
    if all(name in completed for name in pair):
        import numpy as np
        manifest = lines(root/plan['test_manifest'])
        speaker_of = {r['id']:r.get('speaker_id') or r.get('source_recording_id') or r['id'] for r in manifest}
        speakers = sorted(set(speaker_of.values()))
        position = {name:i for i,name in enumerate(speakers)}
        values = np.zeros((len(speakers),3))
        for column, name in enumerate(pair):
            for item in lines(destination/name/'utterances.jsonl'):
                index = position[speaker_of[item['id']]]
                values[index,column] += item['word_counts']['errors']
                if column==0:
                    values[index,2] += item['word_counts']['reference_units']
        counts = np.random.default_rng(42).multinomial(len(speakers), np.full(len(speakers),1/len(speakers)), size=1000)
        samples = counts@values
        delta = 100*(samples[:,1]-samples[:,0])/samples[:,2]
        comparison = {'difference':'mixed minus HParl-only WER, percentage points (negative favors mixed)',
            'delta_percentage_points':100*(completed[pair[1]][0]['wer']-completed[pair[0]][0]['wer']),
            'paired_speaker_bootstrap_95_ci':np.quantile(delta,[.025,.975]).tolist(), 'speakers':len(speakers), 'seed':42,'resamples':1000}
        report['paired_data_composition_comparison']=comparison
        ci_low, ci_high = comparison['paired_speaker_bootstrap_95_ci']
        if ci_high < 0:
            ci_reading = ('The paired 95% CI lies entirely below zero: on this test cohort and this single training seed, '
                          'adding CSS10 gave a small but statistically supported WER reduction.')
        elif ci_low > 0:
            ci_reading = ('The paired 95% CI lies entirely above zero: on this test cohort and this single training seed, '
                          'adding CSS10 gave a statistically supported WER increase.')
        else:
            ci_reading = 'The paired 95% CI spans zero, so it does not support a directional improvement claim.'
        text += ['## Matched data-composition comparison','',f"Mixed minus HParl-only WER: **{comparison['delta_percentage_points']:.2f} percentage points**; paired speaker-bootstrap 95% CI [{ci_low:.2f}, {ci_high:.2f}].",
                 'Both conditions share the original checkpoint, seed, effective batch, learning rate, warmup, 1,600-update training allocation and full dev/test sets. Full-dev loss selected step 500 in both runs. '+ci_reading+' It does not estimate variation over training seeds.','']
    extended_rows = extended_section(root, protocol, rows, text)
    if extended_rows:
        report['extended_rows'] = extended_rows
    readme_block(root, rows, extended_rows)
    data_block(root)
    available = [row for row in rows if row['status']=='complete']
    if available:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures = destination/'figures'
        figures.mkdir(exist_ok=True)
        colors = ['#0072B2' if row['role']=='baseline' else '#D55E00' for row in available]
        for metric, title in [('wer_percent','WER (%)'), ('cer_percent','CER (%)')]:
            fig, ax = plt.subplots(figsize=(10,max(2.8,.42*len(available)+1.6)))
            ax.barh([r['system'] for r in available], [r[metric] for r in available], color=colors)
            short = metric.split('_')[0]
            for index, row in enumerate(available):
                low, high = row[short+'_ci95_low_percent'], row[short+'_ci95_high_percent']
                ax.hlines(index,low,high,color='black',linewidth=1.2)
                ax.plot([low,high],[index,index],'|',color='black',markersize=5)
                ax.text(max(high,row[metric])+.6,index,f"{row[metric]:.2f}",va='center',fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(title+' — lower is better')
            ax.set_title(f"Greek parliamentary ASR ({len(available)}/{len(rows)} complete)\n95% speaker-bootstrap intervals",fontsize=11)
            ax.grid(axis='x', alpha=.2)
            ax.margins(x=.12)
            fig.tight_layout()
            for extension in ('png','svg'):
                fig.savefig(figures/(metric+'.'+extension), dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(9,6))
        for index,row in enumerate(available):
            ax.scatter(row['rtfx'],row['wer_percent'],color='#D55E00' if row['role']=='contribution' else '#0072B2')
            ax.annotate(row['system'],(row['rtfx'],row['wer_percent']),fontsize=7,xytext=(4,5 if index%2==0 else -9),textcoords='offset points')
        if len(available)>1: ax.set_xscale('log')
        ax.set_xlabel('RTFx — higher is faster (log scale with multiple systems)'); ax.set_ylabel('WER (%) — lower is better')
        ax.set_title('Accuracy / speed on the same frozen test cohort'); ax.grid(alpha=.2); fig.tight_layout()
        for extension in ('png','svg'): fig.savefig(figures/('speed_accuracy.'+extension),dpi=180)
        plt.close(fig)
        comparison_names=('whisper_pretrained','whisper_hparl','whisper_mixed')
        if all(name in completed for name in comparison_names):
            fig, axes=plt.subplots(1,2,figsize=(10,4))
            labels=['Pretrained','HParl only','HParl + CSS10']
            for ax,metric in zip(axes,('wer','cer')):
                values=[100*completed[name][0][metric] for name in comparison_names]
                ax.bar(labels,values,color=['#0072B2','#D55E00','#009E73'])
                for index,name in enumerate(comparison_names):
                    low,high=[100*x for x in completed[name][0]['speaker_bootstrap_95_ci'][metric]]
                    ax.vlines(index,low,high,color='black',linewidth=1.2)
                    ax.plot([index,index],[low,high],'_',color='black')
                    ax.text(index,max(high,values[index])+.5,f'{values[index]:.2f}',ha='center',fontsize=9)
                ax.set_ylabel(metric.upper()+' (%) — lower is better'); ax.set_ylim(bottom=0)
                ax.margins(y=.2); ax.grid(axis='y',alpha=.2)
            fig.suptitle('Whisper-small: same original test, 95% speaker-bootstrap intervals',fontsize=12)
            fig.tight_layout()
            for extension in ('png','svg'): fig.savefig(figures/('whisper_conditions.'+extension),dpi=180)
            plt.close(fig)
        complete_extended = [row for row in extended_rows if row['status']=='complete']
        if complete_extended:
            # Separate figure: original vs registered extended training, same frozen test.
            fig, axes = plt.subplots(1,2,figsize=(10,4))
            labels = [row['system'].replace('_extended','') for row in complete_extended]
            positions = list(range(len(labels)))
            for ax, metric in zip(axes, ('wer','cer')):
                before = [row['original_'+metric+'_percent'] for row in complete_extended]
                after = [row[metric+'_percent'] for row in complete_extended]
                ax.bar([x-.2 for x in positions], before, width=.4, color='#999999', label='original allocation')
                ax.bar([x+.2 for x in positions], after, width=.4, color='#0072B2', label='registered extended training')
                for x, value in zip(positions, after):
                    ax.text(x+.2, value+1, f'{value:.2f}', ha='center', fontsize=9)
                ax.set_xticks(list(positions)); ax.set_xticklabels(labels)
                ax.set_ylabel(metric.upper()+' (%) — lower is better'); ax.set_ylim(bottom=0); ax.margins(y=.15)
                ax.grid(axis='y', alpha=.2)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
            fig.suptitle('Group-trained CTC baselines: original vs registered extended training (same frozen test)', fontsize=11)
            fig.tight_layout(rect=(0, .07, 1, 1))
            for extension in ('png','svg'): fig.savefig(figures/('extended_ctc.'+extension), dpi=180)
            plt.close(fig)
    text += ['## Limits','', 'One training seed and finite update budgets; from-scratch models may remain undertrained.',
             'The original plain_ctc and hubert_ctc allocations did not learn usable transcriptions (WER near 100%); their rows are kept unchanged. Registered extended training made HuBERT-CTC usable, while extended plain CTC still produces degenerate repeated-character output (see the follow-up section).',
             'The official session split is not speaker-disjoint: 49 test speakers also occur in selected training; results are not unseen-speaker generalization.',
             'Case/punctuation are unscored; accents, digits and symbols are retained. Empty normalized references remain in the full cohort.',
             'CPU/GPU and batch settings differ by system; speed is hardware/runtime dependent. Speaker-bootstrap CIs do not estimate variation over training seeds.',
             'Manual data-listening decisions remain zero; audit flags are not verified new annotations. No bonus award is claimed.','']
    (destination/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (destination/'SUMMARY.md').write_text('\n'.join(text),encoding='utf-8')
    brief = ['# Serhan için gerçek sonuç paketi', '', f"Durum: {len(completed)}/{len(rows)} sistem tamamlandı; {'son' if complete else 'kısmi'} sonuçlar.",
             '', 'Tablolar: results/benchmark/results.csv ve SUMMARY.md. Grafikler: results/benchmark/figures/.',
             'Her modelin score.json dosyasında ham hata sayıları ve speaker-bootstrap aralıkları; error_examples.json dosyasında otomatik seçilmiş gerçek hata örnekleri var.',
             'Veri: 30.883408 saat HParl; ek 4.137269 saat CSS10; karışım 35.020678 saat. Test: 8,679 değişmeyen kayıt.',
             'Baseline ve katkı satırlarını ayır. Qwen/Canary/Parakeet çalıştırmak kendi model katkımız değildir.',
             'İki Whisper koşulunda fark yalnız eğitim bileşimi; aynı 1,600 güncelleme/effective batch 16. İkisinde de tam dev kaybıyla seçilen ağırlık 500. adımdan. İyileşme iddiasını gerçek delta ve güven aralığına göre kur.',
             'Resmî split konuşmacı ayrık değildir; bunu sunumda belirt. Sınırlamalar ve başarısız/çok zayıf modelleri gizleme.',
             'Kendi sunum/analiz katkını contributions/OZAN_SERHAN_DENIZHAN_220911758_serhandenizhan.md içinde kendi hesabınla kaydet; kendi branch/PR üzerinden sunumu ekle.','']
    (root/'results/PRESENTATION_BRIEF_TR.md').write_text('\n'.join(brief),encoding='utf-8')
    if 'paired_data_composition_comparison' in report:
        comparison=report['paired_data_composition_comparison']
        low,high=comparison['paired_speaker_bootstrap_95_ci']
        extra=['','## Tamamlanan katkı deneyinin gerçek sonuçları','',
               f"HParl-only WER: %{100*completed['whisper_hparl'][0]['wer']:.2f}; HParl+CSS10 WER: %{100*completed['whisper_mixed'][0]['wer']:.2f}.",
               f"Karışım eksi HParl farkı: {comparison['delta_percentage_points']:.2f} yüzde puan; eşleştirilmiş %95 konuşmacı-bootstrap aralığı [{low:.2f}, {high:.2f}].",
               'Bu, tek eğitim seed’i ve bu resmî test kohortu için gözlenen farktır; farklı seed/konuşmacı/alanlarda garanti değildir.',
               'Çok zayıf CTC sonuçlarını ve Whisper baseline hata örneklerini gizleme; bunlar da aynı testin gerçek sonuçlarıdır.','']
        for row in extended_rows:
            if row['status']=='complete':
                extra.append(f"Kayıtlı uzatılmış eğitim — {row['system']}: WER %{row['original_wer_percent']:.2f} → %{row['wer_percent']:.2f}, "
                             f"CER %{row['original_cer_percent']:.2f} → %{row['cer_percent']:.2f} (seçilen adım {row['selected_update']:,}); ayrı satırdır, orijinal satır korunur.")
        with (root/'results/PRESENTATION_BRIEF_TR.md').open('a',encoding='utf-8') as stream:
            stream.write('\n'.join(extra))
    print(json.dumps({'status':report['status'],'completed':len(completed),'expected':len(rows)}),flush=True)


if __name__ == '__main__':
    main()
