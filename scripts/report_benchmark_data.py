"""Report actual selected cohorts and literal/final-policy overlap; never edit data."""
import hashlib
import collections
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
    cleaning_dir='data/clean/selected_train_hparl_30.5h_seed42/'
    cleaning=json.loads((ROOT/cleaning_dir/'clean_summary.json').read_text())
    log=rows(cleaning_dir+'cleaning_log.jsonl')
    dropped=[r for r in log if r['action']=='drop']
    removed={r['id'] for r in splits['raw_hparl_train']}-{r['id'] for r in splits['hparl_train']}
    assert {r['id'] for r in dropped}==removed
    assert sha(cleaning_dir+'cleaning_log.jsonl')==cleaning['log_sha256']
    report['cleaning']={'log':cleaning_dir+'cleaning_log.jsonl','log_sha256':cleaning['log_sha256'],
        'removed_records':len(dropped),'dropped_by_rule':dict(collections.Counter(r['rule'] for r in dropped)),
        'text_policy_log_entries':cleaning['log_entries'].get('modify_text',0),
        'human_review_decisions':cleaning['review_decisions_applied'],'review_queue_records':cleaning['review_queue_records']}
    selected=json.loads((ROOT/'configs/selected_checkpoints.json').read_text())
    report['audio_plus_selected_weights_gib']=selected['audio_plus_weights_gib']
    output = ROOT/'results/data_integrity.json'
    output.write_bytes((json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    text=['# Kullanılan veri ve gerçek teslim raporu','',
        'Bu rapor `python scripts/report_benchmark_data.py` ile gerçek seçili manifestler ve temizleme günlüklerinden üretilir; sayıları elle değiştirmeyin.',
        'Kapsam: eğitimde gerçekten kullanılan seçili HParl, eğitim-only CSS10 ve değişmeyen resmî dev/test. Tam kaynak kataloğunun kullanılmayan ~99 saatlik train envanteri bu teslimin ses kapsamı değildir.','',
        '| Kohort | Kayıt | Kodlanmış konuşmacı | Oturum/kayıt ID | Saat |',
        '|---|---:|---:|---:|---:|']
    for name,cohort in report['cohorts'].items():
        text.append(f"| {name} | {cohort['utterances']} | {cohort['speakers_with_ids']} | {cohort['recordings_with_ids']} | {cohort['hours']:.6f} |")
    text+=['',report['css10_speaker_note'],
        'CSS10 üzerinde yeni train/dev/test bölmesi yapılmadı; kaynağın tamamı yalnız train kullanılır. HParl için üreticinin resmî oturum bölmesi korunur.','',
        '## Gerçek temizlik ve izlenebilirlik','',
        f"Seçili ham HParl ile temiz HParl arasındaki {len(removed)} çıkarılan ID, kayıttaki drop kararlarıyla birebir eşleşir.", '',
        '| Çıkarma kuralı | Kayıt |','|---|---:|']
    for rule,count in sorted(report['cleaning']['dropped_by_rule'].items()): text.append(f'| {rule} | {count} |')
    text+=['',f"Metin politikası günlük girdisi: {report['cleaning']['text_policy_log_entries']}. Bunlar annotation tag/karakter/harf/boşluk uyumlamasını kaydeder; formatlama tek başına gerçek veri temizliği veya yeni gold anotasyon sayılmaz.",
        f"İnsan inceleme kararı: {report['cleaning']['human_review_decisions']}; inceleme kuyruğu: {report['cleaning']['review_queue_records']} kayıt. İncelenmemiş flag'ler doğrulanmış etiket hatası diye sunulmaz.",
        'Orijinal WAV baytları korunur. Temiz veri, aynı gerçek WAV dosyalarını seçen temiz manifest ve referanslardan oluşur. CSS10 float32 doğru format tag ile okunur; gain/ses normalizasyonu yoktur.',
        f"Tam nedenler, before/after metin/hash ve kanıt: `{report['cleaning']['log']}`. Günlük SHA-256: `{report['cleaning']['log_sha256']}`.",'',
        '## Resmî split ve örtüşmeler','',
        '| Train koşulu → test | Ortak konuşmacı | Bu seslere ait test kaydı | Ortak oturum | Ortak ses hash | Literal cümle | Final normalize cümle |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,overlap in report['training_to_test_overlap'].items():
        text.append(f"| {name} | {overlap['speakers']} | {overlap['test_clips_with_shared_speaker']} | {overlap['recordings']} | {overlap['shared_audio_sha256_values']} | {overlap['literal_matching_test_utterances']} | {overlap['final_policy_matching_test_utterances']} |")
    text+=['','Bu benchmark speaker-disjoint değildir ve yeni konuşmacılara genelleme diye sunulamaz. Dokümante resmî split istisnası kullanılır; yeni rastgele klip bölmesi yapılmamıştır. Literal ve normalize eşleşme ayrı ölçümlerdir.',
        'Test referans/sesleri değiştirilmez, zor kayıt çıkarılmaz. Her sistem aynı orijinal test IDsini ve tek dondurulmuş scoring politikasını kullanır.','',
        '## Depo teslimi ve doğrulama','',
        'Seçili ham HParl WAVlarının tamamı (çıkarılanlar dahil), dev/test/CSS10 WAVları Git LFS ile depodadır. Ham metaveri ve ham/temiz manifest/referanslar normal Git içindedir. Drive veya indirme betiği veri teslimi yerine kullanılmaz.',
        f"Özgün ses + seçili inference ağırlıkları: {report['audio_plus_selected_weights_gib']:.6f} GiB; 10 GiB depolama sınırı içinde. Aylık indirme kotası ayrı bir hesap sınırıdır; bu rapor kalan bandwidth'i ölçmez.",
        'Temiz public clone, mevcut yüklenmiş LFS nesne önbelleğiyle gerçek ses üzerinde doğrulandı (bant genişliği için yeniden indirilmedi). Kanıt: `docs/reviews/final_benchmark_2026-11-21/clean_clone_verification.json`. Anonim örnek nesne erişimi: aynı klasörde `public_delivery.json`.',
        'Kaynak kimliği/lisans: `DATA_LICENSE.md`, `configs/data_source.json`. Dondurulmuş seçim: `configs/training_data_selection.json`. Makine-okunur tüm sayılar ve manifest hashleri: `results/data_integrity.json`.','',
        '## Tekrar üretim','', '```bash',
        'python -m greek_asr.data verify-source-metadata',
        'python -m greek_asr.data audit --selection configs/training_data_selection.json',
        'python -m greek_asr.data clean --selection configs/training_data_selection.json',
        'python -m greek_asr.data verify --selection configs/training_data_selection.json',
        'python -m greek_asr.data verify-freeze',
        'python scripts/report_benchmark_data.py','```','',
        'Tarihsel `python -m greek_asr.data report` komutu tam-kaynak/envanter raporudur ve bu dosyaya yazar; mevcut benchmark için son komut olarak yukarıdaki seçili-kohort rapor üreticisini çalıştırın. Eski envanter boyutlarını teslim edilmiş eğitim saati olarak kullanmayın.','']
    (ROOT/'docs/DATA_REPORT_TR.md').write_bytes(('\n'.join(text)).encode('utf-8'))
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':
    main()
