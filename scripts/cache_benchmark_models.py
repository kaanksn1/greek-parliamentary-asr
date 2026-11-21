"""Fetch the exact upstream checkpoints before the deliberately offline test queue."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-only',action='store_true',help='Use existing files only; no network or GPU inference')
    parser.add_argument('--include-training-encoder',action='store_true',help='Also cache the bare mHuBERT encoder for reproducing training, not ASR inference')
    args=parser.parse_args()
    root=args.root.resolve()
    from huggingface_hub import hf_hub_download, snapshot_download
    lock=json.loads((root/'configs/model_sources.lock.json').read_text(encoding='utf-8-sig'))
    entries={r['model_id']:r for r in lock['models']}
    chosen={'whisper':'openai/whisper-small','wav2vec':'lighteternal/wav2vec2-large-xlsr-53-greek',
            'qwen':'Qwen/Qwen3-ASR-0.6B','canary':'nvidia/canary-1b-v2','parakeet':'nvidia/parakeet-tdt-0.6b-v3'}
    if args.include_training_encoder:
        extra=json.loads((root/'configs/kemal_model_sources.json').read_text(encoding='utf-8-sig'))
        entries.update({r['model_id']:r for r in extra['models']})
        chosen['hubert-encoder']='utter-project/mHuBERT-147'
    records=[]
    for backend,model_id in chosen.items():
        entry=entries[model_id]
        revision=entry['revision']
        if not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('model revision must be immutable')
        folder=root/'models'/(backend+'-'+revision[:12])
        folder.mkdir(parents=True,exist_ok=True)
        options={'repo_id':model_id,'revision':revision,'token':False,'local_files_only':args.check_only}
        if backend in ('canary','parakeet'):
            filename='canary-1b-v2.nemo' if backend=='canary' else 'parakeet-tdt-0.6b-v3.nemo'
            artifact=Path(hf_hub_download(**options,filename=filename,cache_dir=str(folder)))
            if artifact.stat().st_size<100_000_000:
                raise ValueError('missing/invalid cached NeMo archive')
        else:
            patterns=['*.json','*.txt','*.safetensors']
            if backend=='wav2vec':
                patterns=['config.json','preprocessor_config.json','vocab.json','tokenizer_config.json',
                          'special_tokens_map.json','added_tokens.json','model.safetensors','pytorch_model.bin']
            if backend=='whisper':
                # Fine-tuning's standard loader uses the global HF snapshot cache.
                snapshot_download(**options,allow_patterns=patterns)
            local=Path(snapshot_download(**options,local_dir=str(folder),allow_patterns=patterns))
            for name in ('config.json','preprocessor_config.json'):
                if not (local/name).is_file():
                    raise FileNotFoundError('missing processor/config: '+str(local/name))
            if not any(local.glob('*.safetensors')) and not (local/'pytorch_model.bin').is_file():
                raise FileNotFoundError('cached ASR weights missing: '+str(local))
            artifact=local
        record={'backend':backend,'model_id':model_id,'revision':revision,'cache':str(artifact),'status':'available'}
        records.append(record)
        print(json.dumps(record),flush=True)
    print(json.dumps({'status':'ready','upstream_models':len(records),'check_only':args.check_only,
                      'test_inference_performed':False,'group_trained_weights':'provided separately by repository Git LFS'}))


if __name__=='__main__':
    main()
