"""Owner's observable full-run wrapper; submitted model algorithms stay unchanged."""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = {"ctc": "configs/kemal/ctc.json", "hubert": "configs/kemal/hubert_ctc.json",
           "attention": "configs/sequence/attention.json", "transducer": "configs/sequence/transducer.json",
           "whisper_hparl": "configs/sequence/whisper_hparl_only.json", "gmm": "configs/kemal/gmm_hmm.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=CONFIGS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/hparl_training_acceptance.json")
    args = parser.parse_args()
    from greek_asr.models import common
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or not acceptance.get("hparl_only_training_authorized"):
        raise ValueError("HParl raw-to-clean acceptance has not passed")
    selection_path = ROOT / "configs/training_data_selection.json"
    if common.hash_file(selection_path) != acceptance["selection_sha256"]:
        raise ValueError("accepted data selection changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config_path = ROOT / CONFIGS[args.job]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, artifact in (("train_manifest", "clean_manifest"), ("train_references", "clean_references")):
        expected = selection["artifacts"][artifact]
        if config[key] != expected["path"] or common.hash_file(ROOT / config[key]) != expected["sha256"]:
            raise ValueError("job does not use the accepted HParl-only training pool")
    for key, relative in (("dev_audio_manifest", config["dev_manifest"]), ("dev_references", config["dev_references"])):
        if common.hash_file(ROOT / relative) != acceptance["reproduced_artifacts"][key]["expected_sha256"]:
            raise ValueError("accepted dev input changed")
    for relative, digest in acceptance["frozen_source_before"].items():
        if common.hash_file(ROOT / relative) != digest:
            raise ValueError("frozen original test metadata changed")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "greek_asr", "scripts", "configs"], cwd=ROOT).returncode:
        raise ValueError("commit runtime code/configs before full training")
    if args.output.exists():
        raise FileExistsError("never overwrite a previous training run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    live_path = args.output.with_name(args.output.name + ".live.json")
    state = {"schema": "p3-owner-live-training-v1", "status": "starting", "job": args.job, "pid": os.getpid(),
             "created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": common.git_commit_hash(ROOT),
             "config": CONFIGS[args.job], "config_sha256": common.hash_file(config_path),
             "acceptance_sha256": common.hash_file(args.acceptance), "optimizer_updates": 0,
             "expected_optimizer_updates": config.get("max_steps"), "final_test_used": False,
             "runtime_env": {key: os.environ.get(key) for key in ("HSA_ENABLE_DXG_DETECTION", "MIOPEN_FIND_MODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    started = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        temporary = live_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, live_path)

    save()
    try:
        import torch
        torch.set_num_threads(2)
        cpu = args.job == "gmm"
        if not cpu:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU required; no CPU fallback")
            torch.cuda.set_per_process_memory_fraction(.85)
            state["gpu"] = torch.cuda.get_device_name(0)
            state["torch"] = torch.__version__
            state["hip"] = torch.version.hip
        state["status"] = "running"
        save()
        module_name = {"whisper_hparl": "whisper_finetune", "gmm": "gmm_hmm"}.get(args.job, args.job)
        module = importlib.import_module("greek_asr.models." + module_name)
        if cpu:
            original_frontend = module.MfccFrontend.__call__
            def observed_features(frontend, *positional, **keywords):
                result = original_frontend(frontend, *positional, **keywords)
                state["feature_utterances_completed"] = state.get("feature_utterances_completed", 0) + 1
                if state["feature_utterances_completed"] % 1000 == 0:
                    save()
                    print(json.dumps({"job": args.job, "phase": "features", "utterances": state["feature_utterances_completed"]}), flush=True)
                return result
            module.MfccFrontend.__call__ = observed_features
            module.train(config, project_root=ROOT, output_dir=args.output, device="cpu", progress=True)
        else:
            original_clip = torch.nn.utils.clip_grad_norm_
            @functools.wraps(original_clip)
            def finite_clip(*positional, **keywords):
                keywords["error_if_nonfinite"] = True
                norm = original_clip(*positional, **keywords)
                state["_last_gradient_norm"] = norm.detach()
                return norm
            torch.nn.utils.clip_grad_norm_ = finite_clip
            original_step = torch.optim.AdamW.step
            @functools.wraps(original_step)
            def observed_step(optimizer, *positional, **keywords):
                result = original_step(optimizer, *positional, **keywords)
                state["optimizer_updates"] += 1
                count = state["optimizer_updates"]
                if count in (1, 10, 25) or count % 50 == 0:
                    torch.cuda.synchronize()
                    norm = state.pop("_last_gradient_norm", None)
                    if norm is not None:
                        state["last_gradient_norm"] = float(norm)
                    state["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    save()
                    print(json.dumps({"job": args.job, "optimizer_updates": count,
                                      "target": config["max_steps"], "elapsed_seconds": round(state["elapsed_seconds"], 1),
                                      "finite_gradient_norm": state.get("last_gradient_norm")}), flush=True)
                # Tensors must never enter the JSON progress file.
                state.pop("_last_gradient_norm", None)
                return result
            torch.optim.AdamW.step = observed_step
            module.train(config, project_root=ROOT, output_dir=args.output, device="cuda")
            if state["optimizer_updates"] != config["max_steps"]:
                raise RuntimeError("actual optimizer updates differ from the committed budget")
        state["status"] = "complete"
    except Exception as error:
        state.pop("_last_gradient_norm", None)
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state.pop("_last_gradient_norm", None)
        save()


if __name__ == "__main__":
    main()
