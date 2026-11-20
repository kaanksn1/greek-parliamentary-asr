"""Run independent CPU GMM and sequential GPU HParl jobs; never run final test."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema": "p3-primary-training-queue-v1", "status": "running", "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": {},
              "mixed_whisper": "waiting_for_corrected_CSS10_acceptance", "final_test_authorized": False}

    def save():
        temporary = output / "queue.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output / "queue.json")

    def launch(job, cpu=False):
        environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           HSA_ENABLE_DXG_DETECTION="0" if cpu else "1", MIOPEN_FIND_MODE="FAST",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        log = (output / (job + ".log")).open("w", encoding="utf-8")
        command = [sys.executable, str(ROOT / "scripts/run_training_job.py"), "--job", job,
                   "--output", str(output / job)]
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status["jobs"][job] = {"pid": process.pid, "status": "running", "command": command,
                                "log": str(output / (job + ".log")), "started_utc": datetime.now(timezone.utc).isoformat()}
        save()
        print(json.dumps({"started_job": job, "pid": process.pid}), flush=True)
        return process, log

    cpu_process, cpu_log = launch("gmm", cpu=True)
    save()
    try:
        for job in ("ctc", "whisper_hparl", "attention", "hubert", "transducer"):
            process, log = launch(job)
            try:
                while process.poll() is None:
                    if cpu_process.poll() is not None:
                        status["jobs"]["gmm"].update(status="complete" if cpu_process.returncode == 0 else "failed", exit_code=cpu_process.returncode)
                    save()
                    time.sleep(5)
            finally:
                log.close()
            status["jobs"][job].update(status="complete" if process.returncode == 0 else "failed", exit_code=process.returncode,
                                      ended_utc=datetime.now(timezone.utc).isoformat())
            save()
        while cpu_process.poll() is None:
            save()
            time.sleep(5)
        status["jobs"]["gmm"].update(status="complete" if cpu_process.returncode == 0 else "failed", exit_code=cpu_process.returncode)
        status["status"] = "primary_training_complete" if all(job["status"] == "complete" for job in status["jobs"].values()) else "primary_training_has_failures"
    finally:
        cpu_log.close()
        save()


if __name__ == "__main__":
    main()
