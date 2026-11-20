"""Run the selected-scope data chain on real audio and record every command's output.

Samed's acceptance run for the CSS10 IEEE float32 fix (handoff 20 Nov 2026):
source-metadata and freeze checks, selected-scope audit -> clean (CSS10 +
mixture + lineage) -> verify, the requested CSS10/dev verify, the generated
report and the whole unittest suite. Nothing is skipped silently: each command,
exit code, duration, stdout and stderr is written to the evidence directory.
The script never uses ``--skip-audio`` and never writes audio.

    python scripts/run_selected_data_checks.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = "configs/training_data_selection.json"
STEPS = [
    ("verify_source_metadata", ["-m", "greek_asr.data", "verify-source-metadata"]),
    ("verify_freeze_with_audio", ["-m", "greek_asr.data", "verify-freeze"]),
    ("audit_selected", ["-m", "greek_asr.data", "audit", "--selection", SELECTION]),
    ("clean_selected", ["-m", "greek_asr.data", "clean", "--selection", SELECTION]),
    ("verify_selected_with_audio", ["-m", "greek_asr.data", "verify", "--selection", SELECTION]),
    ("verify_css10_dev_with_audio", ["-m", "greek_asr.data", "verify", "--only", "css10_el_train.jsonl", "dev.jsonl"]),
    ("report", ["-m", "greek_asr.data", "report"]),
    ("unittest", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
]


def git(*arguments: str) -> str:
    try:
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", default="docs/reviews/css10_reaudit_2026-11-20")
    args = parser.parse_args()
    evidence = ROOT / args.evidence
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    summary = {"schema": "p3-data-acceptance-run-v1", "git_head": git("rev-parse", "HEAD"),
               "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
               "python": platform.python_version(), "numpy": numpy_version, "system": platform.system(),
               "machine": platform.machine(), "skip_audio_used": False, "steps": []}
    environment = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    for number, (name, arguments) in enumerate(STEPS, 1):
        command = [sys.executable, *arguments]
        print(f"[{number}/{len(STEPS)}] {' '.join(['python', *arguments])}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                errors="backslashreplace", env={**os.environ, **environment})
        seconds = round(time.monotonic() - started, 1)
        log = evidence / f"{number:02d}_{name}.txt"
        log.write_text(f"$ python {' '.join(arguments)}\n# exit code {result.returncode}, {seconds} s\n\n"
                       f"## stdout\n{result.stdout}\n## stderr\n{result.stderr}", encoding="utf-8", newline="\n")
        step = {"name": name, "command": "python " + " ".join(arguments), "exit_code": result.returncode,
                "seconds": seconds, "log": log.relative_to(ROOT).as_posix()}
        if name == "unittest":
            tail = [line for line in result.stderr.splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
            step["result_lines"] = tail
            step["skipped_reasons"] = {}
            for line in result.stderr.splitlines():
                if " ... skipped " in line:
                    reason = line.split(" ... skipped ", 1)[1].strip().strip("'")
                    step["skipped_reasons"][reason] = step["skipped_reasons"].get(reason, 0) + 1
        summary["steps"].append(step)
        print(f"    exit {result.returncode} ({seconds} s) -> {step['log']}", flush=True)
    summary["all_passed"] = all(step["exit_code"] == 0 for step in summary["steps"])
    (evidence / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
    print("ALL PASSED" if summary["all_passed"] else "SOME STEPS FAILED", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
