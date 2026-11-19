"""Owner's independent CPU-only RNN-T value/gradient and padding review.

Enumerates every valid alignment in small synthetic lattices. Does not run
models, use corpus audio/references, or alter the submitted source files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.submission_root.resolve()
    if args.output.exists():
        raise FileExistsError("choose a new output path; preserve earlier evidence")

    import torch

    source = root / "greek_asr/models/rnnt_loss.py"
    spec = importlib.util.spec_from_file_location("submitted_rnnt_loss", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss_fn = module.rnnt_loss
    torch.set_num_threads(2)
    torch.manual_seed(1911)
    tolerance = 1e-10
    report = {
        "schema": "p3-rnnt-submission-math-review-v1",
        "submission_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source, root / "greek_asr/models/transducer.py"]
        },
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "seed": 1911, "threads": 2,
        "scope": "Synthetic exhaustive alignment values/gradients and padded-target contract only",
        "corpus_audio_or_references_used": False, "model_training_performed": False,
        "tolerance": tolerance, "batch_size": 2, "vocab_size": 5, "blank_id": 2,
        "alignment_cases": [],
    }

    for max_time, max_u in itertools.product(range(1, 5), range(0, 4)):
        logits = torch.randn(2, max_time, max_u + 1, 5, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, 4, (2, max_u))
        targets = targets + (targets >= 2)  # exclude blank=2; include label ID zero
        times = torch.tensor([max_time, max(1, max_time - 1)])
        lengths = torch.tensor([max_u, max(0, max_u - 1)])
        log_probs = logits.log_softmax(-1)
        oracle_losses = []
        path_counts = []
        for batch_index in range(2):
            paths = []
            real_time, real_u = int(times[batch_index]), int(lengths[batch_index])

            def enumerate_paths(t, u, log_probability):
                if t == real_time:
                    if u == real_u:
                        paths.append(log_probability)
                    return
                enumerate_paths(t + 1, u, log_probability + log_probs[batch_index, t, u, 2])
                if u < real_u:
                    label_id = targets[batch_index, u]
                    enumerate_paths(t, u + 1,
                                    log_probability + log_probs[batch_index, t, u, label_id])

            enumerate_paths(0, 0, logits.new_zeros(()))
            oracle_losses.append(-torch.logsumexp(torch.stack(paths), dim=0))
            path_counts.append(len(paths))
        expected = torch.stack(oracle_losses)
        actual = loss_fn(logits, targets, times, lengths, blank=2, reduction="none")
        actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
        value_error = float((actual - expected).abs().max().detach())
        gradient_error = float((actual_gradient - expected_gradient).abs().max().detach())
        report["alignment_cases"].append({
            "max_time": max_time, "max_target_length": max_u,
            "real_time_lengths": times.tolist(), "real_target_lengths": lengths.tolist(),
            "enumerated_paths_per_item": path_counts,
            "actual_losses": actual.detach().tolist(), "oracle_losses": expected.detach().tolist(),
            "max_absolute_value_error": value_error, "max_absolute_gradient_error": gradient_error,
            "finite_values_and_gradients": bool(torch.isfinite(actual).all() and
                                                torch.isfinite(actual_gradient).all()),
        })

    cases = report["alignment_cases"]
    report["max_absolute_value_error"] = max(case["max_absolute_value_error"] for case in cases)
    report["max_absolute_gradient_error"] = max(case["max_absolute_gradient_error"] for case in cases)
    report["all_alignment_checks_passed"] = all(
        case["finite_values_and_gradients"] and case["max_absolute_value_error"] <= tolerance
        and case["max_absolute_gradient_error"] <= tolerance for case in cases)

    # The submitted docstring promises targets beyond real length are never
    # read. Compare a normal in-range padding ID with an ignored sentinel.
    logits = torch.randn(2, 3, 3, 4, dtype=torch.float64)
    times, lengths = torch.tensor([3, 2]), torch.tensor([2, 1])
    baseline = loss_fn(logits, torch.tensor([[1, 2], [2, 0]]), times, lengths,
                       blank=0, reduction="none")
    padding_probe = {
        "padding_value": -100, "padded_position_is_outside_real_target_length": True,
        "baseline_in_range_padding_losses": baseline.tolist(),
    }
    try:
        result = loss_fn(logits, torch.tensor([[1, 2], [2, -100]]), times, lengths,
                         blank=0, reduction="none")
        padding_probe.update(status="returned", losses=result.tolist(),
                             contract_satisfied=bool(torch.allclose(result, baseline, atol=tolerance, rtol=0)))
    except Exception as error:
        padding_probe.update(status="raised", error_type=type(error).__name__, error=str(error),
                             contract_satisfied=False)
    report["ignored_padding_contract_probe"] = padding_probe

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"alignment_cases": len(cases),
                      "all_alignment_checks_passed": report["all_alignment_checks_passed"],
                      "max_absolute_value_error": report["max_absolute_value_error"],
                      "max_absolute_gradient_error": report["max_absolute_gradient_error"],
                      "ignored_padding_contract_satisfied": padding_probe["contract_satisfied"]}))
    return 0 if report["all_alignment_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
