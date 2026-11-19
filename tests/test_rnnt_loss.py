"""Correctness evidence for greek_asr.models.rnnt_loss (the pure PyTorch/autograd
RNN-T loss chosen in docs/SEQUENCE_MODELS.md "RNN-T loss backend" to replace
torchaudio.functional.rnnt_loss, whose compiled CUDA/ROCm kernel is missing in
the project's AMD wheel -- see docs/reviews/ADNAN_0437631_GPU.json finding 3).

Two independent checks, not just one: a lattice solved by hand (enumerating
every valid alignment, with no library involved) and a value+gradient
comparison against torchaudio's own CPU kernel across several random,
variable-length batches.
"""
from __future__ import annotations

import unittest

from greek_asr.models import rnnt_loss as pure

try:
    import torch
    from torchaudio.functional import rnnt_loss as torchaudio_rnnt_loss
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

_SKIP_REASON = "requires the .venv-sequence environment (torch/torchaudio); see requirements-sequence.txt"


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class HandEnumeratedLatticeTests(unittest.TestCase):
    def test_two_path_lattice_matches_independent_enumeration(self):
        """batch=1, 2 frames, 1 label, vocab={blank, label}: exactly two valid
        alignments exist (label-then-blank-blank, or blank-label-blank; a
        trailing blank-blank-label is impossible since no frame is left to
        attach the final label to). Sum them by hand, with no library call.
        """
        torch.manual_seed(0)
        logits = torch.randn(1, 2, 2, 2, dtype=torch.float64)
        targets = torch.tensor([[1]])
        logit_lengths = torch.tensor([2])
        target_lengths = torch.tensor([1])

        log_probs = torch.log_softmax(logits, dim=-1)
        blank_lp = lambda t, u: log_probs[0, t, u, 0]
        label_lp = lambda t, u: log_probs[0, t, u, 1]
        path_label_first = label_lp(0, 0) + blank_lp(0, 1) + blank_lp(1, 1)
        path_blank_first = blank_lp(0, 0) + label_lp(1, 0) + blank_lp(1, 1)
        expected_loss = -torch.logaddexp(path_label_first, path_blank_first)

        got = pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="none")
        self.assertTrue(torch.allclose(got, expected_loss.unsqueeze(0), atol=1e-6))


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class MatchesTorchaudioCpuTests(unittest.TestCase):
    def _compare(self, batch, max_time, max_u, vocab, seed):
        torch.manual_seed(seed)
        logits_pure = torch.randn(batch, max_time, max_u + 1, vocab, dtype=torch.float32, requires_grad=True)
        logits_ta = logits_pure.detach().clone().requires_grad_(True)
        targets = torch.randint(1, vocab, (batch, max_u))  # never emit blank(0) as a target id
        target_lengths = torch.randint(1, max_u + 1, (batch,))
        logit_lengths = torch.stack(
            [torch.randint(int(length), max_time + 1, (1,)).squeeze(0) for length in target_lengths])
        logit_lengths[-1], target_lengths[-1] = max_time, max_u  # at least one full-length row, like a real batch

        loss_pure = pure.rnnt_loss(logits_pure, targets, logit_lengths, target_lengths, blank=0, reduction="mean")
        loss_ta = torchaudio_rnnt_loss(logits_ta, targets.to(torch.int32), logit_lengths.to(torch.int32),
                                       target_lengths.to(torch.int32), blank=0, reduction="mean")
        loss_pure.backward()
        loss_ta.backward()
        self.assertTrue(torch.isfinite(loss_pure))
        self.assertTrue(torch.allclose(loss_pure, loss_ta, atol=1e-4),
                        f"value mismatch: pure={loss_pure.item()} torchaudio={loss_ta.item()}")
        self.assertTrue(torch.allclose(logits_pure.grad, logits_ta.grad, atol=1e-4),
                        "gradient mismatch against torchaudio's CPU kernel")

    def test_small_fixed_shapes(self):
        for batch, max_time, max_u, vocab, seed in [
            (1, 2, 1, 2, 1), (2, 3, 2, 3, 2), (3, 5, 4, 5, 3), (4, 8, 6, 4, 4),
        ]:
            with self.subTest(batch=batch, max_time=max_time, max_u=max_u, vocab=vocab):
                self._compare(batch, max_time, max_u, vocab, seed)

    def test_random_variable_length_batches(self):
        for seed in range(5, 15):
            with self.subTest(seed=seed):
                self._compare(2, 6, 5, 4, seed)

    def test_single_item_batch_with_no_padding(self):
        self._compare(1, 4, 3, 6, 99)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class IgnoredPaddingTests(unittest.TestCase):
    def test_out_of_vocabulary_padding_preserves_loss_and_gradients(self):
        """Ignored sentinels must behave exactly like ordinary zero padding,
        including for a sample with an empty target sequence.
        """
        torch.manual_seed(1911)
        logits = torch.randn(3, 4, 4, 5, dtype=torch.float64)
        logit_lengths = torch.tensor([4, 2, 3])
        target_lengths = torch.tensor([3, 1, 0])
        baseline_logits = logits.clone().requires_grad_(True)
        baseline = pure.rnnt_loss(
            baseline_logits, torch.tensor([[1, 2, 3], [2, 0, 0], [0, 0, 0]]),
            logit_lengths, target_lengths, blank=0, reduction="none")
        baseline.sum().backward()
        self.assertTrue(torch.isfinite(baseline).all())
        self.assertTrue(torch.isfinite(baseline_logits.grad).all())

        for sentinel in (-100, 5, 2 ** 31 - 1):
            with self.subTest(padding=sentinel):
                padded_logits = logits.clone().requires_grad_(True)
                targets = torch.tensor([[1, 2, 3], [2, sentinel, sentinel],
                                        [sentinel, sentinel, sentinel]])
                actual = pure.rnnt_loss(padded_logits, targets, logit_lengths,
                                        target_lengths, blank=0, reduction="none")
                actual.sum().backward()
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.isfinite(padded_logits.grad).all())
                self.assertTrue(torch.allclose(actual, baseline, atol=1e-12, rtol=0))
                self.assertTrue(torch.allclose(padded_logits.grad, baseline_logits.grad,
                                               atol=1e-12, rtol=0))

    def test_out_of_vocabulary_real_targets_are_not_hidden_as_padding(self):
        logits = torch.randn(2, 4, 4, 5)
        for invalid_label in (-100, 5):
            with self.subTest(label=invalid_label):
                targets = torch.tensor([[1, 2, 3], [invalid_label, 0, 0]])
                with self.assertRaises(RuntimeError):
                    pure.rnnt_loss(logits, targets, torch.tensor([4, 2]),
                                   torch.tensor([3, 1]), blank=0)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class DeviceAgnosticConstructionTests(unittest.TestCase):
    def test_runs_entirely_from_ops_available_on_meta_device(self):
        """No part of this loss depends on a backend-specific compiled kernel:
        every op it uses (log_softmax, gather, logaddexp, elementwise add/compare,
        torch.where) has a 'meta' (shape-only, backend-free) implementation.
        This checks shape-only execution, not real device availability,
        backward correctness or training throughput on ROCm/CUDA hardware.
        """
        logits = torch.randn(2, 4, 3, 5, device="meta", requires_grad=True)
        targets = torch.randint(1, 5, (2, 2), device="meta")
        logit_lengths = torch.tensor([4, 3], device="meta")
        target_lengths = torch.tensor([2, 1], device="meta")
        loss = pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="mean")
        self.assertEqual(loss.device.type, "meta")
        self.assertEqual(loss.shape, ())

    def test_reduction_modes(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 3, 3)
        targets = torch.tensor([[1, 2], [1, 0]])
        logit_lengths = torch.tensor([3, 2])
        target_lengths = torch.tensor([2, 1])
        none = pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="none")
        total = pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="sum")
        mean = pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="mean")
        self.assertEqual(none.shape, (2,))
        self.assertTrue(torch.allclose(total, none.sum()))
        self.assertTrue(torch.allclose(mean, none.mean()))
        with self.assertRaises(ValueError):
            pure.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0, reduction="bogus")


if __name__ == "__main__":
    unittest.main()
