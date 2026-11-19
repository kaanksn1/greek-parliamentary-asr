"""Pure PyTorch/autograd RNN-T loss (forward algorithm, log-space).

Why this exists: Serdar's real RX 7800 XT (ROCm) check found that
``torchaudio.functional.rnnt_loss`` -- the backend ``transducer.py`` used
before this fix -- raises ``NotImplementedError: Could not run
'torchaudio::rnnt_loss' ... 'CUDA' backend`` on that machine's AMD wheel (see
docs/reviews/ADNAN_0437631_GPU.json, finding 3). The op is a custom compiled
kernel that this environment only has a CPU build of; a CPU-only test passing
is not evidence it runs during real GPU training.

This module replaces that compiled op with a loss built entirely from
ordinary ``torch.autograd`` primitives (``log_softmax``, ``gather``,
``logaddexp``, elementwise add/compare). This avoids the separate compiled
RNN-T operator dependency; it does not establish correctness or availability
on every device, dtype or library build. Real forward/backward and optimizer
checks on the target device remain necessary when the implementation or
environment changes. See docs/SEQUENCE_MODELS.md "RNN-T loss backend".

Correctness evidence (not assumed from the algorithm description alone):
tests/test_rnnt_loss.py checks this function's value and gradient against
(1) a hand-enumerated two-path lattice solved independently of any library,
and (2) ``torchaudio.functional.rnnt_loss`` on CPU, for both a fixed small
example and multiple random batches with variable per-sample lengths.

At submission commit 0c13ce1, a real RX 7800 XT train-pilot check completed
with finite gradients and a parameter update; that versioned evidence is in
docs/reviews/ADNAN_0c13ce1_GPU.json. It does not prove throughput or memory
feasibility for full training. This is a straightforward O(time * label)
sequential-over-time, batch-vectorized dynamic program, not the
anti-diagonal/compiled-kernel optimization used by torchaudio's operator.
Representative train/dev measurements are still required before selecting
a full training budget.
"""
from __future__ import annotations

_NEG_INF = -1.0e9


def rnnt_loss(logits: "torch.Tensor", targets: "torch.Tensor", logit_lengths: "torch.Tensor",  # noqa: F821
              target_lengths: "torch.Tensor", blank: int, reduction: str = "mean") -> "torch.Tensor":  # noqa: F821
    """Signature-compatible drop-in for ``torchaudio.functional.rnnt_loss``.

    logits: ``(batch, max_time, max_target_len + 1, vocab)`` joint-network
        output (pre-softmax).
    targets: ``(batch, max_target_len)`` label ids; padding at/after
        ``target_lengths`` is never read.
    logit_lengths: ``(batch,)`` real encoder frame counts.
    target_lengths: ``(batch,)`` real label counts.
    """
    import torch

    if logits.dim() != 4:
        raise ValueError("logits must be shaped (batch, time, target_len + 1, vocab)")
    batch, max_time, max_u1, _vocab = logits.shape
    max_u = max_u1 - 1
    if tuple(targets.shape) != (batch, max_u):
        raise ValueError(f"targets must be shaped (batch, {max_u}), got {tuple(targets.shape)}")
    device, dtype = logits.device, logits.dtype

    log_probs = torch.log_softmax(logits, dim=-1)
    blank_lp = log_probs[..., blank]  # (batch, max_time, max_u + 1)
    if max_u > 0:
        label_index = torch.arange(max_u, device=device).view(1, max_u)
        valid_label = label_index < target_lengths.view(batch, 1)  # (batch, max_u)
        # Ignored padding may contain a sentinel outside the vocabulary. Make
        # only those positions safe before gather; invalid real labels must
        # still raise instead of being silently mapped to another token.
        safe_targets = targets.long().masked_fill(~valid_label, 0)
        index = safe_targets.view(batch, 1, max_u, 1).expand(batch, max_time, max_u, 1)
        label_lp = torch.gather(log_probs[:, :, :max_u, :], dim=-1, index=index).squeeze(-1)  # (batch, max_time, max_u)
    else:
        label_lp = torch.zeros(batch, max_time, 0, dtype=dtype, device=device)

    # A frame/label index beyond a given sample's real length is padding, not
    # data: mask it to -inf so no path can use it. Without this, a shorter
    # sample in the same batch as a longer one would silently score an extra,
    # physically-impossible path through zero-padded logits.
    time_index = torch.arange(max_time, device=device).view(1, max_time)
    valid_time = time_index < logit_lengths.view(batch, 1)  # (batch, max_time)
    neg_inf = torch.full((), _NEG_INF, dtype=dtype, device=device)
    blank_lp = torch.where(valid_time.unsqueeze(-1), blank_lp, neg_inf)
    if max_u > 0:
        valid_label_lp = valid_time.unsqueeze(-1) & valid_label.unsqueeze(1)
        label_lp = torch.where(valid_label_lp, label_lp, neg_inf)

    alpha = torch.full((batch, max_time + 1, max_u + 1), _NEG_INF, dtype=dtype, device=device)
    alpha[:, 0, 0] = 0.0
    for u in range(1, max_u + 1):
        alpha[:, 0, u] = alpha[:, 0, u - 1] + label_lp[:, 0, u - 1]
    for t in range(1, max_time + 1):
        from_blank = alpha[:, t - 1, :] + blank_lp[:, t - 1, :]  # (batch, max_u + 1)
        if t <= max_time - 1:
            # Row t can still emit a label at frame index t (a valid joint-
            # network row), so each column also depends on the column before
            # it in this same row -- a genuine sequential dependency on u.
            row = from_blank.clone()
            for u in range(1, max_u + 1):
                row[:, u] = torch.logaddexp(from_blank[:, u], row[:, u - 1] + label_lp[:, t, u - 1])
            alpha[:, t, :] = row
        else:
            # t == max_time: there is no joint-network row at this index (logits
            # only has rows 0..max_time-1), so this row is reachable only by a
            # final blank from row t - 1; no within-row label transition exists.
            alpha[:, t, :] = from_blank

    final_t, final_u = logit_lengths.long(), target_lengths.long()
    batch_index = torch.arange(batch, device=device)
    losses = -alpha[batch_index, final_t, final_u]

    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"unsupported reduction {reduction!r}")
