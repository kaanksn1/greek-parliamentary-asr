"""Standard MFCC + GMM-HMM ASR baseline: grapheme HMMs, Viterbi training and a
grapheme-loop decoder with a character n-gram language model.

Implementation choice: plain PyTorch/torchaudio (no Kaldi/HTK install). The
recipe follows the classical monophone pipeline (Young et al., HTK Book;
Povey et al., Kaldi ``train_mono.sh``), with graphemes as the acoustic units
because Greek orthography is close to phonemic and no licensed Greek
pronunciation lexicon is part of the project data.

- Features: Kaldi-compatible MFCC (``torchaudio.compliance.kaldi.mfcc``:
  25 ms window, 10 ms shift, 23 mel bins, 13 cepstra, no dither) + delta +
  delta-delta = 39 dims, per-utterance cepstral mean normalization.
- Acoustic units: every non-space character of the train references plus a
  silence unit; each unit is a 3-state left-to-right HMM whose states have
  diagonal-covariance Gaussian mixtures.
- Lexicon: grapheme lexicon (a word is its letter sequence). Optional silence
  between words and at both utterance ends.
- Training: flat start (global mean/variance), equal alignment, then repeated
  batched Viterbi forced alignment + per-state mixture EM, with Gaussian
  splitting at configured iterations and variance flooring.
- Decoder: frame-synchronous Viterbi over (character history, unit, state).
  The character n-gram LM (Witten-Bell interpolation, order 2-3) is trained
  only on the train references; word boundaries are LM space tokens realised
  either by the silence unit or by a zero-duration skip. Output is text.
  LM weight and insertion penalty are chosen on dev with ``tune``.

The decoder has no word vocabulary: it is an open-vocabulary grapheme
recognizer, weaker than a word-lexicon/word-LM Kaldi system. Training items
with fewer frames than mandatory HMM states are reported and handled by the
configured policy, never silently skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from greek_asr.models import common, ctc_utils

ROOT = Path(__file__).resolve().parents[2]
SIL = "<sil>"
SCHEMA = "greek-asr-gmm-hmm-v1"


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    num_ceps: int = 13
    num_mel_bins: int = 23
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    deltas: bool = True
    cmn: str = "utterance_mean"

    @property
    def dim(self) -> int:
        return self.num_ceps * (3 if self.deltas else 1)


@dataclass(frozen=True)
class TrainConfig:
    states_per_unit: int = 3
    num_iterations: int = 25
    split_iterations: tuple[int, ...] = (2, 4, 6, 8)
    max_gaussians_per_state: int = 16
    variance_floor_fraction: float = 0.01
    min_gaussian_occupancy: float = 20.0
    initial_self_loop: float = 0.5
    self_loop_bounds: tuple[float, float] = (0.05, 0.95)
    align_batch_cells: int = 20_000_000
    lm_order: int = 3
    infeasible_policy: str = "exclude_and_report"
    max_excluded_fraction: float = 0.01


@dataclass(frozen=True)
class DecoderSettings:
    lm_weight: float = 10.0
    insertion_penalty: float = 0.0
    space_skip_logprob: float = 0.0


# --------------------------------------------------------------------------- #
# Features.
# --------------------------------------------------------------------------- #


class MfccFrontend:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def __call__(self, waveform):
        import torch
        import torchaudio

        cfg = self.config
        if waveform.dim() != 1:
            raise ValueError("expected a mono waveform")
        # Kaldi works on int16-scale samples; the constant scale only shifts c0,
        # which per-utterance mean normalization removes anyway.
        ceps = torchaudio.compliance.kaldi.mfcc(
            (waveform * 32768.0).unsqueeze(0), sample_frequency=cfg.sample_rate, num_ceps=cfg.num_ceps,
            num_mel_bins=cfg.num_mel_bins, frame_length=cfg.frame_length_ms, frame_shift=cfg.frame_shift_ms,
            dither=0.0, energy_floor=0.0, snip_edges=True)
        if ceps.shape[0] == 0:
            raise ValueError("audio is shorter than one analysis window")
        features = [ceps]
        if cfg.deltas:
            delta = torchaudio.functional.compute_deltas(ceps.T.unsqueeze(0), win_length=5).squeeze(0).T
            delta2 = torchaudio.functional.compute_deltas(delta.T.unsqueeze(0), win_length=5).squeeze(0).T
            features += [delta, delta2]
        output = torch.cat(features, dim=1)
        if cfg.cmn == "utterance_mean":
            output = output - output.mean(dim=0, keepdim=True)
        elif cfg.cmn != "none":
            raise ValueError("cmn must be 'utterance_mean' or 'none'")
        return output.contiguous()

    def num_frames(self, num_samples: int) -> int:
        window = int(self.config.sample_rate * self.config.frame_length_ms / 1000)
        shift = int(self.config.sample_rate * self.config.frame_shift_ms / 1000)
        return 0 if num_samples < window else 1 + (num_samples - window) // shift


# --------------------------------------------------------------------------- #
# Units, transcripts and alignment chains.
# --------------------------------------------------------------------------- #


@dataclass
class UnitInventory:
    graphemes: list[str]
    states_per_unit: int

    @property
    def units(self) -> list[str]:
        return self.graphemes + [SIL]

    @property
    def sil_unit(self) -> int:
        return len(self.graphemes)

    @property
    def num_states(self) -> int:
        return len(self.units) * self.states_per_unit

    def grapheme_ids(self, word: str) -> list[int]:
        index = {grapheme: position for position, grapheme in enumerate(self.graphemes)}
        try:
            return [index[character] for character in word]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is not an acoustic unit of this model") from error


def build_inventory(references: Sequence[str], states_per_unit: int) -> UnitInventory:
    graphemes = sorted({character for text in references for character in text if not character.isspace()})
    if SIL in graphemes or not graphemes:
        raise ValueError("train references must provide at least one grapheme")
    return UnitInventory(graphemes=graphemes, states_per_unit=states_per_unit)


@dataclass
class Chain:
    """Linear state sequence of one training utterance with optional silence tokens."""

    states: list[int]
    skip_into: list[bool]
    can_start: list[bool]
    can_end: list[bool]
    mandatory: list[int]


def build_chain(text: str, inventory: UnitInventory) -> Chain:
    words = text.split()
    if not words:
        raise ValueError("empty training transcript")
    S, sil = inventory.states_per_unit, inventory.sil_unit
    tokens: list[tuple[int, bool]] = [(sil, True)]
    for word in words:
        tokens.extend((unit, False) for unit in inventory.grapheme_ids(word))
        tokens.append((sil, True))
    states, skip_into, mandatory = [], [], []
    for position, (unit, optional) in enumerate(tokens):
        for s in range(S):
            states.append(unit * S + s)
            # The first state of a token may be entered by skipping an optional silence just before it.
            skip_into.append(s == 0 and position >= 2 and tokens[position - 1][1])
            if not optional:
                mandatory.append(unit * S + s)
    count = len(states)
    can_start = [False] * count
    can_end = [False] * count
    can_start[0] = True
    can_start[S] = True  # skip the leading optional silence
    can_end[count - 1] = True
    can_end[count - 1 - S] = True  # skip the trailing optional silence
    return Chain(states, skip_into, can_start, can_end, mandatory)


# --------------------------------------------------------------------------- #
# Diagonal Gaussian mixtures.
# --------------------------------------------------------------------------- #


class GaussianMixtures:
    def __init__(self, num_states: int, max_components: int, dim: int, *, device):
        import torch

        self.num_states, self.max_components, self.dim = num_states, max_components, dim
        self.means = torch.zeros(num_states, max_components, dim, dtype=torch.float32, device=device)
        self.variances = torch.ones(num_states, max_components, dim, dtype=torch.float32, device=device)
        self.log_weights = torch.full((num_states, max_components), -math.inf, device=device)
        self.log_weights[:, 0] = 0.0
        self.active_components = 1

    def flat_start(self, mean, variance) -> None:
        self.means[:] = mean
        self.variances[:] = variance

    def state_loglik(self, features, chunk_frames: int = 20000):
        """Per-frame log-likelihood of every state: (frames, num_states)."""
        import torch

        M = self.active_components
        inv = 1.0 / self.variances[:, :M]
        a = (-0.5 * inv).reshape(-1, self.dim).T
        b = (self.means[:, :M] * inv).reshape(-1, self.dim).T
        c = (-0.5 * ((self.means[:, :M] ** 2) * inv + torch.log(2 * math.pi * self.variances[:, :M])).sum(-1)
             + self.log_weights[:, :M]).reshape(-1)
        outputs = []
        for start in range(0, features.shape[0], chunk_frames):
            x = features[start:start + chunk_frames]
            component = (x * x) @ a + x @ b + c
            outputs.append(torch.logsumexp(component.view(x.shape[0], self.num_states, M), dim=-1))
        return torch.cat(outputs, dim=0)

    def component_posteriors(self, features, states):
        """Responsibilities of each component of the frame's own aligned state: (frames, M)."""
        import torch

        M = self.active_components
        mu, var = self.means[states, :M], self.variances[states, :M]
        diff = features.unsqueeze(1) - mu
        loglik = -0.5 * ((diff * diff) / var + torch.log(2 * math.pi * var)).sum(-1) + self.log_weights[states, :M]
        return torch.softmax(loglik, dim=-1)

    def split(self, perturbation: float = 0.2) -> bool:
        """Double the components of every state (Kaldi/HTK-style mean perturbation)."""
        import torch

        M = self.active_components
        if 2 * M > self.max_components:
            return False
        offset = perturbation * torch.sqrt(self.variances[:, :M])
        self.means[:, M:2 * M] = self.means[:, :M] + offset
        self.means[:, :M] = self.means[:, :M] - offset
        self.variances[:, M:2 * M] = self.variances[:, :M]
        halved = self.log_weights[:, :M] - math.log(2.0)
        self.log_weights[:, :M] = halved
        self.log_weights[:, M:2 * M] = halved
        self.active_components = 2 * M
        return True


class Accumulator:
    def __init__(self, gmm: GaussianMixtures):
        import torch

        device, M = gmm.means.device, gmm.active_components
        self.M = M
        self.occupancy = torch.zeros(gmm.num_states, M, dtype=torch.float64, device=device)
        self.first = torch.zeros(gmm.num_states, M, gmm.dim, dtype=torch.float64, device=device)
        self.second = torch.zeros(gmm.num_states, M, gmm.dim, dtype=torch.float64, device=device)
        self.frames = torch.zeros(gmm.num_states, dtype=torch.float64, device=device)
        self.visits = torch.zeros(gmm.num_states, dtype=torch.float64, device=device)
        self.log_likelihood = 0.0
        self.total_frames = 0

    def add(self, gmm: GaussianMixtures, features, states, chunk_frames: int = 20000) -> None:
        import torch

        M, D = self.M, gmm.dim
        for start in range(0, features.shape[0], chunk_frames):
            x, z = features[start:start + chunk_frames], states[start:start + chunk_frames]
            gamma = gmm.component_posteriors(x, z).double()
            index = (z.unsqueeze(1) * M + torch.arange(M, device=z.device)).reshape(-1)
            xd = x.double()
            self.occupancy.view(-1).index_add_(0, index, gamma.reshape(-1))
            self.first.view(-1, D).index_add_(0, index, (gamma.unsqueeze(-1) * xd.unsqueeze(1)).reshape(-1, D))
            self.second.view(-1, D).index_add_(0, index, (gamma.unsqueeze(-1) * (xd * xd).unsqueeze(1)).reshape(-1, D))
        self.frames.index_add_(0, states, torch.ones_like(states, dtype=torch.float64))
        starts = torch.ones_like(states, dtype=torch.bool)
        starts[1:] = states[1:] != states[:-1]
        self.visits.index_add_(0, states[starts], torch.ones(int(starts.sum()), dtype=torch.float64, device=states.device))
        self.total_frames += int(states.shape[0])


def update_parameters(gmm: GaussianMixtures, acc: Accumulator, *, variance_floor, min_occupancy: float,
                      bounds: tuple[float, float]):
    """M-step. States/components without enough data keep their previous parameters."""
    import torch

    M = acc.M
    occupancy = acc.occupancy
    state_total = occupancy.sum(-1)
    safe = occupancy.clamp_min(1e-10).unsqueeze(-1)
    mean = acc.first / safe
    var = (acc.second / safe - mean * mean).clamp_min(0.0)
    var = torch.maximum(var, variance_floor.double().view(1, 1, -1))
    usable = occupancy >= min_occupancy
    # Never leave a trained state with no usable component: keep its best one.
    best = occupancy.argmax(-1)
    has_data = state_total > 0
    usable[torch.arange(gmm.num_states, device=usable.device)[has_data], best[has_data]] = True
    gmm.means[:, :M] = torch.where(usable.unsqueeze(-1), mean.float(), gmm.means[:, :M])
    gmm.variances[:, :M] = torch.where(usable.unsqueeze(-1), var.float(), gmm.variances[:, :M])
    usable_occupancy = torch.where(usable, occupancy, torch.zeros_like(occupancy))
    weights = usable_occupancy / usable_occupancy.sum(-1, keepdim=True).clamp_min(1e-10)
    new_log_weights = torch.where(usable, torch.log(weights.clamp_min(1e-30)), torch.full_like(weights, -math.inf))
    gmm.log_weights[:, :M] = torch.where(has_data.unsqueeze(-1), new_log_weights.float(), gmm.log_weights[:, :M])
    frames = acc.frames
    self_loop = ((frames - acc.visits) / frames.clamp_min(1.0)).clamp(*bounds)
    return self_loop.float(), has_data


# --------------------------------------------------------------------------- #
# Alignment.
# --------------------------------------------------------------------------- #


def equal_alignment(num_frames: int, chain: Chain) -> list[int]:
    mandatory = chain.mandatory
    if num_frames < len(mandatory):
        raise ValueError("fewer frames than mandatory HMM states")
    return [mandatory[(t * len(mandatory)) // num_frames] for t in range(num_frames)]


def viterbi_align(state_logliks: Sequence, chains: Sequence[Chain], self_lp, next_lp, states_per_unit: int):
    """Batched forced alignment. Returns (per-utterance state paths, per-utterance best scores)."""
    import torch

    device = self_lp.device
    B = len(chains)
    lengths = [int(loglik.shape[0]) for loglik in state_logliks]
    T, J = max(lengths), max(len(chain.states) for chain in chains)
    neg = -math.inf
    chain_states = torch.zeros(B, J, dtype=torch.long, device=device)
    valid = torch.zeros(B, J, dtype=torch.bool, device=device)
    skip_into = torch.zeros(B, J, dtype=torch.bool, device=device)
    can_start = torch.zeros(B, J, dtype=torch.bool, device=device)
    can_end = torch.zeros(B, J, dtype=torch.bool, device=device)
    emissions = torch.full((B, T, J), neg, device=device)
    for b, (chain, loglik) in enumerate(zip(chains, state_logliks)):
        n = len(chain.states)
        states = torch.tensor(chain.states, dtype=torch.long, device=device)
        chain_states[b, :n] = states
        valid[b, :n] = True
        skip_into[b, :n] = torch.tensor(chain.skip_into, device=device)
        can_start[b, :n] = torch.tensor(chain.can_start, device=device)
        can_end[b, :n] = torch.tensor(chain.can_end, device=device)
        emissions[b, :lengths[b], :n] = loglik[:, states]
    stay_lp = torch.where(valid, self_lp[chain_states], torch.full_like(emissions[:, 0], neg))
    move_lp = torch.where(valid, next_lp[chain_states], torch.full_like(emissions[:, 0], neg))
    jump = states_per_unit + 1
    end_frame = torch.tensor(lengths, device=device) - 1
    backpointers = torch.zeros(B, T, J, dtype=torch.int8, device=device)
    alpha = torch.where(can_start, emissions[:, 0], torch.full_like(emissions[:, 0], neg))
    neg_col = torch.full((B, 1), neg, device=device)
    for t in range(1, T):
        stay = alpha + stay_lp
        advance = torch.cat([neg_col, (alpha + move_lp)[:, :-1]], dim=1)
        if J > jump:
            skip = torch.cat([neg_col.expand(B, jump), (alpha + move_lp)[:, :-jump]], dim=1)
            skip = torch.where(skip_into, skip, torch.full_like(skip, neg))
        else:
            skip = torch.full_like(stay, neg)
        best, choice = torch.stack([stay, advance, skip]).max(dim=0)
        updated = best + emissions[:, t]
        active = (t <= end_frame).unsqueeze(1)
        alpha = torch.where(active, updated, alpha)
        backpointers[:, t] = choice.to(torch.int8)
    final = torch.where(can_end, alpha, torch.full_like(alpha, neg))
    scores, position = final.max(dim=1)
    steps = torch.tensor([0, 1, jump], device=device)
    paths = torch.full((B, T), -1, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)
    for t in range(T - 1, -1, -1):
        inside = t <= end_frame
        paths[:, t] = torch.where(inside, chain_states[rows, position], paths[:, t])
        if t > 0:
            move = steps[backpointers[rows, t, position].long()]
            position = torch.where(inside, position - move, position)
    return [paths[b, :lengths[b]] for b in range(B)], scores


# --------------------------------------------------------------------------- #
# Character n-gram language model (Witten-Bell interpolation).
# --------------------------------------------------------------------------- #


@dataclass
class CharLM:
    order: int
    vocab: int  # graphemes + space + boundary; boundary is BOS in histories, EOS as a prediction
    log_probs: Any  # (vocab ** (order-1), vocab) float tensor

    @property
    def space(self) -> int:
        return self.vocab - 2

    @property
    def boundary(self) -> int:
        return self.vocab - 1


def train_char_lm(token_sequences: Sequence[Sequence[int]], *, num_graphemes: int, order: int) -> CharLM:
    """Witten-Bell interpolated character n-gram over graphemes, space and a boundary token.

    Each sequence is wrapped as BOS^(n-1), space, characters, space, EOS. The rows of
    the returned table are proper distributions before the decoder's structural masks.
    """
    import numpy as np
    import torch

    if order < 2 or order > 3:
        raise ValueError("lm_order must be 2 or 3 (dense decoder state space)")
    V = num_graphemes + 2
    space, boundary = V - 2, V - 1
    counts = [np.zeros((V,) * k, dtype=np.float64) for k in range(1, order + 1)]
    for sequence in token_sequences:
        tokens = [boundary] * (order - 1) + [space, *sequence, space, boundary]
        for k in range(1, order + 1):
            grams = np.array([tokens[i - k + 1:i + 1] for i in range(order - 1, len(tokens))])
            np.add.at(counts[k - 1], tuple(grams.T), 1.0)
    unigram = counts[0]
    probabilities = (unigram + 1.0) / (unigram.sum() + V)
    for k in range(2, order + 1):
        c = counts[k - 1]
        history_total = c.sum(-1, keepdims=True)
        types = (c > 0).sum(-1, keepdims=True).astype(np.float64)
        lower = np.broadcast_to(probabilities[np.newaxis, ...], c.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            interpolated = (c + types * lower) / (history_total + types)
        probabilities = np.where(history_total > 0, interpolated, lower)
    table = torch.from_numpy(np.log(probabilities.reshape(-1, V))).float()
    return CharLM(order=order, vocab=V, log_probs=table)


# --------------------------------------------------------------------------- #
# Decoder.
# --------------------------------------------------------------------------- #


class GraphemeDecoder:
    """Viterbi decoding over states (history of n-1 LM tokens incl. current unit, HMM state)."""

    def __init__(self, inventory: UnitInventory, lm: CharLM, self_lp, next_lp, settings: DecoderSettings):
        import torch

        self.inventory, self.lm, self.settings = inventory, lm, settings
        S, V, n = inventory.states_per_unit, lm.vocab, lm.order
        if V != len(inventory.graphemes) + 2:
            raise ValueError("LM vocabulary does not match the acoustic units")
        device = self_lp.device
        self.device, self.S, self.V = device, S, V
        self.H, self.R = V ** (n - 1), V ** (n - 2)
        last_token = torch.arange(self.H, device=device) % V
        unit = torch.where(last_token == lm.space, torch.full_like(last_token, inventory.sil_unit), last_token)
        self.emitting = last_token != lm.boundary
        unit = torch.where(self.emitting, unit, torch.zeros_like(unit))
        self.last_token = last_token
        self.state_table = unit.unsqueeze(1) * S + torch.arange(S, device=device)
        self.self_lp, self.next_lp = self_lp[self.state_table], next_lp[self.state_table]
        log_probs = lm.log_probs.to(device).clone()
        rows_ending_in_space = (torch.arange(self.H, device=device) % V) == lm.space
        log_probs[rows_ending_in_space, lm.space] = -math.inf  # no two consecutive spaces
        # Scale only finite scores: a forbidden (-inf) transition must stay forbidden even
        # at lm_weight 0, where 0 * -inf would otherwise become NaN.
        allowed = torch.isfinite(log_probs)
        scaled = torch.where(allowed, settings.lm_weight * log_probs, torch.full_like(log_probs, -math.inf))
        self.scaled_lm = torch.where(allowed, scaled + settings.insertion_penalty, scaled)
        self.end_lm = scaled[:, lm.boundary]
        self.r_index = torch.arange(self.R, device=device)
        bos_history = 0
        for _ in range(n - 1):
            bos_history = bos_history * V + lm.boundary
        self.bos_history = bos_history

    def _expand(self, exits):
        """From unit-exit scores over histories to next-unit entry scores (with optional frameless space)."""
        import torch

        V, R, space = self.V, self.R, self.lm.space
        neg = -math.inf
        table = self.scaled_lm.view(V, R, V)
        direct = exits.view(V, R).unsqueeze(-1) + table
        best_direct, from_direct = direct.max(dim=0)  # (R, V)
        source_direct = from_direct * R + self.r_index.unsqueeze(1)
        tee = torch.full((self.H,), neg, device=self.device)
        tee_source = torch.zeros(self.H, dtype=torch.long, device=self.device)
        tee_histories = self.r_index * V + space
        tee[tee_histories] = best_direct[:, space] + self.settings.space_skip_logprob
        tee_source[tee_histories] = source_direct[:, space]
        through = tee.view(V, R).unsqueeze(-1) + table
        best_tee, from_tee = through.max(dim=0)
        best_tee[:, space] = neg
        source_tee = from_tee * R + self.r_index.unsqueeze(1)
        use_tee = best_tee > best_direct
        entries = torch.where(use_tee, best_tee, best_direct)
        sources = torch.where(use_tee, source_tee, source_direct)
        entries[:, self.lm.boundary] = neg  # a boundary token is never an acoustic unit
        return entries.reshape(-1), sources.reshape(-1), use_tee.reshape(-1), tee, tee_source

    def decode(self, state_loglik) -> str:
        import torch

        S, H, V = self.S, self.H, self.V
        T = int(state_loglik.shape[0])
        if T < S:
            raise ValueError(f"utterance has {T} frames; at least {S} are required by any path")
        neg = -math.inf
        exits = torch.full((H,), neg, device=self.device)
        exits[self.bos_history] = 0.0
        entries, sources, via_tee, tee, tee_source = self._expand(exits)
        entry_sources = [sources]
        entry_via_tee = [via_tee]
        tee_sources = [tee_source]
        choices = torch.zeros(T, H, S, dtype=torch.bool, device=self.device)
        scores = torch.full((H, S), neg, device=self.device)
        emissions = state_loglik.to(self.device)
        for t in range(T):
            stay = scores + self.self_lp
            previous = torch.empty_like(stay)
            previous[:, 0] = entries
            previous[:, 1:] = scores[:, :-1] + self.next_lp[:, :-1]
            take_previous = previous > stay
            scores = torch.where(take_previous, previous, stay) + emissions[t][self.state_table]
            scores[~self.emitting] = neg
            choices[t] = take_previous
            exits = scores[:, S - 1] + self.next_lp[:, S - 1]
            if t < T - 1:
                entries, sources, via_tee, tee, tee_source = self._expand(exits)
                entry_sources.append(sources)
                entry_via_tee.append(via_tee)
                tee_sources.append(tee_source)
        # Utterance end: the last token is a space (silence unit or frameless skip), then EOS.
        ends_in_space = self.last_token == self.lm.space
        direct_end = torch.where(ends_in_space, exits + self.end_lm, torch.full_like(exits, neg))
        _, _, _, final_tee, final_tee_source = self._expand(exits)
        tee_end = final_tee + self.end_lm
        best_direct, h_direct = direct_end.max(0)
        best_tee, h_tee = tee_end.max(0)
        if not torch.isfinite(torch.maximum(best_direct, best_tee)):
            raise ValueError("no complete decoding path")
        tokens: list[int] = []
        if best_tee > best_direct:
            tokens.append(self.lm.space)
            history = int(final_tee_source[h_tee])
        else:
            history = int(h_direct)
        state, t = S - 1, T - 1
        while t >= 0:
            from_previous = bool(choices[t, history, state])
            if state == 0 and from_previous:
                tokens.append(history % V)
                source = int(entry_sources[t][history])
                if bool(entry_via_tee[t][history]):
                    tokens.append(self.lm.space)
                    source = int(tee_sources[t][source])
                history, state = source, S - 1
            elif from_previous:
                state -= 1
            t -= 1
        if history != self.bos_history:
            raise RuntimeError("decoder traceback did not return to the utterance start")
        characters = [" " if token == self.lm.space else self.inventory.graphemes[token] for token in reversed(tokens)]
        return " ".join("".join(characters).split())


# --------------------------------------------------------------------------- #
# Model container, training, tuning, prediction.
# --------------------------------------------------------------------------- #


@dataclass
class GmmHmmModel:
    inventory: UnitInventory
    features: FeatureConfig
    gmm: GaussianMixtures
    self_loop: Any
    lm: CharLM

    def transition_logprobs(self):
        import torch

        return torch.log(self.self_loop), torch.log1p(-self.self_loop)

    def decoder(self, settings: DecoderSettings) -> GraphemeDecoder:
        self_lp, next_lp = self.transition_logprobs()
        return GraphemeDecoder(self.inventory, self.lm, self_lp, next_lp, settings)

    def save(self, path: Path, report: dict) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": SCHEMA, "graphemes": self.inventory.graphemes,
            "states_per_unit": self.inventory.states_per_unit, "features": asdict(self.features),
            "means": self.gmm.means.cpu(), "variances": self.gmm.variances.cpu(),
            "log_weights": self.gmm.log_weights.cpu(), "active_components": self.gmm.active_components,
            "self_loop": self.self_loop.cpu(), "lm_order": self.lm.order, "lm_log_probs": self.lm.log_probs.cpu(),
            "training_report": report,
        }, path)

    @staticmethod
    def load(path: Path, device) -> "GmmHmmModel":
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} model")
        inventory = UnitInventory(graphemes=list(data["graphemes"]), states_per_unit=int(data["states_per_unit"]))
        features = FeatureConfig(**data["features"])
        means = data["means"]
        gmm = GaussianMixtures(means.shape[0], means.shape[1], means.shape[2], device=device)
        gmm.means, gmm.variances = means.to(device), data["variances"].to(device)
        gmm.log_weights, gmm.active_components = data["log_weights"].to(device), int(data["active_components"])
        lm = CharLM(order=int(data["lm_order"]), vocab=len(inventory.graphemes) + 2,
                    log_probs=data["lm_log_probs"])
        return GmmHmmModel(inventory, features, gmm, data["self_loop"].to(device), lm)


def _train_config(config: dict) -> TrainConfig:
    values = dict(config.get("training", {}))
    for key in ("split_iterations", "self_loop_bounds"):
        if key in values:
            values[key] = tuple(values[key])
    return TrainConfig(**values)


def _extract_features(utterances, frontend: MfccFrontend):
    """Features stay on CPU; batches are moved to the compute device when used."""
    features = []
    for utterance in utterances:
        waveform = common.load_waveform(utterance.audio_path, target_sample_rate=frontend.config.sample_rate)
        features.append(frontend(waveform))
    return features


def _global_moments(features, indices):
    import torch

    total, first, second = 0, None, None
    for i in indices:
        x = features[i].double()
        first = x.sum(0) if first is None else first + x.sum(0)
        second = (x * x).sum(0) if second is None else second + (x * x).sum(0)
        total += x.shape[0]
    mean = first / total
    return mean.float(), (second / total - mean * mean).clamp_min(1e-8).float()


def _alignment_batches(items: Sequence[tuple[int, int, int]], max_cells: int) -> list[list[int]]:
    """Group (index, frames, chain_states) items, sorted by size, under a cell budget."""
    order = sorted(items, key=lambda item: (item[1], item[2]))
    batches, current, frames_max, states_max = [], [], 0, 0
    for index, frames, states in order:
        new_frames, new_states = max(frames_max, frames), max(states_max, states)
        if current and (len(current) + 1) * new_frames * new_states > max_cells:
            batches.append(current)
            current, new_frames, new_states = [], frames, states
        current.append(index)
        frames_max, states_max = new_frames, new_states
    if current:
        batches.append(current)
    return batches


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path, device: str = "cpu",
          progress: bool = True) -> dict[str, Any]:
    import torch

    resolved = common.resolve_device(device)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    feature_config = FeatureConfig(**config.get("features", {}))
    train_config = _train_config(config)
    if train_config.infeasible_policy not in ("error", "exclude_and_report"):
        raise ValueError("training.infeasible_policy must be 'error' or 'exclude_and_report'")
    utterances = common.load_utterances(project_root / config["train_manifest"],
                                        project_root / config["train_references"],
                                        project_root=project_root, split="train")
    subset = None
    if config.get("train_subset_hours") is not None:
        rows = [json.loads(line) for line in (project_root / config["train_manifest"]).read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
        selected, subset = common.select_grouped_subset(
            rows, group_key_priority=("speaker_id", "source_recording_id"), seed=seed,
            target_hours=float(config["train_subset_hours"]))
        keep = {row["id"] for row in selected}
        utterances = [utterance for utterance in utterances if utterance.id in keep]
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inventory = build_inventory([u.reference for u in utterances], train_config.states_per_unit)
    frontend = MfccFrontend(feature_config)
    features = _extract_features(utterances, frontend)
    feature_seconds = time.perf_counter() - started
    chains, excluded = [], []
    for utterance, feats in zip(utterances, features):
        chain = build_chain(utterance.reference, inventory)
        if feats.shape[0] < len(chain.mandatory):
            excluded.append({"id": utterance.id, "reason": "fewer_frames_than_mandatory_states",
                             "frames": int(feats.shape[0]), "mandatory_states": len(chain.mandatory)})
        chains.append(chain)
    if excluded and train_config.infeasible_policy == "error":
        raise ValueError(f"{len(excluded)} training items are too short for their transcripts: {excluded[:3]!r}")
    if len(excluded) / len(utterances) > train_config.max_excluded_fraction:
        raise ValueError(f"{len(excluded)}/{len(utterances)} infeasible items exceed max_excluded_fraction")
    excluded_ids = {row["id"] for row in excluded}
    usable = [i for i, u in enumerate(utterances) if u.id not in excluded_ids]

    global_mean, global_var = _global_moments(features, usable)
    variance_floor = train_config.variance_floor_fraction * global_var.to(resolved)
    gmm = GaussianMixtures(inventory.num_states, train_config.max_gaussians_per_state, feature_config.dim,
                           device=resolved)
    gmm.flat_start(global_mean.to(resolved), global_var.to(resolved))
    self_loop = torch.full((inventory.num_states,), train_config.initial_self_loop, device=resolved)
    history = []

    # Iteration 0: equal alignment over mandatory states.
    acc = Accumulator(gmm)
    for i in usable:
        path = torch.tensor(equal_alignment(int(features[i].shape[0]), chains[i]), device=resolved)
        acc.add(gmm, features[i].to(resolved), path)
    self_loop_new, has_data = update_parameters(gmm, acc, variance_floor=variance_floor,
                                                min_occupancy=train_config.min_gaussian_occupancy,
                                                bounds=train_config.self_loop_bounds)
    self_loop = torch.where(has_data, self_loop_new, self_loop)
    history.append({"iteration": 0, "alignment": "equal", "components": gmm.active_components,
                    "frames": acc.total_frames})
    if 0 in train_config.split_iterations:
        gmm.split()
    batches = _alignment_batches([(i, int(features[i].shape[0]), len(chains[i].states)) for i in usable],
                                 train_config.align_batch_cells)
    for iteration in range(1, train_config.num_iterations + 1):
        iteration_started = time.perf_counter()
        self_lp, next_lp = torch.log(self_loop), torch.log1p(-self_loop)
        acc = Accumulator(gmm)
        total_score, failed = 0.0, []
        for batch in batches:
            batch_features = [features[i].to(resolved) for i in batch]
            logliks = gmm.state_loglik(torch.cat(batch_features)).split([x.shape[0] for x in batch_features])
            paths, scores = viterbi_align(logliks, [chains[i] for i in batch], self_lp, next_lp,
                                          inventory.states_per_unit)
            keep_features, keep_paths = [], []
            for i, x, path, score in zip(batch, batch_features, paths, scores.tolist()):
                if not math.isfinite(score) or bool((path < 0).any()):
                    failed.append(utterances[i].id)
                    continue
                total_score += score
                keep_features.append(x)
                keep_paths.append(path)
            if keep_features:
                acc.add(gmm, torch.cat(keep_features), torch.cat(keep_paths))
        if failed:
            raise RuntimeError(f"forced alignment failed for {len(failed)} items, e.g. {failed[:3]!r}")
        self_loop_new, has_data = update_parameters(gmm, acc, variance_floor=variance_floor,
                                                    min_occupancy=train_config.min_gaussian_occupancy,
                                                    bounds=train_config.self_loop_bounds)
        self_loop = torch.where(has_data, self_loop_new, self_loop)
        record = {"iteration": iteration, "alignment": "viterbi", "components": gmm.active_components,
                  "frames": acc.total_frames, "avg_loglik_per_frame": total_score / max(acc.total_frames, 1),
                  "seconds": time.perf_counter() - iteration_started}
        if iteration in train_config.split_iterations:
            record["split_to"] = 2 * gmm.active_components if gmm.split() else gmm.active_components
        history.append(record)
        if progress:
            print(f"gmm-hmm iteration {iteration}/{train_config.num_iterations}: "
                  f"{record['avg_loglik_per_frame']:.3f} per frame, {gmm.active_components} components/state",
                  flush=True)

    lm = train_char_lm([_lm_tokens(utterances[i].reference, inventory) for i in usable],
                       num_graphemes=len(inventory.graphemes), order=train_config.lm_order)
    model = GmmHmmModel(inventory, feature_config, gmm, self_loop, lm)
    report = {
        "schema": "greek-asr-gmm-hmm-train-v1", "status": "complete", "config": config,
        "training": asdict(train_config), "features": asdict(feature_config), "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/common.py",
                                               ROOT / "greek_asr/models/ctc_utils.py"),
        "train_manifest_sha256": common.hash_file(project_root / config["train_manifest"]),
        "train_references_sha256": common.hash_file(project_root / config["train_references"]),
        "train_utterances_loaded": len(utterances), "train_utterances_used": len(usable),
        "train_frames_used": int(sum(int(features[i].shape[0]) for i in usable)),
        "train_subset": subset, "excluded": excluded, "units": inventory.units,
        "acoustic_states": inventory.num_states, "gaussians_per_state_max": gmm.active_components,
        "lm": {"order": lm.order, "smoothing": "Witten-Bell interpolation, add-one unigram",
               "trained_on": "train references only"},
        "device_requested": device, "device_used": str(resolved), "feature_seconds": feature_seconds,
        "training_seconds": time.perf_counter() - started, "iterations": history,
        "environment": ctc_utils.environment_record(),
    }
    model.save(output_dir / "model.pt", report)
    (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _lm_tokens(text: str, inventory: UnitInventory) -> list[int]:
    space = len(inventory.graphemes)
    tokens: list[int] = []
    for position, word in enumerate(text.split()):
        if position:
            tokens.append(space)
        tokens.extend(inventory.grapheme_ids(word))
    return tokens


def _decode_records(model: GmmHmmModel, settings: DecoderSettings, audio_paths, device):
    frontend = MfccFrontend(model.features)
    decoder = model.decoder(settings)
    for path in audio_paths:
        waveform = common.load_waveform(path, target_sample_rate=model.features.sample_rate)
        loglik = model.gmm.state_loglik(frontend(waveform).to(device))
        yield decoder.decode(loglik)


def tune(model_path: Path, config: dict[str, Any], *, project_root: Path, output_dir: Path,
         device: str = "cpu") -> dict[str, Any]:
    """Choose LM weight / insertion penalty on a seeded random dev subset (never test)."""
    import random

    from greek_asr import scoring

    resolved = common.resolve_device(device)
    tuning = config.get("tuning", {})
    seed = int(config.get("seed", 42))
    utterances = common.load_utterances(project_root / config["dev_manifest"], project_root / config["dev_references"],
                                        project_root=project_root, split="dev")
    # A bounded, seeded utterance sample: dev only selects two decoder scalars, and whole-speaker
    # groups would make the decode budget depend on one speaker's clip count.
    keep = {u.id for u in random.Random(seed).sample(utterances, min(int(tuning.get("dev_utterances", 200)),
                                                                     len(utterances)))}
    utterances = [u for u in utterances if u.id in keep]
    subset = {"seed": seed, "utterances": len(utterances),
              "hours": sum(u.duration_seconds for u in utterances) / 3600.0,
              "ids_sha256": hashlib.sha256("\n".join(sorted(keep)).encode("utf-8")).hexdigest()}
    references = {u.id: ctc_utils.strip_annotation_tags(u.reference) for u in utterances}
    model = GmmHmmModel.load(model_path, resolved)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = []
    for lm_weight in tuning.get("lm_weights", [10.0]):
        for penalty in tuning.get("insertion_penalties", [0.0]):
            settings = DecoderSettings(lm_weight=float(lm_weight), insertion_penalty=float(penalty),
                                       space_skip_logprob=float(tuning.get("space_skip_logprob", 0.0)))
            hypotheses = dict(zip([u.id for u in utterances],
                                  _decode_records(model, settings, [u.audio_path for u in utterances], resolved)))
            score = scoring.score_corpus(references, hypotheses)
            results.append({"settings": asdict(settings), "wer": score["wer"], "cer": score["cer"]})
            print(f"tune lm_weight={lm_weight} insertion_penalty={penalty}: WER={score['wer']:.4f} "
                  f"CER={score['cer']:.4f}", flush=True)
    best = min(results, key=lambda row: (row["wer"], row["cer"]))
    report = {
        "schema": "greek-asr-gmm-hmm-tuning-v1", "split": "dev", "model": str(model_path),
        "model_sha256": common.hash_file(model_path), "dev_subset": subset,
        "dev_manifest_sha256": common.hash_file(project_root / config["dev_manifest"]),
        "reference_preparation": "annotation tags such as <spoken_noise> removed in memory for tuning only; "
                                 "provisional greek_asr.scoring normalizer; final policy is the common scorer's",
        "results": results, "selected": best["settings"], "seconds": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "tuning.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def predict(model_path: Path, manifest_path: Path, *, project_root: Path, split: str, output_dir: Path,
            settings: DecoderSettings, device: str = "cpu", allow_test: bool = False,
            test_protocol: Path | None = None) -> dict[str, Any]:
    from greek_asr.infer import _test_protocol, load_manifest

    resolved = common.resolve_device(device)
    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    protocol_hash = _test_protocol(test_protocol, manifest_path, project_root) if split == "test" else None
    records = load_manifest(manifest_path, project_root, split)
    model = GmmHmmModel.load(model_path, resolved)
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = [row["id"] for row in records]
    report = {
        "schema": "greek-asr-gmm-hmm-predict-v1", "status": "running", "model": str(model_path),
        "model_sha256": common.hash_file(model_path), "split": split, "decoder": asdict(settings),
        "manifest_sha256": common.hash_file(manifest_path), "test_protocol_sha256": protocol_hash,
        "records": len(records), "completed_records": 0, "references_used_as_model_input": False,
        "training_performed": False, "device_requested": device, "device_used": str(resolved),
        "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/common.py"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    writer = common.PartialPredictionWriter(output_dir, ids)
    started = time.perf_counter()
    try:
        for row, text in zip(records, _decode_records(model, settings, [r["audio"] for r in records], resolved)):
            writer.write(row["id"], text)
            report["completed_records"] += 1
        writer.finish()
        report["status"] = "complete"
    except Exception as error:
        writer.abandon()
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["inference_seconds"] = time.perf_counter() - started
        report["environment"] = ctc_utils.environment_record()
        (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _settings_from_args(args) -> DecoderSettings:
    settings = DecoderSettings()
    if args.decoder_settings is not None:
        data = json.loads(args.decoder_settings.read_text(encoding="utf-8"))
        settings = DecoderSettings(**data.get("selected", data))
    overrides = {key: value for key, value in (("lm_weight", args.lm_weight),
                                                 ("insertion_penalty", args.insertion_penalty)) if value is not None}
    return DecoderSettings(**{**asdict(settings), **overrides})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--project-root", type=Path, default=ROOT)
    tune_parser = sub.add_parser("tune")
    tune_parser.add_argument("--model", type=Path, required=True)
    tune_parser.add_argument("--config", type=Path, required=True)
    tune_parser.add_argument("--output", type=Path, required=True)
    tune_parser.add_argument("--device", default="cpu")
    tune_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--decoder-settings", type=Path, help="tuning.json from the tune command")
    predict_parser.add_argument("--lm-weight", type=float)
    predict_parser.add_argument("--insertion-penalty", type=float)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser.add_argument("--allow-test", action="store_true")
    predict_parser.add_argument("--test-protocol", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train(json.loads(args.config.read_text(encoding="utf-8")), project_root=args.project_root,
                  output_dir=args.output, device=args.device)
        elif args.command == "tune":
            tune(args.model, json.loads(args.config.read_text(encoding="utf-8")), project_root=args.project_root,
                 output_dir=args.output, device=args.device)
        else:
            predict(args.model, args.manifest, project_root=args.project_root, split=args.split,
                    output_dir=args.output, settings=_settings_from_args(args), device=args.device,
                    allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"gmm-hmm {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
