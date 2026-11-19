"""Tests for Kemal's MFCC + GMM-HMM grapheme baseline.

Synthetic emission matrices check the forced aligner and decoder exactly;
the three committed train-pilot clips exercise the real train -> save ->
reload -> tune -> predict path. Neither is a training result.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from greek_asr.models import ctc_utils, gmm_hmm
    _TORCH = True
except ImportError:
    _TORCH = False

from tests.test_models_attention import PILOT, _write_manifest_and_references

ROOT = Path(__file__).resolve().parents[1]
SKIP = "requires torch/torchaudio"


def _emissions(segments, inventory, *, good=0.0, bad=-50.0):
    """Frames whose best state follows ``segments`` = [(unit, frames_per_state...)]."""
    S = inventory.states_per_unit
    rows = []
    for unit, durations in segments:
        for state, frames in enumerate(durations):
            for _ in range(frames):
                row = torch.full((inventory.num_states,), bad)
                row[unit * S + state] = good
                rows.append(row)
    return torch.stack(rows)


@unittest.skipUnless(_TORCH, SKIP)
class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.inventory = gmm_hmm.UnitInventory(graphemes=["a", "b"], states_per_unit=3)
        self.self_lp = torch.full((self.inventory.num_states,), math.log(0.5))
        self.next_lp = torch.full((self.inventory.num_states,), math.log(0.5))

    def test_chain_marks_optional_silences_and_mandatory_states(self):
        chain = gmm_hmm.build_chain("ab a", self.inventory)
        sil = self.inventory.sil_unit * 3
        self.assertEqual(chain.states[:3], [sil, sil + 1, sil + 2])
        self.assertEqual(chain.mandatory, [0, 1, 2, 3, 4, 5, 0, 1, 2])
        self.assertTrue(chain.can_start[0] and chain.can_start[3])
        self.assertTrue(chain.can_end[-1] and chain.can_end[-4])
        # The first grapheme is reached by the start rule; only the later word start skips a pause.
        self.assertEqual([j for j, flag in enumerate(chain.skip_into) if flag], [12])
        with self.assertRaises(ValueError):
            gmm_hmm.equal_alignment(len(chain.mandatory) - 1, chain)

    def test_forced_alignment_recovers_path_and_skips_unused_silences(self):
        inv = self.inventory
        first = _emissions([(0, (1, 1, 1)), (1, (1, 1, 2)), (2, (1, 1, 1))], inv)  # "ab" + final silence
        second = _emissions([(0, (2, 1, 1)), (1, (1, 1, 1))], inv)  # "a b" with no pause at all
        chains = [gmm_hmm.build_chain("ab", inv), gmm_hmm.build_chain("a b", inv)]
        paths, scores = gmm_hmm.viterbi_align([first, second], chains, self.self_lp, self.next_lp, 3)
        self.assertEqual(paths[0].tolist(), [0, 1, 2, 3, 4, 5, 5, 6, 7, 8])
        self.assertEqual(paths[1].tolist(), [0, 0, 1, 2, 3, 4, 5])
        self.assertTrue(all(math.isfinite(score) for score in scores.tolist()))
        alone, _ = gmm_hmm.viterbi_align([second], chains[1:], self.self_lp, self.next_lp, 3)
        self.assertEqual(alone[0].tolist(), paths[1].tolist())

    def test_too_short_utterance_has_no_alignment(self):
        emissions = _emissions([(0, (1, 1, 1))], self.inventory)
        _, scores = gmm_hmm.viterbi_align([emissions], [gmm_hmm.build_chain("ab", self.inventory)],
                                          self.self_lp, self.next_lp, 3)
        self.assertFalse(math.isfinite(scores.item()))


@unittest.skipUnless(_TORCH, SKIP)
class LanguageModelAndDecoderTests(unittest.TestCase):
    def setUp(self):
        self.inventory = gmm_hmm.UnitInventory(graphemes=["a", "b"], states_per_unit=3)
        self.self_lp = torch.full((self.inventory.num_states,), math.log(0.5))
        self.next_lp = torch.full((self.inventory.num_states,), math.log(0.5))
        self.ab_a = [0, 1, 2, 0]  # "ab a": space token index = number of graphemes

    def test_witten_bell_rows_are_distributions(self):
        for order in (2, 3):
            lm = gmm_hmm.train_char_lm([self.ab_a, [1, 1]], num_graphemes=2, order=order)
            sums = lm.log_probs.exp().sum(-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        with self.assertRaises(ValueError):
            gmm_hmm.train_char_lm([self.ab_a], num_graphemes=2, order=4)

    def test_decoder_follows_the_acoustics_with_silence_word_boundary(self):
        inv = self.inventory
        frames = _emissions([(2, (1, 1, 1)), (0, (1, 1, 1)), (1, (1, 2, 1)), (2, (1, 1, 1)),
                             (0, (1, 1, 1)), (2, (1, 1, 1))], inv)
        for order in (2, 3):
            lm = gmm_hmm.train_char_lm([self.ab_a], num_graphemes=2, order=order)
            decoder = gmm_hmm.GraphemeDecoder(inv, lm, self.self_lp, self.next_lp,
                                              gmm_hmm.DecoderSettings(lm_weight=0.0))
            self.assertEqual(decoder.decode(frames), "ab a")

    def test_language_model_inserts_frameless_word_boundaries(self):
        inv = self.inventory
        frames = _emissions([(0, (1, 1, 1)), (1, (1, 1, 1)), (0, (1, 1, 1))], inv)  # no silence frames
        lm = gmm_hmm.train_char_lm([self.ab_a] * 20, num_graphemes=2, order=3)
        with_lm = gmm_hmm.GraphemeDecoder(inv, lm, self.self_lp, self.next_lp,
                                          gmm_hmm.DecoderSettings(lm_weight=5.0))
        self.assertEqual(with_lm.decode(frames), "ab a")

    def test_too_few_frames_for_any_path_is_an_error(self):
        lm = gmm_hmm.train_char_lm([self.ab_a], num_graphemes=2, order=2)
        decoder = gmm_hmm.GraphemeDecoder(self.inventory, lm, self.self_lp, self.next_lp, gmm_hmm.DecoderSettings())
        with self.assertRaises(ValueError):
            decoder.decode(torch.zeros(2, self.inventory.num_states))


@unittest.skipUnless(_TORCH, SKIP)
class MixtureTests(unittest.TestCase):
    def test_split_doubles_components_and_keeps_weights_normalized(self):
        gmm = gmm_hmm.GaussianMixtures(3, 4, 2, device="cpu")
        gmm.flat_start(torch.zeros(2), torch.ones(2))
        self.assertTrue(gmm.split())
        self.assertEqual(gmm.active_components, 2)
        sums = gmm.log_weights[:, :2].exp().sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones(3)))
        self.assertFalse(torch.equal(gmm.means[:, 0], gmm.means[:, 1]))
        self.assertTrue(gmm.split())
        self.assertFalse(gmm.split())  # max_components reached

    def test_em_update_moves_means_towards_aligned_frames(self):
        gmm = gmm_hmm.GaussianMixtures(2, 1, 2, device="cpu")
        gmm.flat_start(torch.zeros(2), torch.ones(2))
        features = torch.tensor([[1.0, 1.0], [1.2, 0.8], [-2.0, -2.0], [-2.2, -1.8]])
        states = torch.tensor([0, 0, 1, 1])
        acc = gmm_hmm.Accumulator(gmm)
        acc.add(gmm, features, states)
        self_loop, has_data = gmm_hmm.update_parameters(gmm, acc, variance_floor=torch.full((2,), 1e-3),
                                                        min_occupancy=1.0, bounds=(0.05, 0.95))
        self.assertTrue(torch.allclose(gmm.means[0, 0], torch.tensor([1.1, 0.9])))
        self.assertTrue(torch.allclose(gmm.means[1, 0], torch.tensor([-2.1, -1.9])))
        self.assertTrue(bool(has_data.all()))
        self.assertTrue(torch.allclose(self_loop, torch.tensor([0.5, 0.5])))


@unittest.skipUnless(_TORCH, SKIP)
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        clean = [dict(row, reference=ctc_utils.strip_annotation_tags(row["reference"])) for row in PILOT]
        self.train_manifest, self.train_refs = _write_manifest_and_references(self.dir, clean, "train")
        self.dev_manifest, self.dev_refs = _write_manifest_and_references(self.dir, PILOT, "dev")
        self.config = {
            "train_manifest": str(self.train_manifest), "train_references": str(self.train_refs),
            "dev_manifest": str(self.dev_manifest), "dev_references": str(self.dev_refs), "seed": 3,
            "training": {"num_iterations": 3, "split_iterations": [1], "max_gaussians_per_state": 2,
                         "min_gaussian_occupancy": 1.0, "lm_order": 3},
            "tuning": {"dev_utterances": 3, "lm_weights": [1.0, 3.0], "insertion_penalties": [0.0]},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_reload_tune_and_predict_every_id(self):
        report = gmm_hmm.train(self.config, project_root=ROOT, output_dir=self.dir / "model", progress=False)
        self.assertEqual((report["status"], report["train_utterances_used"]), ("complete", 3))
        likelihoods = [row["avg_loglik_per_frame"] for row in report["iterations"][1:]]
        self.assertTrue(all(math.isfinite(value) for value in likelihoods))
        self.assertGreaterEqual(likelihoods[-1], likelihoods[0])
        model = self.dir / "model/model.pt"
        tuning = gmm_hmm.tune(model, self.config, project_root=ROOT, output_dir=self.dir / "tune")
        self.assertEqual(len(tuning["results"]), 2)
        self.assertIn(tuning["selected"], [row["settings"] for row in tuning["results"]])
        settings = gmm_hmm.DecoderSettings(**tuning["selected"])
        result = gmm_hmm.predict(model, self.dev_manifest, project_root=ROOT, split="dev",
                                 output_dir=self.dir / "pred", settings=settings)
        self.assertEqual(result["status"], "complete")
        rows = [json.loads(line) for line in (self.dir / "pred/predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in PILOT])
        self.assertTrue(all(isinstance(row["text"], str) for row in rows))
        self.assertFalse(result["references_used_as_model_input"])

    def test_test_split_requires_allow_flag_and_protocol(self):
        gmm_hmm.train(self.config, project_root=ROOT, output_dir=self.dir / "model", progress=False)
        model = self.dir / "model/model.pt"
        for index, allow in enumerate((False, True)):
            with self.assertRaises(ValueError):
                gmm_hmm.predict(model, self.dev_manifest, project_root=ROOT, split="test",
                                output_dir=self.dir / f"t{index}", settings=gmm_hmm.DecoderSettings(),
                                allow_test=allow)
            self.assertFalse((self.dir / f"t{index}").exists())

    def test_missing_or_extra_reference_ids_are_refused(self):
        lines = self.train_refs.read_text(encoding="utf-8").splitlines()
        self.train_refs.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            gmm_hmm.train(self.config, project_root=ROOT, output_dir=self.dir / "missing", progress=False)
        self.assertFalse((self.dir / "missing").exists())
        self.train_refs.write_text("\n".join(lines) + "\n", encoding="utf-8")
        gmm_hmm.train(self.config, project_root=ROOT, output_dir=self.dir / "model", progress=False)
        extra = json.dumps({"id": "hparl:not-in-manifest", "reference": "α"}, ensure_ascii=False)
        self.dev_refs.write_text(self.dev_refs.read_text(encoding="utf-8") + extra + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            gmm_hmm.tune(self.dir / "model/model.pt", self.config, project_root=ROOT, output_dir=self.dir / "tune")
        self.assertFalse((self.dir / "tune").exists())

    def test_items_too_short_for_their_transcript_follow_policy(self):
        rows = [dict(PILOT[0], reference=" ".join(["αβγδεζηθ"] * 12))] + [
            dict(row, reference=ctc_utils.strip_annotation_tags(row["reference"])) for row in PILOT[1:]]
        bad = self.dir / "bad"
        bad.mkdir()
        manifest, refs = _write_manifest_and_references(bad, rows, "train")
        config = dict(self.config, train_manifest=str(manifest), train_references=str(refs))
        config["training"] = dict(self.config["training"], infeasible_policy="error")
        with self.assertRaises(ValueError):
            gmm_hmm.train(config, project_root=ROOT, output_dir=self.dir / "r1", progress=False)
        config["training"] = dict(self.config["training"], max_excluded_fraction=0.5)
        report = gmm_hmm.train(config, project_root=ROOT, output_dir=self.dir / "r2", progress=False)
        self.assertEqual([row["id"] for row in report["excluded"]], [PILOT[0]["id"]])
        self.assertEqual(report["train_utterances_used"], 2)


if __name__ == "__main__":
    unittest.main()
