"""Batched sampler + EncodedObs stacking — the hot path of vec_rollout."""

from __future__ import annotations

import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation
from owars.policies.features import stack_encoded
from owars.policies.sampling import (
    SampleRecord,
    sample_batch_with_records,
    sample_with_record,
)


def _obs(player: int = 0):
    return {
        "player": player,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 2],
            [2, -1, 50.0, 90.0, 1.0, 10, 1],
        ],
        "fleets": [[0, 0, 30.0, 30.0, 0.5, 0, 20]],
        "angular_velocity": 0.04,
        "initial_planets": [[0, 0, 10.0, 10.0, 1.0, 50, 3], [1, 1, 90.0, 90.0, 1.0, 30, 2]],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def _model() -> OrbitPolicy:
    return OrbitPolicy(OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2))


def test_stack_encoded_preserves_fields():
    obs_a = parse_observation(_obs(player=0))
    obs_b = parse_observation(_obs(player=1))
    feats_a = encode_observation(obs_a)
    feats_b = encode_observation(obs_b)
    stacked = stack_encoded([feats_a, feats_b])
    assert stacked.planet_feats.shape[0] == 2
    assert stacked.planet_feats.shape[1:] == feats_a.planet_feats.shape
    assert stacked.fleet_mask.shape == (2, feats_a.fleet_mask.shape[0])
    # Element 0 must equal the original.
    assert torch.equal(stacked.planet_feats[0], feats_a.planet_feats)
    assert torch.equal(stacked.fleet_feats[1], feats_b.fleet_feats)


def test_batched_sampler_shapes_and_record_lengths():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats, feats, feats])
    model = _model()
    out = model(stacked)
    moves_list, records = sample_batch_with_records(
        out, [o, o, o], deterministic=False
    )
    assert len(moves_list) == 3
    assert len(records) == 3
    for rec in records:
        assert isinstance(rec, SampleRecord)
        assert rec.target_idx.shape == (out.target_logits.shape[1],)
        assert rec.fraction.shape == (out.target_logits.shape[1],)
        assert rec.log_prob.shape == (out.target_logits.shape[1],)


def test_batched_log_probs_finite_under_stochastic_sampling():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats] * 4)
    model = _model()
    out = model(stacked)
    _, records = sample_batch_with_records(out, [o] * 4, deterministic=False)
    for rec in records:
        assert torch.isfinite(rec.log_prob).all(), rec.log_prob


def test_batched_matches_single_under_fixed_seed():
    """Equivalence: a B=1 batched call produces the same record as the
    single-element sampler when the RNG is reset between."""
    o = parse_observation(_obs())
    feats = encode_observation(o)
    model = _model()

    torch.manual_seed(123)
    single_out = model(feats)
    _, single_rec = sample_with_record(single_out, o, deterministic=False)

    torch.manual_seed(123)
    batched_out = model(stack_encoded([feats]))
    _, batched_recs = sample_batch_with_records(
        batched_out, [o], deterministic=False
    )
    assert torch.equal(single_rec.target_idx, batched_recs[0].target_idx)
    assert torch.allclose(single_rec.fraction, batched_recs[0].fraction)
    assert torch.allclose(single_rec.log_prob, batched_recs[0].log_prob, atol=1e-6)


def test_batched_deterministic_argmax_matches_logits():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats, feats])
    model = _model()
    out = model(stacked)
    _, records = sample_batch_with_records(out, [o, o], deterministic=True)
    expected = out.target_logits.argmax(dim=-1)
    assert torch.equal(records[0].target_idx, expected[0])
    assert torch.equal(records[1].target_idx, expected[1])
