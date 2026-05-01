"""Batched sampler + EncodedObs stacking — the hot path of vec_rollout."""

from __future__ import annotations

import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation
from owars.policies.features import stack_encoded
from owars.policies.model import PolicyOutput
from owars.policies.sampling import (
    ActionContext,
    SampleBatchRecord,
    SampleRecord,
    sample_actions,
    sample_batch_actions,
    sample_batch_actions_context,
    sample_batch_actions_raw,
    sample_batch_with_records,
    sample_batch_with_records_context,
    sample_batch_with_records_raw,
    sample_with_record,
)


def _obs(player: int = 0):
    return {
        "player": player,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 10.0, 90.0, 1.0, 30, 2],
            [2, -1, 50.0, 90.0, 1.0, 10, 1],
        ],
        "fleets": [[0, 0, 30.0, 30.0, 0.5, 0, 20]],
        "angular_velocity": 0.04,
        "initial_planets": [[0, 0, 10.0, 10.0, 1.0, 50, 3], [1, 1, 10.0, 90.0, 1.0, 30, 2]],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def _sun_crossing_obs():
    obs = _obs()
    obs["planets"][1] = [1, 1, 90.0, 90.0, 1.0, 30, 2]
    obs["initial_planets"][1] = [1, 1, 90.0, 90.0, 1.0, 30, 2]
    return obs


def _sun_crossing_only_obs():
    obs = _sun_crossing_obs()
    obs["planets"] = obs["planets"][:2]
    obs["initial_planets"] = obs["initial_planets"][:2]
    return obs


def _model() -> OrbitPolicy:
    return OrbitPolicy(OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2))


def _forced_move_output(feats) -> PolicyOutput:
    """PolicyOutput that deterministically launches from planet 0 to planet 1.

    Beta(α=20, β=1) on planet 0 has mode at 1 → send most of the garrison.
    All other planets get α=β=1.69 — neutral, mildly spread fractions.
    """
    batched = feats.planet_ids.dim() == 2
    b = int(feats.planet_ids.shape[0]) if batched else 1
    p = int(feats.planet_ids.shape[-1])
    launch_logits = torch.full((b, p), -100.0)
    launch_logits[:, 0] = 100.0
    target_logits = torch.full((b, p, p), -100.0)
    target_logits[:, 0, :] = -100.0
    target_logits[:, 0, 1] = 100.0
    fraction_alpha = torch.full((b, p), 1.69)
    fraction_alpha[:, 0] = 20.0
    fraction_beta = torch.full((b, p), 1.69)
    fraction_beta[:, 0] = 1.0
    if not batched:
        launch_logits = launch_logits[:1]
        target_logits = target_logits[:1]
        fraction_alpha = fraction_alpha[:1]
        fraction_beta = fraction_beta[:1]
        planet_owned_mask = feats.planet_owned_mask.unsqueeze(0)
        planet_mask = feats.planet_mask.unsqueeze(0)
        planet_ids = feats.planet_ids.unsqueeze(0)
    else:
        planet_owned_mask = feats.planet_owned_mask
        planet_mask = feats.planet_mask
        planet_ids = feats.planet_ids
    return PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        fraction_alpha=fraction_alpha,
        fraction_beta=fraction_beta,
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=planet_owned_mask,
        planet_mask=planet_mask,
        planet_ids=planet_ids,
    )


def _duplicate_source_output() -> PolicyOutput:
    launch_logits = torch.full((1, 3), -100.0)
    launch_logits[:, 0] = 100.0
    launch_logits[:, 1] = 100.0
    target_logits = torch.full((1, 3, 3), -100.0)
    target_logits[:, 0, :] = -100.0
    target_logits[:, 1, :] = -100.0
    target_logits[:, 0, 2] = 100.0
    target_logits[:, 1, 2] = 100.0
    return PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        fraction_alpha=torch.full((1, 3), 20.0),
        fraction_beta=torch.full((1, 3), 2.0),
        value=torch.zeros(1),
        value_logits=torch.zeros(1, 51),
        planet_owned_mask=torch.tensor([[True, True, False]]),
        planet_mask=torch.tensor([[True, True, True]]),
        planet_ids=torch.tensor([[0, 0, 1]]),
    )


def _assert_duplicate_source_actions_do_not_overlaunch(actions: list[list]) -> None:
    assert len(actions) == 2
    assert {int(a[0]) for a in actions} == {0}
    assert sum(int(a[2]) for a in actions) == 49
    assert all(1 <= int(a[2]) < 50 for a in actions)


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
        assert rec.launch.shape == (out.target_logits.shape[1],)
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


def test_batched_sampler_can_return_subset_record_batch():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats, feats, feats, feats])
    model = _model()

    torch.manual_seed(123)
    out = model(stacked)
    _, records = sample_batch_with_records(out, [o] * 4, deterministic=False)

    torch.manual_seed(123)
    out_subset = model(stacked)
    moves, batch_record = sample_batch_with_records(
        out_subset, [o] * 4, deterministic=False, record_rows=[1, 3]
    )

    assert len(moves) == 4
    assert isinstance(batch_record, SampleBatchRecord)
    assert torch.equal(batch_record.launch, torch.stack([records[1].launch, records[3].launch]))
    assert torch.equal(batch_record.target_idx, torch.stack([records[1].target_idx, records[3].target_idx]))
    assert torch.allclose(batch_record.fraction, torch.stack([records[1].fraction, records[3].fraction]))
    assert torch.allclose(batch_record.log_prob, torch.stack([records[1].log_prob, records[3].log_prob]), atol=1e-6)
    assert torch.equal(
        batch_record.target_legal_mask,
        torch.stack([records[1].target_legal_mask, records[3].target_legal_mask]),
    )


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
    assert torch.allclose(single_rec.launch, batched_recs[0].launch)
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
    expected_launch = (out.launch_logits > 0.0).to(records[0].launch.dtype)
    assert torch.equal(records[0].launch, expected_launch[0])
    assert torch.equal(records[1].launch, expected_launch[1])
    assert torch.equal(records[0].target_idx, expected[0])
    assert torch.equal(records[1].target_idx, expected[1])


def test_moves_only_sampler_matches_record_path_deterministic():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves = sample_actions(out, o, deterministic=True)
    moves_with_record, _record = sample_with_record(out, o, deterministic=True)

    assert moves
    assert [m.as_list() for m in moves] == [m.as_list() for m in moves_with_record]


def test_moves_only_sampler_matches_record_path_stochastic_under_fixed_seed():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    torch.manual_seed(123)
    moves = sample_actions(out, o, deterministic=False)
    torch.manual_seed(123)
    moves_with_record, _record = sample_with_record(out, o, deterministic=False)

    assert moves
    assert [m.as_list() for m in moves] == [m.as_list() for m in moves_with_record]


def test_batched_moves_only_sampler_matches_record_path_under_fixed_seed():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats, feats])
    out = _forced_move_output(stacked)

    torch.manual_seed(123)
    moves = sample_batch_actions(out, [o, o], deterministic=False)
    torch.manual_seed(123)
    moves_with_records, _records = sample_batch_with_records(
        out, [o, o], deterministic=False
    )

    assert all(row for row in moves)
    assert [[m.as_list() for m in row] for row in moves] == [
        [m.as_list() for m in row] for row in moves_with_records
    ]


def test_sampler_caps_duplicate_source_actions_to_remaining_garrison():
    o = parse_observation(_obs())
    actions = sample_batch_actions(_duplicate_source_output(), [o])[0]

    _assert_duplicate_source_actions_do_not_overlaunch([m.as_list() for m in actions])


def test_raw_and_context_samplers_cap_duplicate_source_actions():
    obs = _obs()
    out = _duplicate_source_output()
    raw_actions = sample_batch_actions_raw(out, [obs])[0]
    context_actions = sample_batch_actions_context(
        out,
        [ActionContext(planets=obs["planets"], angular_velocity=obs["angular_velocity"])],
    )[0]

    _assert_duplicate_source_actions_do_not_overlaunch(raw_actions)
    _assert_duplicate_source_actions_do_not_overlaunch(context_actions)


def test_sampler_masks_sun_crossing_target_to_legal_alternative():
    obs = _sun_crossing_obs()
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )

    assert sample_actions(out, o, deterministic=True)
    assert sample_batch_actions_raw(out, [obs], deterministic=True)
    assert moves
    assert raw_actions[0]
    assert record.launch[0].item() == 1.0
    assert raw_records[0].launch[0].item() == 1.0
    assert record.target_idx[0].item() == 2
    assert raw_records[0].target_idx[0].item() == 2


def test_sampler_records_noop_when_no_legal_target_exists():
    obs = _sun_crossing_only_obs()
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )
    assert moves == []
    assert raw_actions == [[]]
    assert record.launch[0].item() == 0.0
    assert raw_records[0].launch[0].item() == 0.0
    assert not record.target_legal_mask[0].any()


def test_subset_records_match_legality_masked_target_sampling():
    obs = _sun_crossing_obs()
    feats = encode_observation(parse_observation(obs))
    out = _forced_move_output(feats)
    context = ActionContext(
        planets=obs["planets"],
        angular_velocity=obs["angular_velocity"],
        comet_planet_ids=obs.get("comet_planet_ids", ()),
    )

    _, raw_records = sample_batch_with_records_raw(out, [obs], deterministic=True)
    _, raw_batch = sample_batch_with_records_raw(
        out, [obs], deterministic=True, record_rows=[0]
    )
    _, context_batch = sample_batch_with_records_context(
        out, [context], deterministic=True, record_rows=[0]
    )

    expected = raw_records[0]
    assert raw_batch.launch[0, 0].item() == 1.0
    assert context_batch.launch[0, 0].item() == 1.0
    assert raw_batch.target_idx[0, 0].item() == 2
    assert context_batch.target_idx[0, 0].item() == 2
    assert torch.allclose(raw_batch.launch[0], expected.launch)
    assert torch.equal(raw_batch.target_idx[0], expected.target_idx)
    assert torch.allclose(raw_batch.fraction[0], expected.fraction)
    assert torch.allclose(raw_batch.log_prob[0], expected.log_prob, atol=1e-6)
    assert torch.allclose(context_batch.launch[0], expected.launch)
    assert torch.equal(context_batch.target_idx[0], expected.target_idx)
    assert torch.allclose(context_batch.fraction[0], expected.fraction)
    assert torch.allclose(context_batch.log_prob[0], expected.log_prob, atol=1e-6)


def test_stochastic_launch_without_legal_target_is_recorded_as_noop():
    obs = _sun_crossing_only_obs()
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    torch.manual_seed(0)
    moves, record = sample_with_record(out, o, deterministic=False)
    expected_noop_log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(-20.0),
        torch.zeros_like(out.launch_logits[0, 0]),
        reduction="none",
    )

    assert moves == []
    assert record.launch[0].item() == 0.0
    assert torch.allclose(record.log_prob[0], expected_noop_log_prob, atol=1e-6)


def test_no_launch_record_still_masks_missing_legal_target_support():
    obs = _sun_crossing_only_obs()
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)
    out.launch_logits[:, 0] = -100.0

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )

    assert moves == []
    assert raw_actions == [[]]
    assert record.launch[0].item() == 0.0
    assert raw_records[0].launch[0].item() == 0.0
    expected_noop_log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(-20.0),
        torch.zeros_like(out.launch_logits[0, 0]),
        reduction="none",
    )
    assert torch.allclose(record.log_prob[0], expected_noop_log_prob, atol=1e-6)
    assert torch.allclose(raw_records[0].log_prob[0], expected_noop_log_prob, atol=1e-6)
    assert not record.target_legal_mask[0].any()
    assert not raw_records[0].target_legal_mask[0].any()


def test_sampler_masks_comet_target_to_legal_alternative():
    obs = _obs()
    obs["comet_planet_ids"] = [1]
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    assert moves
    assert record.launch[0].item() == 1.0
    assert record.target_idx[0].item() == 2
    assert sample_actions(out, o, deterministic=True)
    assert sample_batch_actions_context(
        out,
        [
            ActionContext(
                planets=obs["planets"],
                angular_velocity=obs["angular_velocity"],
                comet_planet_ids=obs["comet_planet_ids"],
            )
        ],
        deterministic=True,
    )[0]
