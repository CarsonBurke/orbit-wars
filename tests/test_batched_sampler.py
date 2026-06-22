"""Batched sampler + EncodedObs stacking — the hot path of vec_rollout."""

from __future__ import annotations

import pytest
import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation
from owars.policies.features import MAX_PLANETS, stack_encoded
from owars.policies.model import PolicyOutput
from owars.policies.sampling import (
    ActionContext,
    SampleBatchRecord,
    SampleRecord,
    _batch_record_from_materialized_launch,
    _categorical_action_log_probs,
    _record_from_materialized_launch,
    _ships_to_send,
    sample_actions,
    sample_batch_actions,
    sample_batch_actions_context,
    sample_batch_actions_raw,
    sample_batch_with_records,
    sample_batch_with_records_context,
    sample_batch_with_records_raw,
    sample_with_record,
)


@pytest.fixture(autouse=True)
def _cpu_inference_only():
    """On CPU the policy is only ever run forward for inference — flex_attention
    has no CPU backward, so it rejects inputs that require grad. Run every CPU
    forward in this module under no_grad, exactly as LearnedAgent does in the
    submission shell.
    """
    with torch.no_grad():
        yield


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


def test_ships_to_send_rounds_half_away_from_zero_like_rust():
    """The launched ship count must match the Rust simulator's ``ships_to_send``
    (``(remaining * frac).round()`` clamped to ``[1, remaining - 1]``), whose
    ``f64::round`` rounds half away from zero. Python's built-in ``round`` is
    round-half-to-even, so it would send one fewer ship on exact halves and
    skew training (Rust) vs submission (Python)."""
    # Exact halves: round-half-away-from-zero, not banker's rounding.
    assert _ships_to_send(49, 0.5) == 25  # 24.5 -> 25 (round half up), not 24
    assert _ships_to_send(205, 0.5) == 103  # 102.5 -> 103, not 102
    assert _ships_to_send(50, 0.5) == 25  # 25.0 exactly, no tie
    assert _ships_to_send(3, 0.5) == 2  # 1.5 -> 2 (would clamp at 2 anyway)
    # Non-tie values round to nearest as usual.
    assert _ships_to_send(100, 0.244) == 24  # 24.4 -> 24
    assert _ships_to_send(100, 0.246) == 25  # 24.6 -> 25
    # Clamp to [1, remaining - 1].
    assert _ships_to_send(10, 0.0) == 1
    assert _ships_to_send(10, 1.0) == 9
    assert _ships_to_send(10, 2.0) == 9  # frac clamped to 1.0 first
    assert _ships_to_send(10, -1.0) == 1  # frac clamped to 0.0 first
    # Fewer than two ships can never launch.
    assert _ships_to_send(1, 1.0) == 0
    assert _ships_to_send(0, 1.0) == 0


def test_categorical_action_log_probs_ignore_nonfinite_targets():
    noop_logits = torch.tensor([[0.0]])
    target_logits = torch.tensor([[[float("inf"), 1.0, float("-inf")]]])

    log_probs = _categorical_action_log_probs(
        noop_logits,
        target_logits,
        action_logit_softcap=8.0,
    )

    assert not torch.isnan(log_probs).any()
    assert log_probs[0, 0, 1].isneginf()
    assert log_probs[0, 0, 3].isneginf()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1, 1))


def _blocked_los_obs(*, fallback: bool = True):
    planets = [
        [0, 0, 10.0, 10.0, 1.0, 50, 3],
        [1, 1, 90.0, 10.0, 1.0, 30, 2],
        [2, -1, 50.0, 5.0, 6.0, 10, 1],
    ]
    if fallback:
        planets.append([3, -1, 10.0, 90.0, 1.0, 10, 1])
    return {
        "player": 0,
        "step": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.04,
        "initial_planets": [row.copy() for row in planets],
        "comet_planet_ids": [] if fallback else [2],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def _model() -> OrbitPolicy:
    return OrbitPolicy(OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2))


def _forced_move_output(feats) -> PolicyOutput:
    """PolicyOutput that deterministically launches from planet 0 to planet 1.

    A high-alpha Beta on planet 0 sends most of the garrison. All other
    planets use symmetric Beta parameters, whose deterministic fraction is 0.5.
    """
    batched = feats.planet_ids.dim() == 2
    b = int(feats.planet_ids.shape[0]) if batched else 1
    p = int(feats.planet_ids.shape[-1])
    launch_logits = torch.full((b, p), -100.0)
    launch_logits[:, 0] = 100.0
    target_logits = torch.full((b, p, p), -100.0)
    target_logits[:, 0, :] = -100.0
    target_logits[:, 0, 1] = 100.0
    fraction_alpha = torch.full((b, p), 2.0)
    fraction_beta = torch.full((b, p), 2.0)
    fraction_alpha[:, 0] = 20.0
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
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=planet_owned_mask,
        planet_mask=planet_mask,
        planet_ids=planet_ids,
        fraction_alpha=fraction_alpha,
        fraction_beta=fraction_beta,
    )


def _forced_source_output(feats, target_scores: dict[int, float]) -> PolicyOutput:
    batched = feats.planet_ids.dim() == 2
    b = int(feats.planet_ids.shape[0]) if batched else 1
    p = int(feats.planet_ids.shape[-1])
    launch_logits = torch.full((b, p), -100.0)
    launch_logits[:, 0] = 100.0
    target_logits = torch.full((b, p, p), -100.0)
    for target, score in target_scores.items():
        target_logits[:, 0, target] = score
    fraction_alpha = torch.full((b, p), 2.0)
    fraction_beta = torch.full((b, p), 2.0)
    fraction_alpha[:, 0] = 20.0
    if not batched:
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
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=planet_owned_mask,
        planet_mask=planet_mask,
        planet_ids=planet_ids,
        fraction_alpha=fraction_alpha,
        fraction_beta=fraction_beta,
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
        value=torch.zeros(1),
        value_logits=torch.zeros(1, 51),
        planet_owned_mask=torch.tensor([[True, True, False]]),
        planet_mask=torch.tensor([[True, True, True]]),
        planet_ids=torch.tensor([[0, 0, 1]]),
        fraction_alpha=torch.full((1, 3), 20.0),
        fraction_beta=torch.full((1, 3), 2.0),
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
    assert stacked.fleet_target_planet_idx is None
    feats_a_targets = encode_observation(obs_a, include_fleet_targets=True)
    feats_b_targets = encode_observation(obs_b, include_fleet_targets=True)
    stacked_targets = stack_encoded([feats_a_targets, feats_b_targets])
    assert stacked_targets.fleet_target_planet_idx is not None
    assert stacked_targets.fleet_target_planet_idx.shape == (2, 0)
    assert stacked_targets.planet_inbound_feats is not None
    assert stacked_targets.planet_inbound_feats.shape == (2, MAX_PLANETS, 13)
    # Element 0 must equal the original.
    assert torch.equal(stacked.planet_feats[0], feats_a.planet_feats)
    assert torch.equal(stacked.fleet_feats[1], feats_b.fleet_feats)
    assert torch.equal(
        stacked_targets.fleet_target_planet_idx[0],
        feats_a_targets.fleet_target_planet_idx,
    )
    assert torch.equal(
        stacked_targets.fleet_target_planet_idx[1],
        feats_b_targets.fleet_target_planet_idx,
    )
    assert torch.equal(
        stacked_targets.planet_inbound_feats[0],
        feats_a_targets.planet_inbound_feats,
    )


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


def test_batched_deterministic_uses_categorical_mode():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    stacked = stack_encoded([feats, feats])
    model = _model()
    out = model(stacked)
    _, records = sample_batch_with_records(out, [o, o], deterministic=True)
    target_legal_mask = torch.stack([r.target_legal_mask for r in records])
    target_logits = out.target_logits.masked_fill(~target_legal_mask, float("-inf"))
    action_log_probs = _categorical_action_log_probs(
        out.launch_logits,
        target_logits,
        out.action_logit_softcap,
        deterministic=True,
    )
    action_idx = action_log_probs.argmax(dim=-1)
    expected_launch = (action_idx > 0).to(records[0].launch.dtype)
    expected_target = (action_idx - 1).clamp_min(0)
    source = out.planet_owned_mask & out.planet_mask
    expected_launch = expected_launch * source.to(dtype=expected_launch.dtype)
    assert torch.equal(records[0].launch, expected_launch[0])
    assert torch.equal(records[1].launch, expected_launch[1])
    assert torch.equal(records[0].target_idx[source[0]], expected_target[0][source[0]])
    assert torch.equal(records[1].target_idx[source[1]], expected_target[1][source[1]])


def test_categorical_deterministic_stays_idle_when_only_move_mass_beats_noop():
    obs = _obs()
    obs["angular_velocity"] = 0.0
    o = parse_observation(obs)
    feats = encode_observation(o)
    p = feats.planet_ids.shape[-1]
    out = PolicyOutput(
        launch_logits=torch.full((1, p), -100.0),
        target_logits=torch.full((1, p, p), -100.0),
        value=torch.zeros(1),
        value_logits=torch.zeros(1, 51),
        planet_owned_mask=feats.planet_owned_mask.unsqueeze(0),
        planet_mask=feats.planet_mask.unsqueeze(0),
        planet_ids=feats.planet_ids.unsqueeze(0),
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((1, p), 20.0),
        fraction_beta=torch.full((1, p), 2.0),
    )
    out.launch_logits[:, 0] = 0.0
    out.target_logits[:, 0, 1] = 0.2
    out.target_logits[:, 0, 2] = 0.2

    actions, records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )

    assert actions[0] == []
    assert records[0].launch[0].item() == 0.0
    assert records[0].target_idx[0].item() == 0


def test_categorical_record_keeps_sampled_launch_when_materialization_fails():
    launch = torch.tensor([1.0, 0.0])
    target_idx = torch.tensor([1, 0])
    frac = torch.tensor([0.4, 0.4])
    launch_logits = torch.zeros(2)
    target_logits = torch.tensor(
        [
            [float("-inf"), 0.0],
            [float("-inf"), float("-inf")],
        ]
    )
    alpha = torch.full((2,), 2.0)
    beta = torch.full((2,), 2.0)

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[False, False],
    )

    action_log_probs = _categorical_action_log_probs(
        launch_logits,
        target_logits,
        8.0,
    )
    frac_lp = (
        (alpha - 1.0) * frac.log()
        + (beta - 1.0) * torch.log1p(-frac)
        - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
    )
    expected = action_log_probs[0, target_idx[0] + 1] + frac_lp[0]

    assert record.raw_launch[0].item() == 1.0
    assert record.launch[0].item() == 1.0
    assert torch.allclose(record.log_prob[0], expected, atol=1e-6)
    assert record.launch[1].item() == 0.0


def test_categorical_noop_record_ignores_invalid_fraction_log_prob():
    launch = torch.tensor([0.0])
    target_idx = torch.tensor([0])
    frac = torch.tensor([0.5])
    launch_logits = torch.zeros(1)
    target_logits = torch.full((1, 2), float("-inf"))
    alpha = torch.tensor([float("nan")])
    beta = torch.tensor([2.0])

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[False],
    )

    assert record.launch.item() == 0.0
    assert torch.isfinite(record.log_prob).all()
    assert torch.allclose(record.log_prob, torch.zeros_like(record.log_prob), atol=1e-6)


def test_categorical_batch_noop_record_ignores_invalid_fraction_log_prob():
    launch = torch.tensor([[0.0]])
    target_idx = torch.tensor([[0]])
    frac = torch.tensor([[0.5]])
    launch_logits = torch.zeros(1, 1)
    target_logits = torch.full((1, 1, 2), float("-inf"))
    alpha = torch.tensor([[float("nan")]])
    beta = torch.tensor([[2.0]])

    record = _batch_record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[[False]],
        rows=[0],
    )

    assert record.launch[0, 0].item() == 0.0
    assert torch.isfinite(record.log_prob).all()
    assert torch.allclose(record.log_prob, torch.zeros_like(record.log_prob), atol=1e-6)


def test_categorical_launched_invalid_fraction_log_prob_stays_nonfinite():
    launch = torch.tensor([1.0])
    target_idx = torch.tensor([0])
    frac = torch.tensor([0.5])
    launch_logits = torch.zeros(1)
    target_logits = torch.tensor([[0.0, float("-inf")]])
    alpha = torch.tensor([float("nan")])
    beta = torch.tensor([2.0])

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[True],
    )

    assert record.launch.item() == 1.0
    assert not torch.isfinite(record.log_prob).all()


def test_categorical_launched_invalid_action_log_prob_stays_nonfinite():
    launch = torch.tensor([1.0])
    target_idx = torch.tensor([0])
    frac = torch.tensor([0.5])
    launch_logits = torch.tensor([float("nan")])
    target_logits = torch.tensor([[0.0, float("-inf")]])
    alpha = torch.tensor([2.0])
    beta = torch.tensor([2.0])

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        None,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[True],
    )

    assert record.launch.item() == 1.0
    assert not torch.isfinite(record.log_prob).all()


def test_categorical_softcap_launched_invalid_action_log_prob_stays_nonfinite():
    launch = torch.tensor([1.0])
    target_idx = torch.tensor([0])
    frac = torch.tensor([0.5])
    launch_logits = torch.tensor([0.0])
    target_logits = torch.tensor([[float("nan"), float("-inf")]])
    alpha = torch.tensor([2.0])
    beta = torch.tensor([2.0])

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[True],
    )

    assert record.launch.item() == 1.0
    assert not torch.isfinite(record.log_prob).all()


def test_categorical_noop_invalid_action_log_prob_uses_rust_fallback_score():
    launch = torch.tensor([0.0])
    target_idx = torch.tensor([0])
    frac = torch.tensor([0.5])
    launch_logits = torch.tensor([float("nan")])
    target_logits = torch.tensor([[0.0, float("-inf")]])
    alpha = torch.tensor([2.0])
    beta = torch.tensor([2.0])

    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        None,
        8.0,
        target_logits,
        alpha,
        beta,
        "beta",
        materialized=[False],
    )

    assert record.launch.item() == 0.0
    assert torch.isfinite(record.log_prob).all()
    assert record.log_prob.item() < -1.0e8


def test_moves_only_sampler_matches_record_path_deterministic():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves = sample_actions(out, o, deterministic=True)
    moves_with_record, _record = sample_with_record(out, o, deterministic=True)

    assert moves
    assert [m.as_list() for m in moves] == [m.as_list() for m in moves_with_record]


def test_deterministic_moves_only_falls_back_to_plausible_legal_launch():
    obs = _obs()
    obs["angular_velocity"] = 0.0
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)
    out.launch_logits[:, 0] = -1.0
    context = ActionContext(
        planets=obs["planets"],
        angular_velocity=obs["angular_velocity"],
        comet_planet_ids=obs.get("comet_planet_ids", ()),
    )

    parsed_moves = sample_actions(out, o, deterministic=True)
    raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)[0]
    context_actions = sample_batch_actions_context(
        out, [context], deterministic=True
    )[0]

    assert parsed_moves
    assert raw_actions
    assert context_actions

    out.launch_logits[:, 0] = -100.0
    assert sample_actions(out, o, deterministic=True) == []
    assert sample_batch_actions_raw(out, [obs], deterministic=True) == [[]]


def test_deterministic_moves_only_fallback_allows_multiple_sources():
    obs = _obs()
    obs["angular_velocity"] = 0.0
    obs["planets"][2] = [2, 0, 90.0, 10.0, 1.0, 40, 2]
    obs["planets"].append([3, -1, 90.0, 90.0, 1.0, 20, 1])
    o = parse_observation(obs)
    feats = encode_observation(o)
    p = feats.planet_ids.shape[-1]
    out = PolicyOutput(
        launch_logits=torch.full((1, p), -100.0),
        target_logits=torch.full((1, p, p), -100.0),
        value=torch.zeros(1),
        value_logits=torch.zeros(1, 51),
        planet_owned_mask=feats.planet_owned_mask.unsqueeze(0),
        planet_mask=feats.planet_mask.unsqueeze(0),
        planet_ids=feats.planet_ids.unsqueeze(0),
        fraction_alpha=torch.full((1, p), 2.0),
        fraction_beta=torch.full((1, p), 2.0),
    )
    out.launch_logits[:, 0] = -1.0
    out.launch_logits[:, 2] = -1.2
    out.target_logits[:, 0, 1] = 10.0
    out.target_logits[:, 2, 3] = 10.0

    actions = sample_batch_actions_raw(out, [obs], deterministic=True)[0]

    assert len(actions) == 2


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
    obs = _obs()
    obs["angular_velocity"] = 0.0
    o = parse_observation(obs)
    actions = sample_batch_actions(_duplicate_source_output(), [o])[0]

    _assert_duplicate_source_actions_do_not_overlaunch([m.as_list() for m in actions])


def test_raw_and_context_samplers_cap_duplicate_source_actions():
    obs = _obs()
    obs["angular_velocity"] = 0.0
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
    fast_raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)

    assert sample_actions(out, o, deterministic=True)
    assert fast_raw_actions
    assert moves
    assert raw_actions[0]
    assert fast_raw_actions[0][0][3] == 2
    assert record.launch[0].item() == 1.0
    assert raw_records[0].launch[0].item() == 1.0
    assert record.target_idx[0].item() == 2
    assert raw_records[0].target_idx[0].item() == 2


def test_sampler_masks_planet_blocked_target_to_legal_alternative():
    obs = _blocked_los_obs(fallback=True)
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_source_output(feats, {1: 100.0, 3: 90.0})
    context = ActionContext(
        planets=obs["planets"],
        angular_velocity=obs["angular_velocity"],
        comet_planet_ids=obs.get("comet_planet_ids", ()),
    )

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )
    fast_raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)
    context_actions, context_records = sample_batch_with_records_context(
        out, [context], deterministic=True
    )

    assert moves
    assert raw_actions[0]
    assert fast_raw_actions[0]
    assert context_actions[0]
    assert fast_raw_actions[0][0][3] == 3
    assert record.target_idx[0].item() == 3
    assert raw_records[0].target_idx[0].item() == 3
    assert context_records[0].target_idx[0].item() == 3
    assert not record.target_legal_mask[0, 1]
    assert record.target_legal_mask[0, 3]


def test_sampler_records_noop_when_only_target_is_planet_blocked():
    obs = _blocked_los_obs(fallback=False)
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_source_output(feats, {1: 100.0})
    out.target_logits[:, 0, 2:] = float("-inf")

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )
    fast_raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)

    assert moves == []
    assert raw_actions == [[]]
    assert fast_raw_actions == [[]]
    assert record.launch[0].item() == 0.0
    assert raw_records[0].launch[0].item() == 0.0
    assert not record.target_legal_mask[0, 1]


def test_sampler_records_noop_when_no_legal_target_exists():
    obs = _sun_crossing_only_obs()
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions, raw_records = sample_batch_with_records_raw(
        out, [obs], deterministic=True
    )
    fast_raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)
    assert moves == []
    assert raw_actions == [[]]
    assert fast_raw_actions == [[]]
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


def test_sampler_allows_comet_target_when_route_is_legal():
    obs = _obs()
    obs["comet_planet_ids"] = [1]
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)
    assert moves
    assert record.launch[0].item() == 1.0
    assert record.target_idx[0].item() == 1
    assert raw_actions[0][0][3] == 1
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


def test_sampler_treats_orbit_threshold_comet_as_comet_not_orbiter():
    obs = _obs()
    obs["step"] = 50
    obs["planets"] = [
        [13, 0, 30.230528337169282, 37.113714576315715, 1.0, 20, 1],
        [29, -1, 47.52883413524248, 97.05854540070332, 1.0, 7, 1],
        [18, 1, 70.0, 75.0, 1.0, 20, 1],
    ]
    obs["initial_planets"] = [row.copy() for row in obs["planets"]]
    obs["comet_planet_ids"] = [29]
    obs["comets"] = [
        {
            "planet_ids": [29],
            "paths": [[[47.52883413524248, 97.05854540070332], [47.0, 95.0]]],
            "path_index": 0,
        }
    ]
    o = parse_observation(obs)
    feats = encode_observation(o)
    out = _forced_move_output(feats)

    moves, record = sample_with_record(out, o, deterministic=True)
    raw_actions = sample_batch_actions_raw(out, [obs], deterministic=True)
    context_actions = sample_batch_actions_context(
        out,
        [
            ActionContext(
                planets=obs["planets"],
                angular_velocity=obs["angular_velocity"],
                comet_planet_ids=obs["comet_planet_ids"],
            )
        ],
        deterministic=True,
    )

    assert moves
    assert record.target_legal_mask[0, 1]
    assert record.launch[0].item() == 1.0
    assert raw_actions[0] and raw_actions[0][0][3] == 29
    assert context_actions[0] and context_actions[0][0][3] == 29
