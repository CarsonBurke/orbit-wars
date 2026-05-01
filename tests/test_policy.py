import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation, sample_actions
from owars.policies.model import _fraction_beta_params
from owars.policies.sampling import _deterministic_fraction


def _obs():
    return {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 2],
            [2, -1, 50.0, 90.0, 1.0, 10, 1],
        ],
        "fleets": [[0, 0, 30.0, 30.0, 0.5, 0, 20]],
        "angular_velocity": 0.04,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def test_encode_shapes():
    o = parse_observation(_obs())
    feats = encode_observation(o)
    assert feats.planet_feats.shape == (64, 19)
    assert feats.fleet_feats.shape == (384, 20)
    assert int(feats.planet_mask.sum()) == 3
    assert int(feats.fleet_mask.sum()) == 1
    assert bool(feats.planet_owned_mask[0]) and not bool(feats.planet_owned_mask[1])


def test_policy_forward_shapes():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    o = parse_observation(_obs())
    feats = encode_observation(o)
    out = model(feats)
    # batch dim was added implicitly by the encoder fast path.
    assert out.launch_logits.shape == (1, 64)
    assert out.target_logits.shape == (1, 64, 64)
    assert out.fraction_alpha.shape == (1, 64)
    assert out.fraction_beta.shape == (1, 64)
    assert out.value.shape == (1,)
    # Distributional value head: per-bin logits over the configured support.
    assert out.value_logits.shape == (1, cfg.value_num_bins)
    # Recovered scalar value lives inside the bin support.
    assert cfg.value_min <= float(out.value.item()) <= cfg.value_max


def test_planet_rope_frequency_buffer_stays_fp32_after_bfloat16():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg)

    model.bfloat16()

    assert model.planet_rope.inv_freq.dtype == torch.float32


def test_planet_rope_can_be_disabled():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        planet_rope_fraction=0.0,
    )
    model = OrbitPolicy(cfg)
    out = model(encode_observation(parse_observation(_obs())))

    assert model.planet_rope.rotate_dim == 0
    assert out.launch_logits.shape == (1, 64)


def test_fleet_latent_encoder_compresses_fleet_tokens():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="fleet_latent",
        num_fleet_latents=64,
    )
    model = OrbitPolicy(cfg)
    feats = encode_observation(parse_observation(_obs()))

    h, full_mask, planet_mask, fleet_mask, rope_cache, planet_slice, p, f = (
        model._embed_tokens(feats)
    )

    assert h.shape[1] == 2 + 64 + cfg.num_fleet_latents
    assert full_mask.shape[1] == h.shape[1]
    assert planet_mask.shape[-1] == 64
    assert fleet_mask.shape[-1] == cfg.num_fleet_latents
    assert rope_cache is not None
    assert planet_slice == slice(2, 66)
    assert p == 64
    assert f == cfg.num_fleet_latents


def test_launch_prior_is_stable_across_planet_counts():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg).eval()
    with torch.no_grad():
        model.target_query.weight.zero_()
        model.target_key.weight.zero_()

    obs_small = _obs()
    obs_large = _obs()
    obs_large["planets"] = [
        [i, 0 if i == 0 else -1, 10.0 + i, 10.0 + i, 1.0, 20, 1]
        for i in range(12)
    ]

    with torch.no_grad():
        small = model(encode_observation(parse_observation(obs_small))).launch_logits
        large = model(encode_observation(parse_observation(obs_large))).launch_logits

    expected = torch.sigmoid(torch.tensor(-1.5))
    assert torch.allclose(small.sigmoid()[0, 0], expected, atol=1e-5)
    assert torch.allclose(large.sigmoid()[0, 0], expected, atol=1e-5)


def test_fraction_mode_keeps_gradient_at_concentration_floor():
    mode_logit = torch.tensor([0.0], requires_grad=True)
    concentration_logit = torch.tensor([-80.0], requires_grad=True)

    alpha, beta = _fraction_beta_params(mode_logit, concentration_logit)
    loss = -(alpha - beta).mean()
    loss.backward()

    assert mode_logit.grad is not None
    assert abs(float(mode_logit.grad.item())) > 0.1


def test_deterministic_fraction_uses_parameterized_mode_not_beta_mean():
    mode = torch.tensor([0.1, 0.9])
    concentration = torch.ones_like(mode)
    alpha = 1.0 + concentration * mode
    beta = 1.0 + concentration * (1.0 - mode)

    got = _deterministic_fraction(alpha, beta)

    assert torch.allclose(got, mode)
    assert not torch.allclose(got, alpha / (alpha + beta))


def test_policy_accepts_legacy_fleet_feature_width():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2, fleet_features=15)
    model = OrbitPolicy(cfg)
    feats = encode_observation(parse_observation(_obs()))
    out = model(feats)

    assert out.target_logits.shape == (1, 64, 64)


def test_policy_ignores_padded_token_features():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    o = parse_observation(_obs())
    feats = encode_observation(o)
    noisy = encode_observation(o)
    noisy.planet_feats[~noisy.planet_mask] = torch.randn_like(
        noisy.planet_feats[~noisy.planet_mask]
    )
    noisy.fleet_feats[~noisy.fleet_mask] = torch.randn_like(
        noisy.fleet_feats[~noisy.fleet_mask]
    )

    with torch.no_grad():
        clean_out = model(feats)
        noisy_out = model(noisy)

    valid_planets = feats.planet_mask.unsqueeze(0)
    valid_cols = feats.planet_mask.unsqueeze(0)
    assert torch.allclose(clean_out.value, noisy_out.value)
    assert torch.allclose(
        clean_out.fraction_alpha[valid_planets],
        noisy_out.fraction_alpha[valid_planets],
    )
    assert torch.allclose(
        clean_out.launch_logits[valid_planets],
        noisy_out.launch_logits[valid_planets],
    )
    assert torch.allclose(
        clean_out.target_logits[valid_planets][:, valid_cols.squeeze(0)],
        noisy_out.target_logits[valid_planets][:, valid_cols.squeeze(0)],
    )


def test_sample_actions_returns_legal_moves():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    o = parse_observation(_obs())
    feats = encode_observation(o)
    with torch.no_grad():
        out = model(feats)
    moves = sample_actions(out, o, deterministic=True)
    owned_ids = {p.id for p in o.my_planets()}
    for m in moves:
        assert m.from_planet_id in owned_ids
        assert 1 <= m.num_ships < 50  # less than current garrison
