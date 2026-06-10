import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation, sample_actions
from owars.policies.model import HLGaussLoss
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
    # Distributional MTP value head: per-horizon, per-bin logits.
    assert out.value_logits.shape == (1, cfg.critic_mtp_horizon, cfg.value_num_bins)
    # Recovered scalar value lives inside the bin support.
    assert cfg.value_min <= float(out.value.item()) <= cfg.value_max


def test_planet_rope_frequency_buffer_stays_fp32_after_bfloat16():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg)

    model.bfloat16()

    assert model.planet_rope.inv_freq.dtype == torch.float32


def test_value_histogram_buffers_stay_fp32_after_bfloat16():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg)

    model.bfloat16()

    buffers = dict(model.value_encoder.encoder.named_buffers())
    assert buffers["support"].dtype == torch.float32
    assert buffers["centers"].dtype == torch.float32


def test_symlog_hl_gauss_encodes_wide_raw_margin_targets():
    encoder = HLGaussLoss(
        min_value=-100_000.0,
        max_value=100_000.0,
        num_bins=153,
        symlog=True,
    )
    assert encoder.encoder.min_value == torch.log1p(torch.tensor(100_000.0)).neg().item()
    assert encoder.encoder.max_value == torch.log1p(torch.tensor(100_000.0)).item()

    targets = torch.tensor([-200_000.0, -1000.0, 0.0, 1000.0, 200_000.0])
    probs = encoder.target_probs(targets)

    assert probs.shape == (5, 153)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    zero_logits = torch.zeros(3, 153)
    values = encoder.bins_to_scalar(zero_logits)
    assert torch.allclose(values, torch.zeros(3), atol=1e-4)


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


def test_grouped_query_attention_forward_shapes():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2, n_kv_heads=1)
    model = OrbitPolicy(cfg)
    out = model(encode_observation(parse_observation(_obs())))

    assert model.layers[0].attn.n_kv_heads == 1
    assert model.layers[0].attn.c_k.weight.shape == (16, 32)
    assert model.fleet_tokenizer is not None
    assert model.fleet_tokenizer.layers[0].cross_attn.n_kv_heads == 1
    assert model.fleet_tokenizer.layers[0].cross_attn.c_k.weight.shape == (16, 32)
    assert out.launch_logits.shape == (1, 64)
    assert out.target_logits.shape == (1, 64, 64)


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


def test_noop_column_is_stable_across_planet_counts():
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

    assert torch.allclose(small[0, 0], torch.zeros(()), atol=1e-5)
    assert torch.allclose(large[0, 0], torch.zeros(()), atol=1e-5)


def test_fraction_beta_concentrations_are_unimodal():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg)
    out = model(encode_observation(parse_observation(_obs())))

    assert "fraction_alpha_head.weight" in dict(model.named_parameters())
    assert "fraction_beta_head.weight" in dict(model.named_parameters())
    assert torch.all(out.fraction_alpha >= 1.0)
    assert torch.all(out.fraction_beta >= 1.0)


def test_deterministic_fraction_uses_beta_mean():
    alpha = torch.tensor([1.0, 9.0])
    beta = torch.tensor([9.0, 1.0])

    got = _deterministic_fraction(alpha, beta)

    assert torch.allclose(got, torch.tensor([0.1, 0.9]), atol=1e-6)


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
        clean_out.fraction_beta[valid_planets],
        noisy_out.fraction_beta[valid_planets],
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


def test_ngpt_control_stats_reports_effective_init_values():
    from owars.policies.model import ngpt_control_stats

    cfg = OrbitPolicyConfig(
        dim=32, ff_dim=64, depth=2, n_heads=2,
        eigen_alpha_init=0.05, qk_gain_init=1.0,
        encoder_backend="fleet_latent", num_fleet_latents=4,
        fleet_tokenizer_depth=1,
    )
    model = OrbitPolicy(cfg)
    stats = ngpt_control_stats(model)

    # Effective units: a fresh model sits exactly at its configured inits.
    assert abs(stats["eigen_alpha_mean"] - 0.05) < 1e-6
    assert abs(stats["eigen_alpha_max"] - 0.05) < 1e-6
    assert abs(stats["sqk_q_eff_mean"] - 1.0) < 1e-6
    assert abs(stats["sqk_k_eff_mean"] - 1.0) < 1e-6
    assert abs(stats["suv_mean"] - 1.0) < 1e-6
    assert abs(stats["target_q_gain"] - 1.0) < 1e-6
