import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation, sample_actions
from owars.policies.model import (
    HLGaussLoss,
    _destination_fleet_stats,
    _symexp,
    _symlog,
    justnorm,
    normalize_matrices,
)
from owars.policies.sampling import BETA_SAMPLE_EPS, _deterministic_fraction


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
    assert feats.global_feats is not None
    assert feats.global_feats.shape == (27,)
    assert torch.allclose(feats.global_feats[:2], torch.tensor([0.0, 1.0]))
    # Self slot starts at offset 4: planet_count, production, planet_ships,
    # fleet_count, fleet_ships. Enemy_0 follows at offset 9.
    assert torch.allclose(feats.global_feats[4:6], torch.tensor([1.0 / 64.0, 3.0 / 320.0]))
    assert torch.allclose(feats.global_feats[9:11], torch.tensor([1.0 / 64.0, 2.0 / 320.0]))
    assert torch.allclose(feats.global_feats[24:26], torch.tensor([1.0 / 64.0, 1.0 / 320.0]))
    assert feats.planet_feats.shape == (64, 19)
    assert feats.fleet_feats.shape == (1, 20)
    assert feats.fleet_target_planet_idx is None
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


def test_normalize_matrices_caches_target_modules():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    model = OrbitPolicy(cfg)

    normalize_matrices(model)
    targets = model.__dict__.get("_owars_normalize_targets")
    expected_targets = tuple(
        module
        for module in model.modules()
        if callable(getattr(module, "normalize_weights", None))
    )

    assert targets
    assert targets == expected_targets
    assert all(callable(getattr(module, "normalize_weights", None)) for module in targets)

    normalize_matrices(model)

    assert model.__dict__["_owars_normalize_targets"] is targets


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


def test_symlog_hl_gauss_decodes_expected_raw_scalar():
    encoder = HLGaussLoss(
        min_value=-100_000.0,
        max_value=100_000.0,
        num_bins=153,
        symlog=True,
    )
    centers = encoder.encoder.centers
    logits = -0.5 * ((centers - _symlog(torch.tensor(1000.0))) / 2.0).square()

    got = encoder.bins_to_scalar(logits.unsqueeze(0))
    expected = (logits.softmax(dim=-1) * _symexp(centers)).sum().unsqueeze(0)
    certainty_equivalent = encoder.encoder(logits.unsqueeze(0))

    assert torch.allclose(got, expected, atol=1e-4)
    assert not torch.allclose(got, certainty_equivalent, rtol=0.1, atol=1.0)


def test_policy_value_encoder_uses_configured_hl_gauss_sigma():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=2,
        n_heads=2,
        value_num_bins=153,
        value_sigma_to_bin_ratio=2.0,
        value_symlog=True,
    )
    model = OrbitPolicy(cfg)
    bin_width = (
        model.value_encoder.encoder.support[1] - model.value_encoder.encoder.support[0]
    )

    assert model.value_encoder.encoder.sigma == pytest.approx(
        float(2.0 * bin_width),
        rel=1e-5,
    )


def test_value_head_starts_from_zero_logits_without_bias():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    out = model(encode_observation(parse_observation(_obs())))

    assert model.value_head[-1].bias is None
    assert torch.allclose(out.value_logits, torch.zeros_like(out.value_logits), atol=1e-6)
    assert torch.allclose(out.value, torch.zeros_like(out.value), atol=1e-4)


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

    assert h.shape[1] == 3 + 64 + cfg.num_fleet_latents
    assert full_mask.shape[1] == h.shape[1]
    assert planet_mask.shape[-1] == 64
    assert fleet_mask.shape[-1] == cfg.num_fleet_latents
    assert rope_cache is not None
    assert planet_slice == slice(3, 67)
    assert p == 64
    assert f == cfg.num_fleet_latents


def test_destination_conditioned_encoder_scopes_fleets_before_trunk():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="destination_conditioned",
    )
    model = OrbitPolicy(cfg)
    feats = encode_observation(
        parse_observation(_obs()), include_fleet_targets=True
    )

    h, full_mask, planet_mask, fleet_mask, rope_cache, planet_slice, p, f = (
        model._embed_tokens(feats)
    )
    out = model(feats)

    assert model.fleet_tokenizer is None
    assert model.destination_fleet_conditioner is not None
    assert h.shape[1] == 3 + 64
    assert full_mask.shape[1] == h.shape[1]
    assert planet_mask.shape[-1] == 64
    assert fleet_mask.shape[-1] == 0
    assert rope_cache is not None
    assert planet_slice == slice(3, 67)
    assert p == 64
    assert f == 0
    assert out.launch_logits.shape == (1, 64)
    assert out.target_logits.shape == (1, 64, 64)


def test_destination_conditioned_encoder_is_identity_without_inbound_fleets():
    from owars.policies.model import justnorm

    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="destination_conditioned",
    )
    model = OrbitPolicy(cfg)
    feats = encode_observation(
        parse_observation(_obs()), include_fleet_targets=True
    )
    feats.fleet_target_planet_idx.fill_(-1)

    h, _full_mask, _planet_mask, _fleet_mask, _rope_cache, planet_slice, _p, _f = (
        model._embed_tokens(feats)
    )
    expected_planets = justnorm(model.planet_embed(feats.planet_feats.unsqueeze(0)))

    assert torch.allclose(h[:, planet_slice], expected_planets, atol=1e-6)


def test_destination_conditioned_encoder_is_zero_init_identity_with_inbound_fleets():
    from owars.policies.model import justnorm

    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="destination_conditioned",
    )
    model = OrbitPolicy(cfg)
    feats = encode_observation(
        parse_observation(_obs()), include_fleet_targets=True
    )
    feats.fleet_target_planet_idx.fill_(-1)
    assert feats.planet_inbound_feats is not None
    feats.planet_inbound_feats[0, 0] = 1.0 / 64.0

    assert model.destination_fleet_conditioner is not None
    assert torch.count_nonzero(model.destination_fleet_conditioner.mod.weight) == 0
    assert torch.count_nonzero(model.destination_fleet_conditioner.mod.bias) == 0

    h, _full_mask, _planet_mask, _fleet_mask, _rope_cache, planet_slice, _p, _f = (
        model._embed_tokens(feats)
    )
    expected_planets = justnorm(model.planet_embed(feats.planet_feats.unsqueeze(0)))

    assert torch.allclose(h[:, planet_slice], expected_planets, atol=1e-6)


def test_destination_conditioned_encoder_requires_sidecar():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="destination_conditioned",
    )
    model = OrbitPolicy(cfg)
    feats = encode_observation(parse_observation(_obs()))

    with pytest.raises(ValueError, match="fleet_target_planet_idx"):
        model(feats)


def test_destination_fleet_cross_attention_only_reads_matching_destination():
    from owars.policies.model import DestinationFleetCrossAttention

    torch.manual_seed(0)
    attn = DestinationFleetCrossAttention(dim=32, n_heads=2).eval()
    planets = torch.randn(1, 3, 32)
    fleets = torch.randn(1, 4, 32)
    planet_mask = torch.ones(1, 3, dtype=torch.bool)
    fleet_mask = torch.tensor([[True, True, False, True]])
    target_idx = torch.tensor([[0, 1, 0, -1]])

    with torch.no_grad():
        base = attn(planets, fleets, planet_mask, fleet_mask, target_idx)
        changed_fleets = fleets.clone()
        changed_fleets[:, 0] += 10.0
        changed = attn(planets, changed_fleets, planet_mask, fleet_mask, target_idx)

    assert not torch.allclose(base[:, 0], changed[:, 0])
    assert torch.allclose(base[:, 1], changed[:, 1], atol=1e-6)
    assert torch.allclose(base[:, 2], changed[:, 2], atol=1e-6)


def test_destination_fleet_cross_attention_matches_dense_reference_with_gqa():
    from owars.policies.model import DestinationFleetCrossAttention

    torch.manual_seed(0)
    attn = DestinationFleetCrossAttention(dim=32, n_heads=4, n_kv_heads=1).eval()
    planets = torch.randn(2, 5, 32)
    fleets = torch.randn(2, 7, 32)
    planet_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    fleet_mask = torch.tensor(
        [
            [True, True, False, True, True, True, False],
            [True, False, True, True, False, True, True],
        ]
    )
    target_idx = torch.tensor(
        [
            [0, 1, 0, 2, 2, -1, 4],
            [2, 2, 0, 2, 1, 4, -1],
        ]
    )

    with torch.no_grad():
        got = attn(planets, fleets, planet_mask, fleet_mask, target_idx)

        b, p, _ = planets.shape
        q = attn.c_q(planets).unflatten(-1, (attn.n_heads, attn.head_dim))
        k = attn.c_k(fleets).unflatten(-1, (attn.n_kv_heads, attn.head_dim))
        v = attn.c_v(fleets).unflatten(-1, (attn.n_kv_heads, attn.head_dim))
        sqk_q = (attn.sqk_q * (1.0 / attn.base_scale)).view(
            1, 1, attn.n_heads, attn.head_dim
        )
        sqk_k = (attn.sqk_k * (1.0 / attn.base_scale)).view(
            1, 1, attn.n_kv_heads, attn.head_dim
        )
        q = (sqk_q * justnorm(q)).transpose(1, 2)
        k = (sqk_k * justnorm(k)).transpose(1, 2)
        v = v.masked_fill(~fleet_mask.unsqueeze(-1).unsqueeze(-1), 0.0).transpose(1, 2)
        k = k.repeat_interleave(attn.n_heads // attn.n_kv_heads, dim=1)
        v = v.repeat_interleave(attn.n_heads // attn.n_kv_heads, dim=1)
        valid_dest = fleet_mask & (target_idx >= 0) & (target_idx < p)
        dest_idx = torch.where(valid_dest, target_idx, torch.full_like(target_idx, -1))
        arange_p = torch.arange(p)
        mask = planet_mask[:, :, None] & (dest_idx[:, None, :] == arange_p[None, :, None])
        ref = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None],
            scale=attn.head_dim**0.5,
        )
        ref = attn.out_proj(ref.transpose(1, 2).flatten(-2))

    assert torch.allclose(got, ref, atol=1e-5)


def test_destination_fleet_stats_pool_counts_and_ship_mass():
    fleet_feats = torch.zeros(1, 4, 20)
    fleet_feats[0, :, 4] = torch.log1p(torch.tensor([10.0, 20.0, 5.0, 7.0])) / 8.0
    fleet_feats[0, :, 8] = torch.tensor([0.2, 0.5, 0.9, 0.1])
    fleet_feats[0, 0, 14] = 1.0
    fleet_feats[0, 1, 16] = 1.0
    fleet_feats[0, 2, 16] = 1.0
    fleet_feats[0, 3, 14] = 1.0
    fleet_feats[0, 0, 13] = 1.0
    fleet_feats[0, 0, 10] = 0.25
    fleet_mask = torch.tensor([[True, True, True, False]])
    target_idx = torch.tensor([[2, 2, 1, 2]])

    stats = _destination_fleet_stats(fleet_feats, fleet_mask, target_idx, 4)

    assert stats.shape == (1, 4, 13)
    assert stats[0, 2, 0] == pytest.approx(2.0 / 64.0)
    assert stats[0, 2, 1] == pytest.approx(1.0 / 64.0)
    assert stats[0, 2, 2] == pytest.approx(1.0 / 64.0)
    assert stats[0, 2, 3] == pytest.approx(torch.log1p(torch.tensor(30.0)).item() / 8.0)
    assert stats[0, 2, 4] == pytest.approx(torch.log1p(torch.tensor(10.0)).item() / 8.0)
    assert stats[0, 2, 5] == pytest.approx(torch.log1p(torch.tensor(20.0)).item() / 8.0)
    assert stats[0, 1, 10] == pytest.approx(0.9)
    assert stats[0, 2, 11] == pytest.approx(0.25)
    assert stats[0, 2, 12] == pytest.approx(1.0 / 64.0)


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


def test_deterministic_fraction_uses_beta_mode():
    alpha = torch.tensor([2.0, 9.0, 1.0, 9.0, 1.0])
    beta = torch.tensor([9.0, 2.0, 9.0, 1.0, 1.0])

    got = _deterministic_fraction(alpha, beta)

    expected = torch.tensor(
        [
            1.0 / 9.0,
            8.0 / 9.0,
            BETA_SAMPLE_EPS,
            1.0 - BETA_SAMPLE_EPS,
            0.5,
        ]
    )
    assert torch.allclose(got, expected, atol=1e-6)


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
