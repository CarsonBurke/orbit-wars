import torch

from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation, sample_actions
from owars.policies.model import restore_fp32_params


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
    assert out.target_logits.shape == (1, 64, 65)
    assert out.fraction_alpha.shape == (1, 64)
    assert out.fraction_beta.shape == (1, 64)
    assert out.value.shape == (1,)
    # Distributional value head: per-bin logits over the configured support.
    assert out.value_logits.shape == (1, cfg.value_num_bins)
    # Recovered scalar value lives inside the bin support.
    assert cfg.value_min <= float(out.value.item()) <= cfg.value_max


def test_policy_accepts_legacy_fleet_feature_width():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2, fleet_features=15)
    model = OrbitPolicy(cfg)
    feats = encode_observation(parse_observation(_obs()))
    out = model(feats)

    assert out.target_logits.shape == (1, 64, 65)


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
    valid_cols = torch.cat(
        [
            feats.planet_mask,
            torch.ones(1, dtype=torch.bool),
        ]
    ).unsqueeze(0)
    assert torch.allclose(clean_out.value, noisy_out.value)
    assert torch.allclose(
        clean_out.fraction_alpha[valid_planets],
        noisy_out.fraction_alpha[valid_planets],
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


def test_dense_cuda_encoder_matches_nested_valid_outputs():
    if not torch.cuda.is_available():
        return
    dense_cfg = OrbitPolicyConfig(
        dim=32, ff_dim=64, depth=2, n_heads=2, encoder_backend="dense"
    )
    nested_cfg = OrbitPolicyConfig(
        dim=32, ff_dim=64, depth=2, n_heads=2, encoder_backend="nested"
    )
    dense = OrbitPolicy(dense_cfg).cuda().eval()
    nested = OrbitPolicy(nested_cfg).cuda().eval()
    nested.load_state_dict(dense.state_dict())
    dense.bfloat16()
    nested.bfloat16()
    restore_fp32_params(dense)
    restore_fp32_params(nested)

    obs = parse_observation(_obs())
    feats = encode_observation(obs, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dense_out = dense(feats)
        nested_out = nested(feats)

    valid_rows = dense_out.planet_mask
    valid_targets = torch.cat(
        [
            dense_out.planet_mask,
            torch.ones(
                dense_out.planet_mask.shape[0],
                1,
                dtype=torch.bool,
                device=dense_out.planet_mask.device,
            ),
        ],
        dim=1,
    )
    assert torch.allclose(dense_out.value, nested_out.value, atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        dense_out.fraction_alpha[valid_rows],
        nested_out.fraction_alpha[valid_rows],
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.allclose(
        dense_out.fraction_beta[valid_rows],
        nested_out.fraction_beta[valid_rows],
        atol=2e-2,
        rtol=2e-2,
    )
    dense_logits = dense_out.target_logits[valid_rows][:, valid_targets.squeeze(0)]
    nested_logits = nested_out.target_logits[valid_rows][:, valid_targets.squeeze(0)]
    assert torch.allclose(dense_logits, nested_logits, atol=2e-2, rtol=2e-2)
