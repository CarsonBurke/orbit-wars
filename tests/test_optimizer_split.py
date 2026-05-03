from owars.policies.config import OrbitPolicyConfig
from owars.policies.model import OrbitPolicy
from owars.training.config import OptimCfg
from owars.training.ppo import _grad_clip_groups
from owars.training.train import _build_optimizer, _split_params


def _group_by_name(model: OrbitPolicy) -> dict[str, str]:
    groups = _split_params(model)
    labels = ("muon_blocks", "adamw_default", "adamw_control", "adamw_head")
    by_id = {
        id(param): label
        for label, params in zip(labels, groups, strict=True)
        for param in params
    }
    return {name: by_id[id(param)] for name, param in model.named_parameters()}


def test_optimizer_split_matches_parameter_golf_boundary():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)

    group = _group_by_name(model)

    for name, label in group.items():
        if label == "muon_blocks":
            assert name.startswith(("layers.", "fleet_tokenizer.layers."))
            assert dict(model.named_parameters())[name].ndim == 2

    for name in (
        "layers.0.attn.c_q.weight",
        "layers.0.attn.c_k.weight",
        "layers.0.attn.c_v.weight",
        "layers.0.attn.out_proj.weight",
        "layers.0.ff.0.weight",
        "layers.0.ff.2.weight",
        "fleet_tokenizer.layers.0.cross_attn.c_q.weight",
        "fleet_tokenizer.layers.0.cross_attn.out_proj.weight",
        "fleet_tokenizer.layers.0.self_block.attn.c_q.weight",
        "fleet_tokenizer.layers.0.self_block.ff.2.weight",
    ):
        assert group[name] == "muon_blocks"

    for name in (
        "target_query.weight",
        "target_key.weight",
        "launch_head.weight",
        "launch_head.bias",
        "fraction_head.weight",
        "fraction_head.bias",
        "fraction_log_std",
        "value_head.0.weight",
        "value_head.2.weight",
    ):
        assert group[name] == "adamw_head"

    assert group["planet_embed.weight"] == "adamw_default"
    assert group["fleet_embed.weight"] == "adamw_default"
    assert group["target_q_gain"] == "adamw_control"
    assert group["layers.0.attn.q_gain"] == "adamw_control"


def test_optimizer_group_lrs_follow_split():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    optim_cfg = OptimCfg(
        muon_lr=0.022,
        lr=3e-4,
        control_lr=0.02,
        head_lr=7e-4,
    )

    opt = _build_optimizer(model, optim_cfg)

    assert len(opt.optimizers) == 2
    muon_opt, adamw_opt = opt.optimizers
    assert [group["lr"] for group in muon_opt.param_groups] == [optim_cfg.muon_lr]
    assert [group["lr"] for group in adamw_opt.param_groups] == [
        optim_cfg.lr,
        optim_cfg.control_lr,
        optim_cfg.head_lr,
    ]


def test_grad_clip_groups_match_real_policy_roles():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    actor, critic, shared = _grad_clip_groups(model)
    by_id = {
        id(param): label
        for label, params in (
            ("actor", actor),
            ("critic", critic),
            ("shared", shared),
        )
        for param in params
    }
    groups = {name: by_id[id(param)] for name, param in model.named_parameters()}

    for name in (
        "target_query.weight",
        "target_key.weight",
        "target_q_gain",
        "launch_head.weight",
        "launch_head.bias",
        "fraction_head.weight",
        "fraction_head.bias",
        "fraction_log_std",
    ):
        assert groups[name] == "actor"

    assert groups["value_head.0.weight"] == "critic"
    assert groups["value_head.2.weight"] == "critic"
    assert groups["planet_embed.weight"] == "shared"
    assert groups["actor_token"] == "shared"
    assert groups["critic_token"] == "shared"
    assert groups["layers.0.attn.c_q.weight"] == "shared"
