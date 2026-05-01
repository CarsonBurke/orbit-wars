import torch
import pytest

from owars.agents.learned import LearnedAgent
from owars.game import parse_observation
from owars.policies import OrbitPolicy, OrbitPolicyConfig, encode_observation
from owars.policies.features import encode_raw_observations


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
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def _write_ckpt(path, cfg: OrbitPolicyConfig) -> None:
    model = OrbitPolicy(cfg)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)


def test_learned_agent_cpu_ignores_compile_mode(tmp_path):
    ckpt = tmp_path / "agent.pt"
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    _write_ckpt(ckpt, cfg)

    agent = LearnedAgent(ckpt, device="cpu", compile_mode="reduce-overhead")
    feats = encode_observation(parse_observation(_obs()))
    out = agent._forward(feats, 1)

    assert agent.compile_mode is None
    assert out.launch_logits.shape == (1, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile smoke")
def test_learned_agent_cuda_compiled_padding_matches_direct_model(tmp_path):
    ckpt = tmp_path / "agent.pt"
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2)
    _write_ckpt(ckpt, cfg)
    agent = LearnedAgent(
        ckpt,
        device="cuda",
        compile_mode="reduce-overhead",
        compile_graph_rows=4,
    )
    obs_list = [_obs(), _obs(), _obs()]
    feats = encode_raw_observations(obs_list, device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct = agent.model(feats)
    compiled = agent._forward(feats, len(obs_list))

    assert compiled.launch_logits.shape == direct.launch_logits.shape
    assert torch.allclose(compiled.launch_logits, direct.launch_logits, atol=1e-4)
    assert torch.allclose(compiled.value, direct.value, atol=1e-4)
    finite = torch.isfinite(direct.target_logits)
    assert torch.allclose(
        compiled.target_logits[finite],
        direct.target_logits[finite],
        atol=1e-4,
    )
