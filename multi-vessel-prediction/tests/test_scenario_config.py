"""Model mismatch in scenario configs: a vessel's `actual` block and scenario.mismatch."""

import pytest
import yaml

from colregs_planner.scenario_config import load_config, sim_vessel_cfg


def _write(tmp_path, scenario: dict):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"scenario": scenario}))
    return p


def _scenario(**target):
    return {
        "dt": 0.5, "t_horizon": 10.0,
        "vessels": {
            "ego": {"speed": 5.0, "d_pass": 75.0},
            "target_1": {"encounter": "head_on", "range": 300.0, "d_pass": 75.0, **target},
        },
    }


def test_actual_overrides_only_the_simulated_vessel():
    vcfg = {"d_pass": 75.0, "policy": "vo_analytic", "actual": {"d_pass": 40.0}}
    assert sim_vessel_cfg(vcfg) == {"d_pass": 40.0, "policy": "vo_analytic"}
    assert vcfg["d_pass"] == 75.0


def test_actual_rejects_unknown_parameters(tmp_path):
    with pytest.raises(ValueError, match="d_pas"):
        load_config(_write(tmp_path, _scenario(actual={"d_pas": 40.0})))


def test_mismatch_jitter_is_seeded_and_skips_ego(tmp_path):
    def jittered(seed):
        s = _scenario(actual={"tcpa_threshold": 30.0})
        s["mismatch"] = {"seed": seed, "jitter": {"d_pass": 0.4, "tcpa_threshold": 0.4}}
        _, vcfgs, _, _ = load_config(_write(tmp_path, s))
        return vcfgs

    a, b, c = jittered(1), jittered(1), jittered(2)
    assert a["target_1"]["actual"] == b["target_1"]["actual"] != c["target_1"]["actual"]
    assert 45.0 <= a["target_1"]["actual"]["d_pass"] <= 105.0
    assert a["target_1"]["actual"]["tcpa_threshold"] == 30.0   # set explicitly: kept
    assert a["target_1"]["d_pass"] == 75.0                     # what the predictor assumes
    assert "actual" not in a["ego"]
