import pytest
from pydantic import ValidationError

from framework.config import RunConfig, deep_merge, load_run_config


def test_deep_merge_override_wins():
    base = {"costs": {"commission_pct": 0.001, "slippage_pct": 0.002}, "results_dir": "results"}
    override = {"costs": {"commission_pct": 0.005}}
    merged = deep_merge(base, override)
    assert merged["costs"]["commission_pct"] == 0.005
    assert merged["costs"]["slippage_pct"] == 0.002  # untouched sibling survives
    assert merged["results_dir"] == "results"


def test_load_run_config_merges_global_and_strategy(tmp_path):
    (tmp_path / "global.yaml").write_text("costs:\n  commission_pct: 0.001\n")
    (tmp_path / "strat.yaml").write_text(
        "strategy: sma_crossover\nsymbols: [TEST]\ncosts:\n  slippage_pct: 0.002\n"
    )
    cfg = load_run_config(tmp_path / "strat.yaml", tmp_path / "global.yaml")
    assert cfg.costs.commission_pct == 0.001  # from global
    assert cfg.costs.slippage_pct == 0.002  # from strategy file


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        RunConfig(strategy="x", symbols=["A"], not_a_real_key=1)


def test_empty_symbols_rejected():
    with pytest.raises(ValidationError):
        RunConfig(strategy="x", symbols=[])


def test_sma_params_validated():
    from framework.strategies import get_strategy

    cls = get_strategy("sma_crossover")
    with pytest.raises(ValidationError):
        cls({"fast": 50, "slow": 20})  # fast must be < slow
    with pytest.raises(ValidationError):
        cls({"fast": 10, "slow": 50, "typo_param": 1})  # extra keys forbidden
