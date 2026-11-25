from pathlib import Path

import pytest
from pydantic import ValidationError

from corrforecast.config import RunConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_parse(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.models.benchmark in [s.name for s in cfg.models.specs]


def test_defaults_match_paper() -> None:
    cfg = RunConfig.model_validate(
        {"name": "x", "data": {"intraday": "a", "daily": "b", "monthly": "c"}}
    )
    assert cfg.realized.estimator.params["sampling"] == "15min"
    assert cfg.realized.include_overnight
    assert cfg.features.exp_centers_of_mass == (1.0, 5.0, 21.0, 63.0)
    assert cfg.features.exp_window == 500
    assert len(cfg.features.factor_characteristics) == 15
    assert (cfg.models.train_years, cfg.models.val_years) == (4, 1)
    assert [s.name for s in cfg.models.specs] == ["HAR", "SHAR", "SHAR-F", "SHAR-F-Exp", "LASSO"]


def test_rejects_unknown_keys_and_bad_benchmark() -> None:
    base = {"name": "x", "data": {"intraday": "a", "daily": "b", "monthly": "c"}}
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**base, "typo_section": {}})
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**base, "models": {"benchmark": "NOPE"}})
