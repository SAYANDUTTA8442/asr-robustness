"""Smoke tests for the initial project scaffold."""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_package_imports():
    import asr_robustness

    assert isinstance(asr_robustness.__version__, str)


def test_config_loads(config):
    assert isinstance(config, dict)


def test_seed(config):
    assert config["seed"] == 42


def test_n_utterances(config):
    assert config["n_utterances"] == 20


def test_snr_db(config):
    assert config["snr_db"] == [20, 10, 5]


def test_model_id(config):
    assert config["model_id"] == "openai/whisper-tiny.en"
