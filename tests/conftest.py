from pathlib import Path

import numpy as np
import pytest

from dasa.config import load_config
from dasa.geometry import make_scene


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs/default.yaml", root / "configs/smoke.yaml")


@pytest.fixture
def scene(config):
    return make_scene(config, np.random.default_rng(123))
