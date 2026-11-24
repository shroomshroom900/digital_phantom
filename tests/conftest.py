import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import cache  # noqa: E402
from app.schemas import SimConfig  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """Fresh in-memory caches and a temporary on-disk adjoint-row store, so test
    results (e.g. how many rows were solved) never depend on earlier runs."""
    cache.ROWS = cache.RowStore(disk_dir=str(tmp_path_factory.mktemp("krows")))
    cache.MESHES.clear(); cache.STIFFNESS.clear(); cache.SURFACES.clear()
    yield


def coarse(**overrides):
    """A small but complete configuration (~15k vertices) for fast tests.
    overrides: {"section.field": value}"""
    d = SimConfig().model_dump(mode="json")
    d["mesh"] = dict(h_xy=2e-3, h_z=4e-3)
    d["geometry"]["skin_thickness"] = 2e-3      # must be >= h_xy
    d["fibre"]["dz"] = 1e-3
    for key, v in overrides.items():
        obj = d
        parts = key.split(".")
        for p in parts[:-1]:
            obj = obj[p]
        obj[parts[-1]] = v
    return SimConfig.model_validate(d)
