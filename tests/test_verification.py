"""Analytical verification experiments."""
import pytest

from app import verification as V
from app.verification.blocks import DEFAULT_LAYERS


@pytest.mark.parametrize("name", list(V.EXPERIMENTS))
def test_registered_experiment_passes_with_defaults(name):
    assert V.run_experiment(name)["passed"]


def _val(res, prefix):
    return next(r["value"] for r in res["summary"] if r["quantity"].startswith(prefix))


def test_blocks_formula():
    res = V.run_experiment("stacked_blocks")
    S = 10e-3 ** 2
    expected = 1e-3 * sum(l["length"] / (l["sigma"] * S) for l in DEFAULT_LAYERS)
    assert _val(res, "Analytical") == pytest.approx(expected, rel=1e-12)       # 5.56 V
    assert _val(res, "Relative difference") < 1e-8
    for row in res["layers"]:
        assert row["rel_diff"] < 1e-7


def test_blocks_scaling_and_snapping():
    layers = [dict(name="a", length=2.2e-3, sigma=0.3), dict(name="b", length=5e-3, sigma=2.0)]
    res = V.run_experiment("stacked_blocks", dict(layers=layers, current=2e-3, h_z=1e-3, h_xy=2e-3))
    assert res["passed"]
    assert res["layers"][0]["length_effective_m"] == pytest.approx(2e-3)       # snapped to h_z
    # current doubles -> DeltaV doubles (linearity)
    res1 = V.run_experiment("stacked_blocks", dict(layers=layers, current=1e-3, h_z=1e-3, h_xy=2e-3))
    assert _val(res, "FEM DeltaV") == pytest.approx(2 * _val(res1, "FEM DeltaV"), rel=1e-9)


def test_blocks_invalid():
    with pytest.raises(ValueError):
        V.run_experiment("stacked_blocks", dict(layers=[dict(name="x", length=1e-3, sigma=-1)]))
    with pytest.raises(ValueError):
        V.run_experiment("stacked_blocks", dict(current=float("nan")))
    with pytest.raises(ValueError):
        V.run_experiment("stacked_blocks", dict(bogus=1))
