"""Registry of verification experiments (see app/verification/__init__.py)."""
import math
import time

EXPERIMENTS = {}


def register(name, title, formula, params, assumptions=(), description=""):
    """params: list of dicts {key, label, unit, default, min, max, display_unit, display_scale}.
    A parameter whose default is a list of dicts is shown as an editable table."""
    def deco(fn):
        EXPERIMENTS[name] = dict(name=name, title=title, formula=formula, params=params,
                                 assumptions=list(assumptions), description=description, fn=fn)
        return fn
    return deco


def describe():
    return [{k: v for k, v in e.items() if k != "fn"} for e in EXPERIMENTS.values()]


def _check_finite(x, path="params"):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            raise ValueError(f"{path} must be finite")
    elif isinstance(x, dict):
        for k, v in x.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _check_finite(v, f"{path}[{i}]")


def run_experiment(name, params=None):
    if name not in EXPERIMENTS:
        raise KeyError(name)
    e = EXPERIMENTS[name]
    kw = {p["key"]: p["default"] for p in e["params"]}
    unknown = set(params or {}) - set(kw)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    kw.update(params or {})
    _check_finite(kw)
    t0 = time.perf_counter()
    out = e["fn"](**kw)
    out.update(name=name, title=e["title"], formula=e["formula"], params=kw,
               runtime_s=time.perf_counter() - t0)
    return out
