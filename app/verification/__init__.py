"""
Verification experiments: analytical (or independent) references vs the FEM.

To add a new formula check:
  1. write  def my_check(**params) -> dict  in a new module in this package,
     returning at least {"passed": bool, "summary": [...rows...]} (see blocks.py);
  2. decorate it with @register(name, title, formula, params=[...]);
  3. import the module at the bottom of this file.
It then appears in GET /api/verification, in the UI "Verification" tab, and is
exercised by tests/test_verification.py (every registered experiment is run
with its defaults and must pass).
"""
from app.verification.registry import EXPERIMENTS, register, run_experiment  # noqa: F401

from app.verification import blocks, reciprocity  # noqa: E402,F401  (registers experiments)
