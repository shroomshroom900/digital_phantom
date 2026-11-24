"""
run_all.py -- run the whole project in order.

    python3 run_all.py                # stages 1-5 (roughly 5-10 minutes; stage 5 is most of it)
    python3 run_all.py --skip-timing  # stages 1-4 only (about a minute)

Each stage is an ordinary script, so you can also run them one at a time.
Stage 3 needs the cache written by stage 2, so keep the order.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["stage1_geometry.py", "stage2_forward.py", "stage3_single_fibre.py",
          "stage4_muscle.py", "stage5_timing.py"]
if "--skip-timing" in sys.argv:
    STAGES = STAGES[:-1]

t_all = time.time()
for s in STAGES:
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)
    t0 = time.time()
    # run with the project folder as working directory so imports (config, mesh, ...) resolve
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{s} failed (exit code {r.returncode})")
    print(f"[{s} finished in {time.time() - t0:.0f} s]", flush=True)
print(f"\nAll done in {time.time() - t_all:.0f} s.  Figures are in {os.path.join(HERE, 'figures')}")
