"""
app -- integration layer between the emg_twin numerical model and the web UI.

Nothing in here re-implements physics.  It
  * validates per-run configurations (app.schemas),
  * describes every parameter and its provenance (app.params),
  * calls mesh.py / fem.py / fibre.py with those values and caches the
    expensive intermediate results (app.pipeline, app.cache),
  * turns the mesh into display surfaces (app.geometry),
  * runs simulations off the request thread (app.jobs),
  * exposes all of that over HTTP (app.server),
  * hosts analytical verification experiments (app.verification).

Coordinate system (solver = API = 3-D scene)
--------------------------------------------
Right-handed Cartesian (x, y, z), origin on the arm axis at the middle of the
cylinder.  z = long axis of the arm (= fibre direction); +x is the side where the
default electrode array sits (angle theta = 0); theta = atan2(y, x).
All API values are SI (m, s, S/m, A, V).  The frontend converts to mm / ms / uV
in exactly one place (frontend/src/units.ts).
"""
APP_VERSION = "0.1.0"
