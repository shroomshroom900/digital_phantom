# Simplified myoelectric digital twin of the forearm

A small, heavily commented Python re-implementation of the core of
**Maksymenko et al., "Myoelectric digital twin for fundamental research and
bionic applications", Nature Communications 14:1600 (2023)**.
It simulates surface EMG (sEMG) from muscle fibres in a four-layer cylindrical arm.

Only `numpy`, `scipy` and `matplotlib` are used (see `requirements.txt`).
The whole project runs on a laptop CPU in roughly 5-10 minutes (stages 1-4 take about a minute; stage 5 is the slow one).

```
python3 run_all.py                 # everything, in order
python3 run_all.py --skip-timing   # stages 1-4 only (stage 5 is the slow one)
python3 stage3_single_fibre.py     # or one stage at a time (stage 3 needs stage 2 first)
```

## Interactive digital phantom (web app)

A local web application on top of Stages 1-3: explore the 3-D arm, change supported physical
parameters, place electrodes on the skin, run the FEM single-fibre simulation, view the electrode
signals, compare against a baseline, export, and run analytical verification checks.
Every plotted signal comes from `mesh.py` / `fem.py` / `fibre.py`; the app adds validation,
caching and a UI. No Ansys installation or licence is needed: the solver is the project's own
NumPy/SciPy FEM.

### Setup (once)

Requires Python >= 3.11 and Node.js >= 20 (tested: Python 3.14, Node 24, Windows 11).

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-app.txt
cd frontend && npm install && npm run build && cd ..
```

### Run

```
python -m app                      # http://127.0.0.1:8000   (API + built frontend)
```

For frontend development run both `python -m app` (port 8000) and `cd frontend && npm run dev`
(port 5173, proxies `/api` to 8000). API docs: http://127.0.0.1:8000/docs.

First run on the default mesh solves 16 adjoint problems (~18 s); afterwards results reuse the
cached lead fields and take ~0.1 s for fibre/timing changes, ~1 s per moved electrode, ~20 s
for geometry or conductivity changes.

### Tests

```
python -m pytest -q                # ~1 min: baseline vs stage 3, block benchmark, caching, validation, jobs, API
```

### How it fits together

```
app/
├── schemas.py        immutable per-run SimConfig (defaults read from config.py), SI units
├── params.py         parameter metadata: units, ranges, provenance (config.PROVENANCE), recompute level
├── pipeline.py       SimConfig -> mesh.build_cylinder_mesh -> fem.assemble_stiffness -> fem.adjoint_K
│                     -> fem.BasisSolutions -> fibre.simulate; validation + convergence/residual checks
├── cache.py          LRU caches keyed by sha256(model version + every relevant input);
│                     adjoint rows cached per electrode (memory + cache/app/)
├── geometry.py       display surfaces = the mesh's own staircase tissue boundaries; labelled illustrations
├── jobs.py           one worker thread, increasing sequence numbers, queued jobs superseded by newer ones
├── server.py         FastAPI endpoints (also serves frontend/dist)
└── verification/     registry of analytical checks (stacked blocks, adjoint reciprocity)
frontend/src/         React + TypeScript, React Three Fiber (3-D), Plotly (signals); units.ts converts SI
tests/                pytest suite
```

Changes to the numerical modules were limited to optional parameters with unchanged defaults:
`mesh.build_cylinder_mesh(..., radii=None, sigma=None)`, `fem.skin_electrode_positions(..., r_skin=None)`,
`config.sigma_array(labels, sigma=None)`, plus `config.PROVENANCE`, `DURATION_SINGLE`, `CG_RTOL`.
The test suite checks that the default app run is bit-identical to the stage 2/3 code path.

**Coordinates and units.** One system everywhere: right-handed (x, y, z) in metres, z = arm axis
= fibre direction, origin on the axis at mid-length, θ = atan2(y, x) with θ = 0 on +x (default
electrode side). The API is SI throughout; only `frontend/src/units.ts` converts to mm / ms / µV.

**What a change recomputes** (shown next to every control):

| change | recomputed | reused |
|---|---|---|
| radii, thicknesses, length, mesh spacing | mesh, A, all adjoint rows | – |
| conductivities | A, all adjoint rows | mesh |
| electrode positions / array / CG tolerance | adjoint rows of the electrodes that moved | mesh, A, other rows |
| fibre depth / angle / cell length | basis points (Eq. 5, ms) | mesh, A, all rows |
| NMJ, tendons, velocity, taper, radius, σ_in, duration, sampling | signal (Eq. 10, ms) | everything else |
| camera, visibility, opacity | nothing | – |

**Adding a verification formula.** Write a function returning `{"passed": ..., "summary": [...]}`,
decorate it with `@register(...)` in a module under `app/verification/`, and import it in
`app/verification/__init__.py`. It appears in the API, the Verification tab and the test suite.

**Not yet available (shown greyed out in the UI, with reasons):** fibre tilt (fibres run along z;
the conductivity tensor is diagonal), electrode contact area (point electrodes), electrode-skin
impedance, separate biceps/triceps/brachialis regions (drawn only as labelled illustrations),
Poisson ratio / deformation (no mechanical model), and Stage 4/5 (motor units, recruitment, timing):
implemented as scripts but not wired into the app (`POST /api/simulations/future` is reserved and
returns 501).

## Folder layout

```
emg_twin/
├── README.md              this file
├── requirements.txt
├── run_all.py             runs the five stages in order
│
├── config.py              every number in one place, tagged [PAPER] / [PROMPT] / [ASSUMED]
├── mesh.py                four-layer cylinder -> tetrahedral mesh (structured grid, 6 tets per box)
├── fem.py                 FEM matrix A, conjugate gradient, selection matrix S,
│                          point locator + barycentric coordinates, adjoint method, V_basis
├── fibre.py               Rosenfalck action potential, Tukey windows, current source (Eq. 8),
│                          single-fibre EMG (Eqs. 9-10), many-fibre version
├── muscle.py              motor units, recruitment, firing, MUAPs, final sEMG
│
├── stage1_geometry.py     Stage 1: geometry and mesh                      -> figures/stage1_geometry.png
├── stage2_forward.py      Stage 2: FEM + adjoint + validation (exp. D)    -> figures/stage2_forward.png
├── stage3_single_fibre.py Stage 3: one fibre, 15 channels (exp. A)        -> figures/stage3_*.png
├── stage4_muscle.py       Stage 4: muscle ramp, 8 channels (exp. B)       -> figures/stage4_*.png
├── stage5_timing.py       Stage 5: timing and reuse (exp. C)              -> figures/stage5_timing.png
├── stage5_plot.py         draws the stage 5 figure from cache/cache_stage5.pkl
│
├── figures/               all output pictures
└── cache/                 big intermediate arrays (safe to delete; stage 2 recreates them)
```

Dependency order of the modules: `config` -> `mesh` -> `fem` -> `fibre` -> `muscle`.

## Where each equation of the brief lives

| Brief / paper | Code |
|---|---|
| Poisson equation, Neumann boundary | solved implicitly by `fem.assemble_stiffness` (insulating boundary = the natural boundary condition of the FEM) |
| P1 stiffness matrix `A_ij = ∫ σ ∇w_i·∇w_j` (Eq. 2) | `fem.assemble_stiffness` |
| Null space, `sum(v) = 0` | `fem.cg_solve` (docstring explains why this is safe) |
| Selection matrix `S` (Eq. 3) | `fem.selection_matrix` |
| Barycentric coordinates, `v_point = V_basis λ` (Eqs. 4-5) | `fem.locate_points`, `fem.BasisSolutions.point_potentials` |
| Adjoint method `A Kᵀ = Sᵀ`, `V_basis = K B` (Eqs. 6-7) | `fem.adjoint_K`, `fem.BasisSolutions` |
| Rosenfalck `Vm = 96 z³ e⁻ᶻ − 90` | `fibre.vm_rosenfalck`, `fibre.psi_rosenfalck` |
| Tukey windows `w_L` | `fibre.tukey_window` |
| Current source density (Eq. 8) | `fibre.F_source`, `fibre.csd` |
| Fibre signal, discretised (Eqs. 9-10) | `fibre.simulate`, `fibre.responses_many` |
| Motor-unit distribution | `muscle.make_motor_units`, `muscle.place_fibres` |
| Size-principle recruitment, rate coding, ISI variability | `muscle.spike_trains` |
| MUAP = sum of fibre signals; sEMG = MUAPs at spike times | `muscle.assemble_muaps`, `muscle.assemble_semg` |

## Results at a glance

| Experiment | Result |
|---|---|
| **D** adjoint vs brute force (coarse mesh, 1,536 sources) | NMSE 2.7×10⁻²⁴ (vs CG brute force), 5.2×10⁻²⁵ (vs dense pseudo-inverse). Barycentric blend of Eq. 5 vs a source at an arbitrary point: 7×10⁻²⁵. On the fine mesh the adjoint needs 16 solves instead of 74,176. |
| **A** single fibre, 15 channels | NMJ channel is ~9× smaller than neighbours (not exactly zero: see limitations); the waveform travels at 3.95 m/s (input 4.0) across channels 7-11; waveform changes shape and grows towards the tendon (end-of-fibre effect); amplitude falls ~40× from 1 mm to 11 mm fibre depth. |
| **B** muscle ramp, 8 bipolar channels | RMS follows the excitation (correlation 0.98); electrode over the muscle reads 33× more than the one opposite; recruitment follows the size principle (correlation 0.96); all 200 MUs recruited by 75% MVC. |
| **C** timing | full pipeline ≈ 20-24 s; changing fibre properties ≈ 9-11 s; MU distribution ≈ 4 s; excitation/firing model ≈ 0.01-0.03 s. Short paths reproduce full recomputation exactly (NMSE 0). |

### Timing table (116k-vertex mesh, one laptop-class CPU thread)

| Step | Time |
|---|---|
| mesh + assemble A | 1.1 s |
| general forward solution (8 adjoint solves) | 5.1 s |
| MU distribution | 3.6 s |
| fibres basis points (50k fibres) | 1.1 s |
| fibres EMG response | 9.1 s |
| MUAPs assembling | 0.2 s |
| recruitment + firing | 0.01 s |
| raw sEMG assembling (2 s signal) | 0.01 s |
| **total** | **≈ 20 s** |

Timings are from the final clean run; the same machine gave 20-24 s totals in different sessions (±15 %), so treat them as indicative. When comparing with the paper's Table 1, remember that
its mesh has 2.1 M vertices and its sEMG is 1 minute long (ours: 2 s), so only the *pattern* is comparable.

## Assumptions that are NOT in the paper

Everything below is flagged `[ASSUMED]` in `config.py` or `muscle.py`. Please question these.

**Volume conductor**
- Cylinder length 12 cm.
- Mesh: structured 1 mm × 1 mm × 2 mm boxes, each cut into 6 tetrahedra, so layer boundaries are
  "staircases" (the paper uses a smooth CGAL mesh with 2.1 M vertices).
- Conductivities (S/m), from memory of the cylindrical-model literature (Farina et al. 2004) and
  **not checked against a source** because there was no internet: bone 0.02, fat 0.04, skin 1.0, muscle 0.1 across and 0.5 along the fibres.
- Fibres run exactly along z, so the muscle conductivity tensor is diagonal.

**Electrodes**
- Point electrodes (the paper also allows area-averaged electrodes).
- Experiment A: 16 electrodes in a line on the skin, 5 mm apart.
- Experiment B: 8 bipolar pairs around the arm, 20 mm apart along z (z = 5 and 25 mm).
- Electrodes are moved inwards in 0.1 mm steps until they lie inside the staircase skin.

**Single fibre**
- Radius 25 µm, intracellular conductivity 1.01 S/m, velocity 4 m/s, Tukey taper 5 mm.
- Example geometry: 3 mm below the muscle surface, NMJ at z = −20 mm, tendons at +40 and −45 mm.
- Fibre cells 0.5 mm (single fibre) and 1 mm (muscle); single-fibre plots sampled at 10 kHz.
- **Formula change:** the brief writes the left-travelling wave as ψ(z + z₀ − vt). I used its mirror image about the NMJ, ψ(z₀ − z − vt), with the same sign structure, because the brief's argument does not describe a wave starting at z₀ and moving left. This is my own derivation.
- The Rosenfalck formula is oriented so that the wave front is at z = z₀ + vt, with the profile *behind* the front.

**Muscle and motor units**
- Muscle belly = ellipse, centre (13.5, 0) mm, semi-axes 5.5 × 9 mm (the paper morphs the unit circle into a real brachioradialis).
- In the paper: 50,000 fibres, 200 MUs, sizes 11 to 1150 fibres (exponential), territories 10-50 % of the cross-section, firing 8-35 Hz, all MUs recruited at 75 % MVC.
- Mine: recruitment thresholds spread exponentially (range 30×, Fuglevand-style); rate rises linearly from 8 Hz at threshold to 35 Hz at 100 %; ISI coefficient of variation 20 %; excitation is a raised-cosine ramp.
- "Probability proportional to MU density" taken literally lets big MUs crowd out small ones. I added correction weights so MU sizes come out near the exponential targets (median size error 6 %; the smallest realised MU has 7 fibres).
- Per-fibre properties: NMJ position ~ N(0, 2.5 mm), semi-fibre length 40 ± 3 mm, velocity 4 ± 0.3 m/s.

**Numerics**
- Conjugate gradient with a Jacobi (diagonal) preconditioner, relative tolerance 1e-8.
- A lone unit source cannot exist in an insulated body, so each basis source is balanced by a tiny uniform sink. It cancels exactly for real fibres because their total current is zero (verified: 4×10⁻¹⁶).

## Known limitations and things not done

- **No comparison with the analytical cylinder solution.** The paper validates against Farina's multilayer cylinder model (3-5 % error). That needs a much heavier piece of mathematics and was left out; experiment D (adjoint vs brute force) is fully done.
- **NMJ channel not exactly zero.** With symmetric tendons the NMJ channel still leaves 1.7 % of its neighbours' amplitude. My hypothesis is that the Kuhn tetrahedra are not mirror-symmetric along z; I did not test this.
- **Small wiggles in single-fibre waveforms** (0.5 mm fibre cells on a 2 mm mesh). I think they come from the mesh's linear interpolation but did not test it.
- **Amplitudes** (peak ≈ 2 mV, RMS ≈ 0.5 mV on the best channel) are the right order of magnitude for surface EMG but were not compared with the paper's figures.
- **Staircase boundaries** limit accuracy near the tissue interfaces; its effect was not quantified because there is no analytical reference here.
- Only a single muscle, a straight cylinder and one set of electrodes; no real arm geometry or multi-muscle movements.

## Reproducibility

All randomness uses fixed seeds (`config.RNG_SEED` and the seeds at the top of `stage5_timing.py`), so repeated runs give identical signals.
Conductivities, radii, mesh spacing and all fibre and MU parameters can be changed in `config.py` / the top of `muscle.py`; the stage scripts pick them up automatically.
Delete `cache/` at any time: stage 2 recreates what stage 3 needs.

## A bug found along the way

Stage 5 exposed an infinite loop in `muscle.spike_trains` caused by float round-off when converting a time to a sample index
(a time like 0.35 s could map to sample 699 instead of 700). It is fixed with `int(t * fs + 1e-6)` and Stage 4's results did not change.
