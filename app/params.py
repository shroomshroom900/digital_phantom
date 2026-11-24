"""
Parameter metadata for the UI: name, units, range, provenance, solver support and
what a change forces the pipeline to recompute.

Defaults and provenance are NOT duplicated here: they are read from config.py
(the default values via app.schemas, the tags via config.PROVENANCE).

Recompute levels (each implies all the ones after it):
    mesh       -> rebuild mesh, reassemble A, re-solve all adjoint rows
    stiffness  -> reassemble A on the same mesh, re-solve all adjoint rows
    adjoint    -> re-solve adjoint rows for electrodes whose position changed
                  (A is reused; unchanged electrodes reuse their cached row)
    basis      -> re-interpolate V_basis at the fibre points (Eq. 5; milliseconds)
    signal     -> re-evaluate Eq. (10) only (milliseconds)
    display    -> no numerical recomputation
"""
import math

import config as C
from app.schemas import SimConfig

RECOMPUTE_LEVELS = ["mesh", "stiffness", "adjoint", "basis", "signal", "display"]

_D = SimConfig()  # defaults


def _p(key, label, unit, display_unit, scale, lo, hi, step, prov_key, recompute, group,
       note="", supported=True, integer=False):
    tag, why = C.PROVENANCE.get(prov_key, ("[ASSUMED]", "")) if prov_key else ("[ASSUMED]", "")
    obj = _D
    for part in key.split("."):
        obj = getattr(obj, part)
    return dict(key=key, label=label, unit=unit, display_unit=display_unit, display_scale=scale,
                min=lo, max=hi, step=step, default=obj, provenance=tag, provenance_note=why,
                note=note, supported=supported, recompute=recompute, group=group, integer=integer)


DEG = 180 / math.pi

PARAMETERS = [
    # ---- geometry (remesh)
    _p("geometry.bone_radius", "Bone radius", "m", "mm", 1e3, 2e-3, 15e-3, 0.5e-3, "R_BONE", "mesh", "Geometry",
       "Idealised humerus: a centred cylinder."),
    _p("geometry.muscle_radius", "Muscle outer radius", "m", "mm", 1e3, 10e-3, 40e-3, 0.5e-3, "R_MUSCLE", "mesh", "Geometry",
       "One homogeneous muscle annulus between bone and fat."),
    _p("geometry.fat_thickness", "Fat thickness", "m", "mm", 1e3, 1e-3, 15e-3, 0.5e-3, "R_FAT", "mesh", "Geometry",
       "Subcutaneous adipose layer.  Must be at least one mesh cell (h_xy) thick."),
    _p("geometry.skin_thickness", "Skin thickness", "m", "mm", 1e3, 1e-3, 4e-3, 0.5e-3, "R_SKIN", "mesh", "Geometry",
       "Must be at least one mesh cell (h_xy) thick, otherwise the staircase skin develops holes."),
    _p("geometry.length", "Arm (cylinder) length", "m", "mm", 1e3, 80e-3, 200e-3, 2e-3, "LENGTH", "mesh", "Geometry",
       "Insulated end caps.  The stored fibre spans min(55 mm, L/2 - 5 mm) either side of z = 0."),
    # ---- numerics (remesh)
    _p("mesh.h_xy", "Mesh spacing x, y", "m", "mm", 1e3, 0.5e-3, 2e-3, 0.25e-3, "H_XY", "mesh", "Numerics",
       "Smaller = more accurate and much slower (vertex count ~ 1/h_xy^2)."),
    _p("mesh.h_z", "Mesh spacing z", "m", "mm", 1e3, 1e-3, 4e-3, 0.5e-3, "H_Z", "mesh", "Numerics"),
    _p("solver.cg_rtol", "CG relative tolerance", "1", "", 1, 1e-12, 1e-6, None, "CG_RTOL", "adjoint", "Numerics",
       "Tolerance of each adjoint conjugate-gradient solve (fem.adjoint_K)."),
    # ---- conductivities (reassemble A)
    _p("conductivity.skin", "Skin conductivity", "S/m", "S/m", 1, 1e-3, 10, 0.01, "SIGMA_SKIN", "stiffness", "Conductivity",
       "Isotropic."),
    _p("conductivity.fat", "Fat conductivity", "S/m", "S/m", 1, 1e-3, 10, 0.005, "SIGMA_FAT", "stiffness", "Conductivity",
       "Isotropic."),
    _p("conductivity.muscle_transverse", "Muscle conductivity (across fibres)", "S/m", "S/m", 1, 1e-3, 10, 0.01,
       "SIGMA_MUSCLE_T", "stiffness", "Conductivity", "sigma_x = sigma_y."),
    _p("conductivity.muscle_longitudinal", "Muscle conductivity (along fibres, z)", "S/m", "S/m", 1, 1e-3, 10, 0.01,
       "SIGMA_MUSCLE_L", "stiffness", "Conductivity", "sigma_z.  Fibres run exactly along z, so the tensor is diagonal."),
    _p("conductivity.bone", "Bone conductivity", "S/m", "S/m", 1, 1e-3, 10, 0.005, "SIGMA_BONE", "stiffness", "Conductivity",
       "Isotropic."),
    # ---- electrodes (adjoint rows)
    _p("electrodes.array.n", "Number of electrodes", "1", "", 1, 2, 32, 1, "N_ELEC_LINEAR", "adjoint", "Electrodes",
       "Point electrodes; differential channels pair consecutive electrodes (n - 1 channels).", integer=True),
    _p("electrodes.array.spacing", "Inter-electrode distance", "m", "mm", 1e3, 1e-3, 30e-3, 0.5e-3, "ELEC_SPACING",
       "adjoint", "Electrodes", "Measured along the skin surface."),
    _p("electrodes.array.theta_center", "Array angle around the arm", "rad", "deg", DEG, -math.pi, math.pi,
       math.pi / 180, "ELEC_THETA", "adjoint", "Electrodes", "0 = +x side (above the default fibre)."),
    _p("electrodes.array.z_center", "Array centre along the arm", "m", "mm", 1e3, -60e-3, 60e-3, 1e-3, "ELEC_Z",
       "adjoint", "Electrodes", "Every electrode must stay at least one z-cell inside the end caps."),
    _p("electrodes.array.orientation", "Array orientation on the skin", "rad", "deg", DEG, -math.pi / 2, math.pi / 2,
       math.pi / 180, "ELEC_ORIENT", "adjoint", "Electrodes",
       "Angle between the array axis and the arm axis (0 = along the fibres, 90 deg = around the arm)."),
    # ---- fibre position (basis points)
    _p("fibre.depth", "Fibre depth below muscle surface", "m", "mm", 1e3, 0.5e-3, 12e-3, 0.5e-3, "FIBRE_DEPTH",
       "basis", "Fibre", "Must stay inside the muscle (above the bone)."),
    _p("fibre.theta", "Fibre angle around the arm", "rad", "deg", DEG, -math.pi, math.pi, math.pi / 180,
       "FIBRE_THETA", "basis", "Fibre"),
    _p("fibre.dz", "Fibre cell length", "m", "mm", 1e3, 0.1e-3, 2e-3, 0.1e-3, "FIBRE_DZ", "basis", "Fibre",
       "Discretisation of Eq. (10)."),
    # ---- fibre activity (signal only)
    _p("fibre.nmj_z", "Neuromuscular junction position z", "m", "mm", 1e3, -50e-3, 50e-3, 1e-3, "NMJ_Z", "signal", "Fibre"),
    _p("fibre.l1", "NMJ to +z tendon (L1)", "m", "mm", 1e3, 5e-3, 100e-3, 1e-3, "L1", "signal", "Fibre"),
    _p("fibre.l2", "NMJ to -z tendon (L2)", "m", "mm", 1e3, 5e-3, 100e-3, 1e-3, "L2", "signal", "Fibre"),
    _p("fibre.cv", "Conduction velocity (input)", "m/s", "m/s", 1, 1.0, 10.0, 0.1, "CV", "signal", "Fibre"),
    _p("fibre.taper", "Tukey taper length", "m", "mm", 1e3, 0.5e-3, 20e-3, 0.5e-3, "TAPER", "signal", "Fibre",
       "Smooth switch-on at the NMJ and switch-off at the tendons."),
    _p("fibre.radius", "Fibre radius", "m", "um", 1e6, 5e-6, 100e-6, 1e-6, "FIBRE_RADIUS", "signal", "Fibre",
       "Scales the source current (sigma_in * pi * r^2)."),
    _p("fibre.sigma_in", "Intracellular conductivity", "S/m", "S/m", 1, 0.1, 10, 0.01, "SIGMA_IN", "signal", "Fibre"),
    # ---- timing (signal only)
    _p("timing.duration", "Simulated duration", "s", "ms", 1e3, 5e-3, 100e-3, 1e-3, "DURATION_SINGLE", "signal", "Timing"),
    _p("timing.fs", "Sampling rate", "Hz", "kHz", 1e-3, 1000.0, 20000.0, 500.0, "FS_SINGLE", "signal", "Timing"),
]

# Controls the brief asks for that the current solver cannot honour.  Shown greyed out.
UNSUPPORTED = [
    dict(key="fibre.orientation", label="Fibre orientation (tilt from arm axis)", group="Fibre",
         reason="fibre.FibreGeometry is a straight line along z and the muscle conductivity tensor is "
                "diagonal in (x, y, z); a tilted fibre needs a rotated tensor and a 3-D source path."),
    dict(key="electrodes.contact_area", label="Electrode contact area / diameter", group="Electrodes",
         reason="fem.selection_matrix samples point electrodes (one barycentric row per electrode). "
                "Area averaging (paper Eq. 3, alternative form) is not implemented."),
    dict(key="electrodes.skin_impedance", label="Electrode-skin impedance", group="Electrodes",
         reason="Not modelled: electrodes are ideal passive voltage probes."),
    dict(key="tissues.separate_muscles", label="Separate biceps / triceps / brachialis", group="Geometry",
         reason="The FEM has a single homogeneous muscle annulus; individual muscles are drawn only as "
                "labelled illustrations."),
    dict(key="mechanics.poisson_ratio", label="Poisson ratio / deformation", group="Geometry",
         reason="Belongs to a mechanical deformation model, which this project does not have."),
    dict(key="stage4.motor_units", label="Motor units, recruitment, firing", group="Stage 4",
         reason="Implemented as scripts (muscle.py, stage4_muscle.py) but not yet wired into the app; "
                "see the /api/model stage list and POST /api/simulations/future."),
]

PARAM_KEYS = {p["key"] for p in PARAMETERS}


def get_path(cfg_dict, key):
    obj = cfg_dict
    for part in key.split("."):
        obj = obj[part]
    return obj


def diff_configs(a: dict, b: dict, prefix=""):
    """List of (key, a_value, b_value) for every leaf that differs."""
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        ka = f"{prefix}{k}"
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_configs(va, vb, ka + ".")
        elif va != vb:
            out.append((ka, va, vb))
    return out


def recompute_level(changed_keys):
    """Most expensive recompute level implied by a set of changed config keys."""
    level = len(RECOMPUTE_LEVELS) - 1
    by_key = {p["key"]: p["recompute"] for p in PARAMETERS}
    for k in changed_keys:
        if k.startswith("electrodes."):
            lv = "adjoint"
        else:
            lv = by_key.get(k, "mesh")      # unknown key -> be conservative
        level = min(level, RECOMPUTE_LEVELS.index(lv))
    return RECOMPUTE_LEVELS[level]
