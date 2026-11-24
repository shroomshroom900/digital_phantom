"""
config.py  --  every number the simulation depends on lives here.

Why a config file?  When you later change a number (e.g. the muscle
conductivity) you change it in ONE place and every stage picks it up.

UNITS: everything inside the code is SI (metres, seconds, siemens/metre,
volts).  Plots convert to cm / mm / ms / uV for readability.

Legend for provenance tags:
  [PAPER]    -> stated in Maksymenko et al. 2023 (Nat. Commun. 14:1600)
  [PROMPT]   -> given in your project brief
  [ASSUMED]  -> NOT in the paper; my choice (flagged so you can question it)
"""
import os
import numpy as np

# Output folders (created on import, always next to this file, wherever you run from)
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")      # all PNG figures
CACHE_DIR = os.path.join(ROOT, "cache")      # large intermediate arrays (safe to delete)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Geometry: four concentric layers (a "cylindrical limb" phantom)
# ----------------------------------------------------------------------
# Radii are measured from the cylinder axis (the z axis).  [PAPER, Methods:
# "Comparison with the cylindrical analytical solution"]
R_BONE   = 0.7e-2    # m   bone      occupies 0   < r <= 0.7 cm
R_MUSCLE = 2.0e-2    # m   muscle    occupies 0.7 < r <= 2.0 cm
R_FAT    = 2.3e-2    # m   fat       occupies 2.0 < r <= 2.3 cm
R_SKIN   = 2.4e-2    # m   skin      occupies 2.3 < r <= 2.4 cm

# Length of the cylinder along z.  The paper only says "sufficiently longer
# than the fibre and the electrode array", so the number is [ASSUMED].
# 12 cm: fibres will be ~8 cm long, leaving ~2 cm of margin at each end so
# the (artificial) insulating end-caps don't distort the fields too much.
LENGTH = 12.0e-2     # m   [ASSUMED]

# Integer labels for the tissues.  We store one label per tetrahedron.
BONE, MUSCLE, FAT, SKIN = 0, 1, 2, 3
TISSUE_NAMES = ["bone", "muscle", "fat", "skin"]

# ----------------------------------------------------------------------
# 2. Mesh resolution   (structured grid -> each box split into 6 tets)
# ----------------------------------------------------------------------
# The paper uses an unstructured mesh with 2.1 M vertices built by CGAL.
# We cannot (no CGAL), so we use a regular box grid.  [ASSUMED]
#   - 1 mm in x,y  : needed so the 1 mm-thick skin is at least one box wide
#   - 2 mm in z    : fibres run along z and the field varies slowly there
H_XY = 1.0e-3        # m   [ASSUMED]
H_Z  = 2.0e-3        # m   [ASSUMED]

# A much coarser mesh for the brute-force validation in Stage 3 (experiment D).
H_XY_COARSE = 4.0e-3   # m  [ASSUMED]
H_Z_COARSE  = 8.0e-3   # m  [ASSUMED]

# ----------------------------------------------------------------------
# 3. Electrical conductivities  (S/m = siemens per metre)
# ----------------------------------------------------------------------
# The paper does NOT list its conductivity values, so these are typical
# literature values as used in the cylindrical-model tradition the paper
# builds on (Farina et al. 2004, IEEE TBME 51:415 -- ref 23 in the paper).
# I am quoting them from memory (no internet), so treat them as
# "reasonable textbook numbers", not exact citations.  [ASSUMED]
#
# Muscle is ANISOTROPIC: current flows ~5x more easily ALONG the fibres
# (z axis) than ACROSS them (x, y axes).
SIGMA = {
    BONE:   (0.020, 0.020, 0.020),   # S/m  isotropic
    MUSCLE: (0.100, 0.100, 0.500),   # S/m  (sx, sy, sz): sz = along fibres
    FAT:    (0.040, 0.040, 0.040),   # S/m  isotropic
    SKIN:   (1.000, 1.000, 1.000),   # S/m  isotropic
}
# Because the fibres of our cylinder run exactly along z, the conductivity
# tensor sigma is DIAGONAL: diag(sx, sy, sz).  (In a real arm the tensor is
# rotated to follow each fibre's local direction.)

# ----------------------------------------------------------------------
# 3b. Electrode array for experiment A (also used by the Stage 2 checks)
# ----------------------------------------------------------------------
# 16 point electrodes in a line along z on the skin at angle 0 (the +x side).
# The paper does not give the spacing for this example.        [ASSUMED]
N_ELEC_LINEAR = 16
ELEC_SPACING = 5.0e-3     # m  -> array is 7.5 cm long, centred on z = 0

# ----------------------------------------------------------------------
# 3c. Single-fibre parameters (Stage 3)
# ----------------------------------------------------------------------
# Fibre radius and intracellular conductivity appear in Eq. (8)/(10) but the
# paper gives no numbers.  Typical literature values:              [ASSUMED]
FIBRE_RADIUS = 25e-6      # m   (50 um diameter)
SIGMA_IN = 1.01           # S/m intracellular conductivity (Plonsey/Farina value)
CV = 4.0                  # m/s conduction velocity (typical 3-5 m/s)   [ASSUMED]
# Tukey-window taper length at each end of each semi-fibre.  The paper only
# says "Tukey window"; this sets how smoothly the source switches on at the
# NMJ and off at the tendon.                                           [ASSUMED]
TAPER = 5e-3              # m

# Geometry of the example fibre in experiment A:
FIBRE_DEPTH = 3e-3        # m below the muscle surface                  [ASSUMED]
NMJ_Z = -20e-3            # m  NMJ placed between electrodes 4 and 5 so channel 4
                          #    is the "NMJ channel" (as in the paper's Fig. 3a)
L1 = 60e-3                # m  NMJ -> right tendon (towards +z)         [ASSUMED]
L2 = 25e-3                # m  NMJ -> left tendon  (towards -z)         [ASSUMED]
FIBRE_Z_RANGE = 55e-3     # m  fibre geometry stored for z in [-55, +55] mm
FIBRE_DZ = 0.5e-3         # m  spacing of the discretisation points z_i
FS_SINGLE = 10000.0       # Hz sampling for the single-fibre plot only: at 2 kHz an
                          # AP (~2.5 ms long) has too few samples to look at. [ASSUMED]

# ----------------------------------------------------------------------
# 4. Misc.
# ----------------------------------------------------------------------
RNG_SEED = 2023       # makes every random choice reproducible
FS = 2000.0           # Hz, sampling frequency of simulated signals [PROMPT]


def sigma_array(labels):
    """Look up the diagonal conductivity (sx, sy, sz) for each tet.

    labels : (n_tets,) int array of tissue labels
    returns: (n_tets, 3) float array, in S/m
    """
    table = np.array([SIGMA[k] for k in range(4)])   # shape (4, 3)
    return table[labels]
