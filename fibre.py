"""
fibre.py -- EMG of ONE muscle fibre (paper: Methods, "EMG signal of a single
fibre activation", Eqs. 8, 9, 10).

Physical picture
----------------
A muscle fibre is a long thin cable.  A nerve impulse reaches the fibre at
the neuromuscular junction (NMJ, at position z0).  From there an electrical
wave -- the intracellular action potential (IAP) Vm -- runs in BOTH
directions along the fibre at speed v until it reaches the two ends
(tendons).  As the wave passes, current leaks across the fibre membrane into
the surrounding tissue; that leaking current is what the skin electrodes feel.

Sign convention used throughout (same as the FEM of Stage 2)
------------------------------------------------------------
current entering the surrounding tissue = positive "source"  (A x = b, b>0).

The three equations, in words
-----------------------------
Eq. (8)  How much current per metre of fibre leaks out at position z, time t:
             I(z,t) = sigma_in * pi * r^2 * d/dz F(z,t)
         with  F = psi_right * window1  -  psi_left * window2
         where psi = dVm/dz (slope of the action potential).
         (Cable theory: leaking current ~ 2nd derivative of Vm; we write it as
         d/dz of the slope, with the window INSIDE the derivative.  That has a
         neat consequence: the integral of I over the whole fibre is exactly 0,
         because it is the derivative of something that vanishes at both ends.
         Total current in = total current out, as physics requires.)
Eq. (9)  Voltage at an electrode = sum over the fibre of
             [voltage at the electrode per unit source at z] * I(z,t) dz
Eq. (10) Chop the fibre into small cells centred at z_i with half-width D:
             v_fibre(t) = sigma_in pi r^2 * sum_i vpoint(z_i) * [F(z_i+D,t) - F(z_i-D,t)]
         because the integral of dF/dz over a cell is just F at the right edge
         minus F at the left edge (fundamental theorem of calculus!) -- exact,
         no numerical integration needed.  vpoint(z_i) comes from Stage 2.
"""
import numpy as np
import config as C


# ----------------------------------------------------------------------
# Rosenfalck intracellular action potential  (paper: Vm(z) = 96 z^3 e^-z - 90)
# ----------------------------------------------------------------------
# Units of the formula: Vm in mV, z in mm.
# ORIENTATION: Rosenfalck's formula describes the profile at distance s >= 0
# BEHIND the wavefront (s = 0 at the front, rising to +39 mV peak at s = 3 mm,
# decaying back toward rest).  In front of the wave the fibre is at rest, -90 mV.
# For a wave moving in +z the wavefront is at z = z0 + v t, so
#         xi = z - z0 - v t  <= 0  behind the front,  s = -xi.
# We therefore define  vm(xi) = formula(s = -xi)  for xi < 0 and -90 mV for xi >= 0.
# (The formula is smooth at s=0: value, slope and curvature all vanish there, so
# clamping to rest in front of the wave creates no kink in the current.)
def vm_rosenfalck(xi):
    """Membrane potential (volts) at xi = z - z0 - v t (metres)."""
    s = np.maximum(-xi * 1e3, 0.0)                    # mm behind the front
    return 1e-3 * (96.0 * s ** 3 * np.exp(-s) - 90.0)


def psi_rosenfalck(xi):
    """psi = dVm/dxi  in V/m  (the 'slope' of the action potential).

    d/ds [96 s^3 e^-s] = 96 s^2 (3 - s) e^-s     (product rule),
    and  xi = -s  flips the sign;  mV/mm = V/m numerically.
    """
    s = np.maximum(-xi * 1e3, 0.0)
    return -96.0 * s ** 2 * (3.0 - s) * np.exp(-s)


# ----------------------------------------------------------------------
# Tukey window
# ----------------------------------------------------------------------
def tukey_window(z, a, b, taper):
    """Tukey ("tapered cosine") window equal to 0 outside [a, b], 1 in the
    middle and rising/falling smoothly over `taper` metres at each end:
        w = 0.5 (1 - cos(pi d / taper)),  d = distance from the nearest end.
    Identical to scipy's tukey window with alpha = 2*taper/(b-a).
    Used so the source switches on/off smoothly at NMJ and tendon (Eq. 8's w_L).
    """
    # (written with numpy functions only, so a, b may be arrays: Stage 4 gives every
    #  fibre its own NMJ position and tendon lengths.  If b <= a the window is 0.)
    L = b - a
    T = np.maximum(np.minimum(taper, L / 2), 1e-12)
    d = np.minimum(z - a, b - z)                       # distance to nearest end
    return np.where(d <= 0, 0.0, np.where(d >= T, 1.0, 0.5 * (1 - np.cos(np.pi * d / T))))


class FibreParams:
    """Everything that does NOT require re-solving the electrical problem."""

    def __init__(self, z0=C.NMJ_Z, L1=C.L1, L2=C.L2, v=C.CV, taper=C.TAPER,
                 radius=C.FIBRE_RADIUS, sigma_in=C.SIGMA_IN):
        self.z0, self.L1, self.L2, self.v, self.taper = z0, L1, L2, v, taper
        self.radius, self.sigma_in = radius, sigma_in


# ----------------------------------------------------------------------
# F(z,t)  -- the bracket inside Eq. (8)
# ----------------------------------------------------------------------
def F_source(z, t, prm):
    """F(z,t) = psi(z - z0 - vt) w1(z)  -  psi(z0 - z - vt) w2(z).      [V/m]

    z : positions (m), shape (n_z, 1);  t : times (s), shape (1, n_t).
    First term : wave travelling to the right (+z) towards tendon 1,
                 window w1 covers [z0, z0 + L1].
    Second term: mirror-image wave travelling left (-z) towards tendon 2,
                 window w2 covers [z0 - L2, z0].  Minus sign: mirror-reflecting
                 a wave flips the sign of its slope.
    NOTE on the brief's formula: it writes the left wave as psi(z + z0 - vt).
    With NMJ at position z0 that argument does not describe a wave that starts
    at z0 and runs left; the mirror image about z0 is psi(z0 - z - vt), which is
    what I implement (my derivation; same structure and sign as Eq. 8).
    """
    w1 = tukey_window(z, prm.z0, prm.z0 + prm.L1, prm.taper)
    w2 = tukey_window(z, prm.z0 - prm.L2, prm.z0, prm.taper)
    right = psi_rosenfalck(z - prm.z0 - prm.v * t) * w1
    left = psi_rosenfalck(prm.z0 - z - prm.v * t) * w2
    return right - left


def csd(z, t, prm, dz=1e-4):
    """Current source density I(z,t) = sigma_in pi r^2 dF/dz   (A/m).  Eq. (8).
    (Only for plotting: the simulation itself never differentiates numerically.)"""
    zz = z[:, None]
    F_hi = F_source(zz + dz / 2, t, prm)
    F_lo = F_source(zz - dz / 2, t, prm)
    return prm.sigma_in * np.pi * prm.radius ** 2 * (F_hi - F_lo) / dz


# ----------------------------------------------------------------------
# Fibre geometry + the "pre-compute once" part
# ----------------------------------------------------------------------
class FibreGeometry:
    """A straight fibre parallel to the z axis at lateral position (x, y).

    Discretised into n cells of length dz centred at z_i (paper: points r(z_i)
    with half-widths D_i = dz/2).  The stored range is the whole anatomical
    fibre; NMJ position and tendon lengths (z0, L1, L2) just choose which part
    is electrically active, so they can change WITHOUT redoing vpoint.
    """

    def __init__(self, x, y, z_half=C.FIBRE_Z_RANGE, dz=C.FIBRE_DZ):
        n = int(round(2 * z_half / dz))
        self.edges = -z_half + dz * np.arange(n + 1)          # cell edges  z_i -/+ D
        self.z = 0.5 * (self.edges[:-1] + self.edges[1:])      # cell centres z_i
        self.dz = dz
        self.pts = np.column_stack([np.full(n, x), np.full(n, y), self.z])


def precompute_basis_points(basis, geom):
    """vpoint(r(z_i)) for all cells: Eq. (5) applied to every fibre point.
    Returns P of shape (n_e, n_cells).  The expensive-ish part; do it ONCE per
    fibre position/geometry (it only needs V_basis; no FEM solve!)."""
    return basis.point_potentials(geom.pts)


def responses_many(Pf, geom, t, z0, L1, L2, v, taper=C.TAPER,
                   radius=C.FIBRE_RADIUS, sigma_in=C.SIGMA_IN):
    """Eq. (10) for MANY fibres at once, each with its own NMJ z0, tendon lengths
    L1, L2 and velocity v (arrays of length n_f).

    Pf : (n_f, n_e, n_cells)  vpoint of every fibre's cells (own lateral position)
    returns (n_f, n_e, n_t) fibre signals in volts.
    """
    assert (z0 - L2 >= geom.edges[0]).all() and (z0 + L1 <= geom.edges[-1]).all()
    prm = FibreParams(z0=z0[:, None, None], L1=L1[:, None, None], L2=L2[:, None, None],
                      v=v[:, None, None], taper=taper, radius=radius, sigma_in=sigma_in)
    Fe = F_source(geom.edges[None, :, None], t[None, None, :], prm)     # (n_f, n+1, n_t)
    D = Fe[:, 1:, :] - Fe[:, :-1, :]                                     # (n_f, n, n_t)
    return sigma_in * np.pi * radius ** 2 * np.matmul(Pf, D)             # batched P @ D


def simulate(P, geom, t, prm):
    """Eq. (10) as a matrix product.  Returns monopolar voltages (n_e, n_t), volts.

    D[i,t] = F(z_i + dz/2, t) - F(z_i - dz/2, t)   (current in cell i, per sigma pi r^2)
    v      = sigma_in pi r^2 * P @ D
    """
    assert prm.z0 - prm.L2 >= geom.edges[0] and prm.z0 + prm.L1 <= geom.edges[-1], \
        "active fibre must lie inside the stored fibre geometry"
    Fe = F_source(geom.edges[:, None], t[None, :], prm)       # (n+1, n_t) F at cell edges
    D = Fe[1:] - Fe[:-1]                                      # (n, n_t)
    return prm.sigma_in * np.pi * prm.radius ** 2 * (P @ D), D
