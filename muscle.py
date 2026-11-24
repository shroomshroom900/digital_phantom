"""
muscle.py -- from single fibres to a whole muscle (paper: Biophysics steps 2-4 and
Methods "Geometrical and physiological modelling of motor units", "MU recruitment
model", "Details of realistic simulation examples").

Pipeline (each step is one function so Stage 5 can time it and show which steps
can be skipped when only some parameters change; compare the paper's Table 1):

    1. bipolar_adjoint        general forward solution   (Stage 2 machinery; ne solves)
    2. make_motor_units       MU distribution            (fibres -> MUs, in a unit circle)
    3. fibre_basis_points     fibre basis points         (Eq. 5 at every fibre cell)
    4. fibre_responses        fibre EMG responses        (Eq. 10 for every fibre)
    5. assemble_muaps         MUAP assembling            (sum of fibre signals per MU)
    6. spike_trains           recruitment + firing       (size principle, rate coding)
    7. assemble_semg          raw sEMG assembling        (MUAPs placed at spike times)

Tags:  [PAPER] stated in the paper,  [ASSUMED] my choice.
"""
import numpy as np
import scipy.sparse as sp
import config as C
import fem
import fibre

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_FIBRES = 50_000                   # [PAPER] "50k muscle fibres"
N_MU = 200                          # [PAPER] "200 motor units"
MU_SIZE_RANGE = (11, 1150)          # [PAPER] MU size varies exponentially 11 -> 1150 fibres
TERRITORY_RANGE = (0.10, 0.50)      # [PAPER] MU territory 10% -> 50% of the muscle cross-section
FR_RANGE = (8.0, 35.0)              # Hz  [PAPER] firing rate 8 -> 35 Hz
RT_MAX = 75.0                       # % MVC [PAPER] all MUs recruited at 75% MVC
RECRUIT_RANGE = 30.0                # ratio of largest/smallest threshold (Fuglevand 1993) [ASSUMED]
ISI_CV = 0.20                       # coefficient of variation of inter-spike intervals [ASSUMED]

# The "single muscle" = an elliptical muscle belly inside the muscle annulus of the
# cylinder (the paper uses the brachioradialis of a real arm).            [ASSUMED]
MUSCLE_CENTER = (13.5e-3, 0.0)      # m   (x, y); +x side of the arm
MUSCLE_AXES = (5.5e-3, 9.0e-3)      # m   semi-axes: radial (x) and tangential (y)
MUSCLE_DZ = 1.0e-3                  # m   fibre cell size in the muscle simulation   [ASSUMED]

# Electrodes: 8 bipolar pairs around the arm (the paper: "8 circular bipolar electrodes")
N_RING = 8
RING_Z = (5e-3, 25e-3)              # m  the two electrodes of each pair (20 mm apart) [ASSUMED]

# Per-fibre properties (the paper lists them but not their distributions)   [ASSUMED]
NMJ_SD, NMJ_CLIP = 2.5e-3, 6e-3     # NMJ position z0 ~ N(0, 2.5 mm), clipped to +-6 mm
TENDON_L = 40e-3                    # semi-fibre length from NMJ to tendon, +-3 mm uniform
TENDON_JITTER = 3e-3
CV_MEAN, CV_SD = 4.0, 0.3           # m/s conduction velocity per fibre, clipped to 3..5


# ---------------------------------------------------------------------------
# 1. bipolar electrodes + adjoint solves
# ---------------------------------------------------------------------------
def ring_electrodes(m):
    """Positions (n_ring, 2, 3): for each of 8 angles, the two electrodes of a pair."""
    th = 2 * np.pi * np.arange(N_RING) / N_RING
    pa = fem.skin_electrode_positions(m, th, np.array([RING_Z[0]]))
    pb = fem.skin_electrode_positions(m, th, np.array([RING_Z[1]]))
    return np.stack([pa, pb], axis=1)


def bipolar_adjoint(m, A, pos, rtol=1e-8):
    """Adjoint solve for BIPOLAR channels: channel j = (potential at electrode a_j) -
    (potential at electrode b_j).  Since everything is linear we fold the subtraction
    into the selection matrix:  S_bip = S_a - S_b  (rows now sum to 0 -- a nice bonus:
    the source vector needs no 'uniform sink' correction), then
    solve  A K^T = S_bip^T  with ONE CG solve per CHANNEL (8 solves, not 16).
    Returns K (8 x n_v) and the BasisSolutions object (Eq. 7).
    """
    Sa = fem.selection_matrix(m, pos[:, 0, :])
    Sb = fem.selection_matrix(m, pos[:, 1, :])
    S = (Sa - Sb).tocsr()
    K, iters = fem.adjoint_K(A, S, rtol=rtol)
    return K, fem.BasisSolutions(m, K)


# ---------------------------------------------------------------------------
# 2. Motor-unit distribution  (paper Fig. 6c)
# ---------------------------------------------------------------------------
def uniform_in_unit_circle(n, rng):
    r = np.sqrt(rng.uniform(size=n))          # sqrt -> uniform per unit AREA
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


class MotorUnits:
    pass


def make_motor_units(rng, n_fibres=N_FIBRES, n_mu=N_MU, uv=None):
    """Scatter fibres uniformly in the unit circle, then give them to motor units.

    1. fibres: uniform random points in the unit circle (so fibre density is uniform
       by construction -- the paper's first advantage of this method);
    2. MU sizes n_k grow exponentially with MU rank k (11 -> 1150 fibres, rescaled so
       that they add up to n_fibres); MU territories are circles of area fraction
       0.10 -> 0.50 of the muscle (radius rho_k = sqrt(fraction) in unit-circle units),
       centred at random points;  rank k = recruitment order (size principle: small first);
    3. each fibre picks ONE of the MUs whose territory contains it, with probability
       proportional to w_k * density_k, density_k = n_k / (area of territory k).

    ASSUMPTION: the paper says "probability proportional to the MU density" but does not
    say how the final MU sizes come out.  Taken literally, big dense MUs crowd out the
    small ones, so I add per-MU correction factors w_k (iterative proportional fitting)
    so that the expected number of fibres of each MU equals its target size n_k.
    """
    k = np.arange(n_mu)
    target = MU_SIZE_RANGE[0] * (MU_SIZE_RANGE[1] / MU_SIZE_RANGE[0]) ** (k / (n_mu - 1))
    target *= n_fibres / target.sum()
    rho = np.sqrt(np.linspace(*TERRITORY_RANGE, n_mu))
    centres = uniform_in_unit_circle(n_mu, rng)
    # `uv` (the fibre positions in the unit circle) may be passed in.  This is what lets
    # Stage 5 draw a NEW motor-unit distribution for the SAME fibres, so that the fibre
    # responses need not be recomputed (paper: "different MU distributions can be
    # generated very quickly, without regenerating fibres ...").
    if uv is None:
        uv = uniform_in_unit_circle(n_fibres, rng)

    dx = uv[:, 0:1] - centres[None, :, 0]
    dy = uv[:, 1:2] - centres[None, :, 1]
    d2 = dx * dx + dy * dy                                   # (n_fibres, n_mu)
    inside = d2 <= rho[None, :] ** 2
    uncovered = ~inside.any(axis=1)                          # rare: give them the MU whose
    inside[np.nonzero(uncovered)[0],                         # territory edge is nearest
           np.argmin(d2[uncovered] / rho[None, :] ** 2, axis=1)] = True
    density = target / (np.pi * rho ** 2)

    w = density.copy()
    for _ in range(60):                                      # fit w so E[size_k] = target_k
        p = inside * w[None, :]
        p /= p.sum(axis=1, keepdims=True)
        expected = p.sum(axis=0)
        w *= (target / expected) ** 0.7                      # damped update for stability
    p = inside * w[None, :]
    p /= p.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    mu_of = np.minimum((rng.uniform(size=(n_fibres, 1)) > cdf).sum(axis=1), n_mu - 1)

    mu = MotorUnits()
    mu.uv, mu.mu_of, mu.centres, mu.rho = uv, mu_of, centres, rho
    mu.target = target
    mu.size = np.bincount(mu_of, minlength=n_mu)
    mu.n_mu = n_mu
    # recruitment thresholds (Fuglevand 1993): exponential spread over the MU pool,
    # smallest MU = RT_MAX/RANGE, largest = RT_MAX  (all recruited by 75% MVC [PAPER])
    mu.threshold = RT_MAX * RECRUIT_RANGE ** (-(n_mu - 1 - k) / (n_mu - 1))
    return mu


def place_fibres(uv):
    """Map unit-circle points into the muscle cross-section (an ellipse here).
    The paper morphs the unit circle into the real muscle outline; for an
    elliptical belly the morph is just a stretch.  Returns (n, 2) x,y in metres."""
    x = MUSCLE_CENTER[0] + MUSCLE_AXES[0] * uv[:, 0]
    y = MUSCLE_CENTER[1] + MUSCLE_AXES[1] * uv[:, 1]
    return np.column_stack([x, y])


def make_fibre_properties(rng, n):
    z0 = np.clip(rng.normal(0.0, NMJ_SD, n), -NMJ_CLIP, NMJ_CLIP)
    L1 = TENDON_L + rng.uniform(-TENDON_JITTER, TENDON_JITTER, n)
    L2 = TENDON_L + rng.uniform(-TENDON_JITTER, TENDON_JITTER, n)
    v = np.clip(rng.normal(CV_MEAN, CV_SD, n), 3.0, 5.0)
    return dict(z0=z0, L1=L1, L2=L2, v=v)


# ---------------------------------------------------------------------------
# 3. Fibre basis points   Eq. (5) at every fibre cell
# ---------------------------------------------------------------------------
def fibre_basis_points(basis, xy, geom, chunk=500):
    """vpoint(r(z_i)) for every cell of every fibre.  Returns (n_f, n_ch, n_cells)."""
    nf, nc = len(xy), len(geom.z)
    out = np.empty((nf, basis.V.shape[0], nc))
    for s in range(0, nf, chunk):
        sl = slice(s, min(s + chunk, nf))
        n = sl.stop - sl.start
        pts = np.empty((n, nc, 3))
        pts[:, :, 0] = xy[sl, 0:1]
        pts[:, :, 1] = xy[sl, 1:2]
        pts[:, :, 2] = geom.z[None, :]
        vp = basis.point_potentials(pts.reshape(-1, 3))            # (n_ch, n*nc)
        out[sl] = vp.reshape(-1, n, nc).transpose(1, 0, 2)
    return out


# ---------------------------------------------------------------------------
# 4. Fibre responses   Eq. (10) for all fibres
# ---------------------------------------------------------------------------
def fibre_responses(Pf, geom, t, props, chunk=500):
    """Returns (n_f, n_ch, n_t) volts.  props: dict of per-fibre z0, L1, L2, v."""
    nf = Pf.shape[0]
    out = np.empty((nf, Pf.shape[1], len(t)))
    for s in range(0, nf, chunk):
        sl = slice(s, min(s + chunk, nf))
        out[sl] = fibre.responses_many(Pf[sl], geom, t, props["z0"][sl], props["L1"][sl],
                                       props["L2"][sl], props["v"][sl])
    return out


# ---------------------------------------------------------------------------
# 5. MUAP assembling   (MUAP = sum of the signals of its fibres)
# ---------------------------------------------------------------------------
def assemble_muaps(R, mu_of, n_mu):
    order = np.argsort(mu_of, kind="stable")
    counts = np.bincount(mu_of, minlength=n_mu)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ne = counts > 0
    muap = np.zeros((n_mu,) + R.shape[1:])
    muap[ne] = np.add.reduceat(R[order], starts[ne], axis=0)
    return muap


# ---------------------------------------------------------------------------
# 6. Recruitment (size principle) + rate coding -> discharge times
# ---------------------------------------------------------------------------
def excitation(t, T=2.0, peak=100.0):
    """Raised-cosine ramp: 0 -> peak -> 0 over T seconds (% MVC).
    (The paper says 'gradually increasing ... and smoothly decreasing'.)"""
    return peak * 0.5 * (1 - np.cos(2 * np.pi * t / T))


def spike_trains(exc_grid, fs, threshold, rng):
    """Discharge times of every MU.

    A MU is recruited when the excitation first exceeds its threshold RT_k
    (Henneman's size principle: small MUs have low thresholds).  While active its
    mean firing rate rises linearly with excitation, from 8 Hz at threshold to 35 Hz
    at 100% (my choice of gain; the paper only gives the 8-35 Hz range), and each
    inter-spike interval is  mean * (1 + CV * N(0,1))  (CV = 20%), floored at 0.3 mean.
    Returns a list of arrays of spike times (s).
    """
    n = len(exc_grid)
    T = n / fs
    fr_lo, fr_hi = FR_RANGE
    trains = []
    for rt in threshold:
        sp_t = []
        t = 0.0
        while t < T:
            # sample index of time t.  The +1e-6 guards against float round-off: without it a
            # time such as 0.35 s can map to sample 699 instead of 700, and the "wait for the
            # threshold" jump below would then land on the same sample forever (infinite loop;
            # found in Stage 5 when the peak excitation was lowered to 60% MVC).
            i = min(int(t * fs + 1e-6), n - 1)
            e = exc_grid[i]
            if e < rt:                                   # not recruited: wait for threshold
                nxt = np.nonzero(exc_grid[i:] >= rt)[0]
                if len(nxt) == 0:
                    break
                t = (i + nxt[0]) / fs
                continue
            sp_t.append(t)
            fr = fr_lo + (fr_hi - fr_lo) * (e - rt) / (100.0 - rt)
            isi = (1.0 / fr) * (1 + ISI_CV * rng.standard_normal())
            t += max(isi, 0.3 / fr)
        trains.append(np.array(sp_t))
    return trains


# ---------------------------------------------------------------------------
# 7. Raw sEMG assembling
# ---------------------------------------------------------------------------
def assemble_semg(muap, trains, fs, n_samples):
    """sEMG(t) = sum over MUs of  (MUAP_k  convolved with  its spike train).
    Spike times are rounded to the nearest sample (0.5 ms at 2 kHz, negligible
    against inter-spike intervals of 30-120 ms with 20% variability)."""
    n_ch, L = muap.shape[1], muap.shape[2]
    emg = np.zeros((n_ch, n_samples + L))
    for k, st in enumerate(trains):
        for idx in np.rint(st * fs).astype(int):
            emg[:, idx:idx + L] += muap[k]
    return emg[:, :n_samples]
