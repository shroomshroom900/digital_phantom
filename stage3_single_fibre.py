"""
stage3_single_fibre.py -- experiment A: one fibre, 16 electrodes, 15 differential channels.
Run:  python3 stage3_single_fibre.py   (uses cache/cache_linear_array.npz from Stage 2,
                                         otherwise it recomputes it in ~15 s)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config as C
import fem
import fibre
from mesh import build_cylinder_mesh

# ---------------------------------------------------------------- set-up
m = build_cylinder_mesh()
if os.path.exists(os.path.join(C.CACHE_DIR, "cache_linear_array.npz")):
    cache = np.load(os.path.join(C.CACHE_DIR, "cache_linear_array.npz"))
    K, elec = cache["K"], cache["elec"]
    assert K.shape[1] == len(m.nodes), "cache belongs to a different mesh"
else:
    z = (np.arange(C.N_ELEC_LINEAR) - (C.N_ELEC_LINEAR - 1) / 2) * C.ELEC_SPACING
    elec = fem.skin_electrode_positions(m, 0.0, z)
    A = fem.assemble_stiffness(m)
    K, _ = fem.adjoint_K(A, fem.selection_matrix(m, elec))
basis = fem.BasisSolutions(m, K)
ne = K.shape[0]

t = np.arange(int(0.030 * C.FS_SINGLE)) / C.FS_SINGLE        # 30 ms
prm = fibre.FibreParams()

# ---------------------------------------------------------------- the fibre
x_f = C.R_MUSCLE - C.FIBRE_DEPTH
geom = fibre.FibreGeometry(x=x_f, y=0.0)
t0 = time.perf_counter()
P = fibre.precompute_basis_points(basis, geom)                # (16, 220)  Eq. (5)
t_basis = time.perf_counter() - t0
t0 = time.perf_counter()
mono, D = fibre.simulate(P, geom, t, prm)                     # Eq. (10)
t_sig = time.perf_counter() - t0
diff = mono[:-1] - mono[1:]                                   # 15 channels: e_k - e_(k+1)
print(f"fibre: {len(geom.z)} cells of {geom.dz*1e3:.1f} mm at depth {C.FIBRE_DEPTH*1e3:.0f} mm, "
      f"NMJ at z={prm.z0*1e3:.0f} mm, tendons at {1e3*(prm.z0-prm.L2):.0f} and {1e3*(prm.z0+prm.L1):.0f} mm")
print(f"  vpoint for all cells: {t_basis*1e3:.1f} ms;  signal (Eq. 10): {t_sig*1e3:.1f} ms")

# ---------------------------------------------------------------- checks
# (1) current conservation: the cell currents must add up to zero at every instant
cons = np.abs(D.sum(axis=0)).max() / np.abs(D).max()
print(f"  check 1, total fibre current = 0:  max|sum_i D_i| / max|D_i| = {cons:.1e}")
# so adding a constant to P (the arbitrary potential offset of Stage 2) changes nothing:
shift = np.abs(fibre.simulate(P + 123.0, geom, t, prm)[0] - mono).max() / np.abs(mono).max()
print(f"  check 2, gauge invariance: adding 123 V/A to every vpoint changes the signal by {shift:.1e} (relative)")

# (3) conduction velocity from the delays between the propagating channels 7..11
#     Method: cross-correlate the waveforms of two channels (after 20x interpolation);
#     the lag at the correlation maximum is the travel time between them.
#     (Picking the largest peak of each channel is unreliable: it can land on
#     different phases of the biphasic waveform in different channels.)
def xcorr_lag(a, b):
    tt = np.linspace(0, t[-1], len(t) * 20)
    fa, fb = np.interp(tt, t, a), np.interp(tt, t, b)
    cc = np.correlate(fb - fb.mean(), fa - fa.mean(), "full")
    return (np.argmax(cc) - (len(tt) - 1)) * (tt[1] - tt[0])
v_est = np.mean([(b - a) * C.ELEC_SPACING / xcorr_lag(diff[a - 1], diff[b - 1])
                 for a, b in ((8, 10), (8, 11), (7, 10))])
print(f"  check 3, velocity from cross-correlation of channels 8-10, 8-11, 7-10: "
      f"{v_est:.2f} m/s  (input {prm.v} m/s)")

# (4) NMJ cancellation: channel 4 sits symmetrically about the NMJ
p2p = diff.max(axis=1) - diff.min(axis=1)
print(f"  check 4, peak-to-peak per channel (uV): " + " ".join(f"{1e6*p:.2f}" for p in p2p))
print(f"           channel 4 / mean(channels 3,5) = {p2p[3] / p2p[[2, 4]].mean():.3f}")
prm_sym = fibre.FibreParams(L1=0.030, L2=0.030)               # symmetric tendons
d_sym = fibre.simulate(P, geom, t, prm_sym)[0]
d_sym = d_sym[:-1] - d_sym[1:]
print(f"           with symmetric tendons (L1=L2=30 mm): channel 4 p2p = "
      f"{1e6*(d_sym[3].max()-d_sym[3].min()):.2e} uV, i.e. "
      f"{100*(d_sym[3].max()-d_sym[3].min())/(d_sym[[2, 4]].max()-d_sym[[2, 4]].min()):.1f}% of channels 3/5 "
      f"(not zero: the Kuhn tets are not mirror-symmetric along z -- my hypothesis, not tested)")

# ---------------------------------------------------------------- depth sweep
depths = np.array([1, 3, 5, 7, 9, 11]) * 1e-3
amp = []
for d in depths:
    g = fibre.FibreGeometry(x=C.R_MUSCLE - d, y=0.0)
    s = fibre.simulate(fibre.precompute_basis_points(basis, g), g, t, prm)[0]
    dd = s[:-1] - s[1:]
    amp.append(dd[8].max() - dd[8].min())                     # channel 9
amp = np.array(amp)
print("  depth sweep, channel 9 peak-to-peak (uV): " +
      ", ".join(f"{1e3*d:.0f} mm: {1e6*a:.2f}" for d, a in zip(depths, amp)))

# ---------------------------------------------------------------- speed of re-simulating
n_rep = 200
t0 = time.perf_counter()
for i in range(n_rep):                                        # change v, NMJ, tendons: reuse P
    fibre.simulate(P, geom, t, fibre.FibreParams(z0=-0.02 + 2e-5 * i, v=3 + 0.01 * i, L2=0.025))
t_re = (time.perf_counter() - t0) / n_rep
print(f"  re-simulating after changing v / NMJ / tendon: {t_re*1e3:.1f} ms each "
      f"(vs {t_basis*1e3:.1f} ms to recompute vpoint, and ~{ne*0.73:.0f} s for a new FEM solve)")

# ================================================================= FIGURES
# ---- Figure 1: the source model -------------------------------------------------
zf = np.arange(-55e-3, 55e-3, 1e-4)
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
xi = np.linspace(-25e-3, 5e-3, 600)
a = ax[0, 0]
a.plot(xi * 1e3, fibre.vm_rosenfalck(xi) * 1e3, "k")
a.set_xlabel("position relative to wavefront  z - z0 - vt  (mm)"); a.set_ylabel("Vm (mV)")
a.set_title("(a) Rosenfalck action potential  Vm = 96 s³ e⁻ˢ − 90\n(wave moves right; front at 0, tail behind)")
a.axhline(-90, color="grey", lw=0.5, ls=":"); a.grid(alpha=0.3)
a = ax[0, 1]
a.plot(xi * 1e3, fibre.psi_rosenfalck(xi), "C3")
a.set_xlabel("position relative to wavefront (mm)"); a.set_ylabel("ψ = dVm/dz  (mV/mm)")
a.set_title("(b) Its slope ψ: what Eq. 8 differentiates once more"); a.grid(alpha=0.3)
a = ax[1, 0]
zz = zf
a.plot(zz * 1e3, fibre.tukey_window(zz, prm.z0, prm.z0 + prm.L1, prm.taper), label="w1 (right half-fibre)")
a.plot(zz * 1e3, fibre.tukey_window(zz, prm.z0 - prm.L2, prm.z0, prm.taper), label="w2 (left half-fibre)")
a.axvline(prm.z0 * 1e3, color="k", ls="--", lw=0.8); a.text(prm.z0 * 1e3 + 1, 0.3, "NMJ", rotation=90, va="center")
a.set_xlabel("z along the fibre (mm)"); a.set_ylabel("window"); a.legend(fontsize=8, loc="lower right")
a.set_title(f"(c) Tukey windows ({prm.taper*1e3:.0f} mm tapers): smooth switch-on/off"); a.grid(alpha=0.3)
a = ax[1, 1]
for tt, c in ((2e-3, "C0"), (6e-3, "C1"), (14e-3, "C2")):
    I = fibre.csd(zf, np.array([[tt]]), prm)[:, 0]
    a.plot(zf * 1e3, I * 1e3 * 1e0, color=c, label=f"t = {tt*1e3:.0f} ms")
a.set_xlabel("z (mm)"); a.set_ylabel("current source density I (µA/mm)")
a.set_title("(d) Eq. 8: leaking current, travelling both ways\n(t=14 ms: right wave dying at the tendon)")
a.legend(fontsize=8); a.grid(alpha=0.3); a.axhline(0, color="k", lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "stage3_source.png"), dpi=130, bbox_inches="tight")

# ---- Figure 2: experiment A -----------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], figure=fig)
a = fig.add_subplot(gs[:, 0])
spacing = np.abs(diff).max() * 0.9
for k in range(15):
    a.plot(t * 1e3, 1e6 * diff[k] + 1e6 * spacing * (14 - k), "k", lw=0.9)
    a.text(-0.6, 1e6 * spacing * (14 - k), str(k + 1), ha="right", va="center", fontsize=8)
a.set_yticks([])
a.set_xlim(0, 30); a.set_xlabel("time (ms)")
a.set_title("(a) Experiment A: 15 differential channels from ONE fibre (3 mm deep)")
# channel c (1-based) is drawn at height  spacing*(15 - c)  -> brackets use the same rule
for (c0, c1), lab, col in (((4, 4), "NMJ: signals cancel", "C3"),
                           (((1, 3)), "non-propagating\n(NMJ generation,\nleft end)", "C1"),
                           (((5, 6)), "non-propagating\n(NMJ generation)", "C1"),
                           (((7, 11)), "propagating", "C2"),
                           (((12, 15)), "end-of-fibre\n(right tendon)", "C0")):
    ytop = 1e6 * spacing * (15 - c0 + 0.4)
    ybot = 1e6 * spacing * (15 - c1 - 0.4)
    a.plot([30.3, 30.6, 30.6, 30.3], [ytop, ytop, ybot, ybot], color=col, lw=1.5, clip_on=False)
    a.text(30.9, (ytop + ybot) / 2, lab, color=col, fontsize=8, va="center")
sb = 10 ** np.floor(np.log10(np.abs(diff).max() * 1e6))          # a "nice" amplitude scale bar (uV)
y0 = -spacing * 1e6 * 0.9                                       # bars live in the empty bottom-left corner
a.plot([1, 1], [y0, y0 + sb], "k", lw=2)
a.text(1.6, y0 + sb / 2, f"{sb:g} µV", ha="left", va="center", fontsize=8)
a.plot([6, 11], [y0, y0], "k", lw=2); a.text(8.5, y0 - 0.35 * sb, "5 ms", ha="center", va="top", fontsize=8)
a.spines[["left", "right", "top"]].set_visible(False)

a = fig.add_subplot(gs[0, 1])                       # geometry sketch
# vertical axis = height above the fibre in mm (stretched ~4x relative to the z axis)
d = C.FIBRE_DEPTH * 1e3                                   # fibre depth below muscle surface (mm)
fat_t = (C.R_FAT - C.R_MUSCLE) * 1e3                      # 3 mm
skin_t = (C.R_SKIN - C.R_FAT) * 1e3                       # 1 mm
a.axhspan(-3, d, color="#c0392b", alpha=0.25, lw=0)                    # muscle
a.axhspan(d, d + fat_t, color="#f4d03f", alpha=0.35, lw=0)             # fat
a.axhspan(d + fat_t, d + fat_t + skin_t, color="#f5cba7", alpha=0.7, lw=0)   # skin
a.text(-58, 0.9 * -2, "muscle", fontsize=7); a.text(-58, d + 1.0, "fat", fontsize=7)
a.text(-58, d + fat_t + 0.3, "skin", fontsize=7)
a.plot([(prm.z0 - prm.L2) * 1e3, (prm.z0 + prm.L1) * 1e3], [0, 0], "C3", lw=5, solid_capstyle="butt", label="active fibre")
a.plot(prm.z0 * 1e3, 0, "k*", ms=14, label="NMJ")
y_el = d + fat_t + skin_t
a.plot(elec[:, 2] * 1e3, np.full(ne, y_el), "ks", ms=5, label="skin electrodes (point)")
for n in range(ne):
    a.text(elec[n, 2] * 1e3, y_el + 0.8, str(n + 1), ha="center", fontsize=7)
a.set_ylim(-3, y_el + 2.5); a.set_xlim(-60, 60); a.set_yticks([])
a.set_xlabel("z (mm)"); a.set_title("(b) Geometry (side view; vertical axis stretched)")
a.legend(fontsize=7, loc="lower right")

a = fig.add_subplot(gs[1, 1])
a.semilogy(depths * 1e3, amp * 1e6, "o-")
a.set_xlabel("fibre depth below the muscle surface (mm)"); a.set_ylabel("channel 9 peak-to-peak (µV)")
a.set_title("(c) Deeper fibre -> smaller signal (volume conduction)"); a.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "stage3_exp_A.png"), dpi=130, bbox_inches="tight")
print("saved stage3_source.png, stage3_exp_A.png")
