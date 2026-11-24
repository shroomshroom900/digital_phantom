"""
stage4_muscle.py -- experiment B: one muscle, excitation 0 -> 100% MVC -> 0 over 2 s,
8 bipolar electrodes around the arm.   Run:  python3 stage4_muscle.py   (~1-2 min)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

import config as C
import fem
import fibre
import muscle as M
from mesh import build_cylinder_mesh

rng = np.random.default_rng(C.RNG_SEED)
T_RAMP, FS = 2.0, C.FS                       # 2 s ramp, 2000 Hz [PROMPT]
n_samp = int(T_RAMP * FS)
t_muap = np.arange(int(0.030 * FS)) / FS     # 30 ms MUAP window = 60 samples
timing = {}


def tic():
    return time.perf_counter()


# ------------------------------------------------------------ 1. forward solution
t0 = tic(); m = build_cylinder_mesh(); timing["mesh"] = tic() - t0
t0 = tic(); A = fem.assemble_stiffness(m); timing["assemble A"] = tic() - t0
pos = M.ring_electrodes(m)                                          # (8, 2, 3)
t0 = tic(); K, basis = M.bipolar_adjoint(m, A, pos); timing["general forward solution (8 solves)"] = tic() - t0
print(f"bipolar channels: 8 (electrodes at z = {M.RING_Z[0]*1e3:.0f} and {M.RING_Z[1]*1e3:.0f} mm), "
      f"n_s = {len(basis.muscle_nodes):,}")

# ------------------------------------------------------------ 2. motor units & fibres
t0 = tic(); mu = M.make_motor_units(rng); timing["MU distribution"] = tic() - t0
xy = M.place_fibres(mu.uv)
props = M.make_fibre_properties(rng, M.N_FIBRES)
print(f"{M.N_FIBRES:,} fibres in {mu.n_mu} MUs; realised MU size min/median/max = "
      f"{mu.size.min()}/{int(np.median(mu.size))}/{mu.size.max()} (targets {mu.target.min():.0f}-{mu.target.max():.0f}); "
      f"median |size error| = {100*np.median(np.abs(mu.size-mu.target)/mu.target):.0f}%")
geom = fibre.FibreGeometry(0.0, 0.0, dz=M.MUSCLE_DZ)

# ------------------------------------------------------------ 3-5. fibres -> MUAPs
t0 = tic(); Pf = M.fibre_basis_points(basis, xy, geom); timing["fibres basis points"] = tic() - t0
t0 = tic(); R = M.fibre_responses(Pf, geom, t_muap, props); timing["fibres EMG response"] = tic() - t0
t0 = tic(); muap = M.assemble_muaps(R, mu.mu_of, mu.n_mu); timing["MUAPs assembling"] = tic() - t0

# consistency check: the many-fibre code must equal the Stage 3 single-fibre code
f = 12345
prm = fibre.FibreParams(z0=props["z0"][f], L1=props["L1"][f], L2=props["L2"][f], v=props["v"][f])
r1 = fibre.simulate(Pf[f], geom, t_muap, prm)[0]
print(f"check, many-fibre vs single-fibre code (fibre {f}): relative difference "
      f"{np.abs(r1 - R[f]).max() / np.abs(r1).max():.1e}")
tail = np.abs(muap[:, :, -5:]).max() / np.abs(muap).max()
print(f"check, MUAP window long enough: |last 5 samples| / peak = {tail:.1e}")

# ------------------------------------------------------------ 6-7. recruitment & sEMG
t_grid = np.arange(n_samp) / FS
E = M.excitation(t_grid, T_RAMP)
t0 = tic(); trains = M.spike_trains(E, FS, mu.threshold, rng); timing["recruitment + firing (spike trains)"] = tic() - t0
t0 = tic(); emg = M.assemble_semg(muap, trains, FS, n_samp); timing["raw sEMG assembling"] = tic() - t0

print("\ntiming of the pipeline steps (cf. the paper's Table 1):")
for k, v in timing.items():
    print(f"  {k:<40s} {v:8.2f} s")
print(f"  {'TOTAL':<40s} {sum(timing.values()):8.2f} s")

# ------------------------------------------------------------ diagnostics
n_sp = sum(len(s) for s in trains)
active = np.array([[(s <= tt).any() and True for s in trains] for tt in [1.0]])  # recruited by peak
first = np.array([s[0] if len(s) else np.inf for s in trains])
print(f"\nspikes: {n_sp:,} in total; MUs recruited by t = 1 s: {int((first <= 1.0).sum())} of {mu.n_mu}"
      f" (all recruited at E >= {M.RT_MAX:.0f}% MVC)")
# check the size principle: recruitment time must increase with MU rank
rank_corr = np.corrcoef(np.arange(mu.n_mu), first)[0, 1]
print(f"size principle check: correlation(MU rank, recruitment time) = {rank_corr:.3f}")
# firing rate & ISI variability of the largest MU during its active period
isi_all = np.concatenate([np.diff(s) for s in trains[:150] if len(s) > 10])
# ISI jitter check at CONSTANT excitation (so the mean rate does not change and only the
# random 20% variability is left):
_Ec = np.full(int(20 * FS), 60.0)                           # 60 % MVC for 20 s
_tr = M.spike_trains(_Ec, FS, mu.threshold[[100]], np.random.default_rng(1))[0]
_isi = np.diff(_tr)
print(f"ISI check at constant 60% MVC (MU 100): mean rate {1/_isi.mean():.1f} Hz, "
      f"CV = {_isi.std()/_isi.mean():.2f}  (input {M.ISI_CV})")

# sliding RMS (100 ms window, 25 ms step)
win, step = int(0.1 * FS), int(0.025 * FS)
starts = np.arange(0, n_samp - win, step)
rms = np.array([np.sqrt((emg[:, s:s + win] ** 2).mean(axis=1)) for s in starts]).T   # (8, n_win)
t_win = (starts + win / 2) / FS
E_win = M.excitation(t_win, T_RAMP)
ang = np.arange(M.N_RING) * 360 // M.N_RING
pk = (t_win > 0.9) & (t_win < 1.1)
rms_pk = rms[:, pk].mean(axis=1)
print("\nRMS at peak excitation (mV), by electrode angle: " +
      ", ".join(f"{a}deg: {1e3*r:.3f}" for a, r in zip(ang, rms_pk)))
print(f"nearest (ch 1, 0 deg) / opposite (ch 5, 180 deg) RMS ratio = {rms_pk[0]/rms_pk[4]:.1f}")
up = t_win < 1.0
print(f"RMS rises with excitation: correlation(RMS ch1, E) over the whole ramp = "
      f"{np.corrcoef(rms[0], E_win)[0,1]:.3f};  RMS(E~100%) / RMS(E~10%) on ch 1 = "
      f"{rms[0][np.argmin(abs(E_win[up]-100))] / rms[0][np.argmin(abs(E_win[up]-10))]:.0f}x")
print(f"peak |sEMG| on channel 1: {1e3*np.abs(emg[0]).max():.2f} mV")

# =========================================================================== FIGURES
# ---- Figure 1: the muscle model ------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
a = ax[0, 0]
th = np.linspace(0, 2 * np.pi, 300)
for r, c in ((C.R_BONE, "0.6"), (C.R_MUSCLE, "C3"), (C.R_FAT, "gold"), (C.R_SKIN, "C1")):
    a.plot(r * 1e3 * np.cos(th), r * 1e3 * np.sin(th), color=c, lw=1)
sub = rng.choice(M.N_FIBRES, 8000, replace=False)
sc = a.scatter(xy[sub, 0] * 1e3, xy[sub, 1] * 1e3, c=mu.mu_of[sub], s=1.5, cmap="viridis")
for j in range(M.N_RING):
    p = pos[j, 0]
    a.plot(p[0] * 1e3, p[1] * 1e3, "ks", ms=5)
    a.text(0.80 * p[0] * 1e3, 0.80 * p[1] * 1e3, str(j + 1), ha="center", va="center", fontsize=10,
           color="k", fontweight="bold")            # label just inside the skin, away from the axes
a.set_aspect("equal"); a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
a.set_title("(a) Cross-section: 8000 of 50,000 fibres coloured by MU rank\nelectrodes 1-8 (bipolar pairs) on the skin")
fig.colorbar(sc, ax=a, label="MU rank (0 = smallest)", shrink=0.8)
a = ax[0, 1]
a.semilogy(mu.target, "k--", label="target size (exponential)")
a.semilogy(mu.size, "C0.", ms=4, label="realised size")
a.set_xlabel("MU rank (recruitment order)"); a.set_ylabel("fibres per MU")
a2 = a.twinx(); a2.plot(mu.threshold, "C3", label="recruitment threshold"); a2.set_ylabel("threshold (% MVC)", color="C3")
a.legend(loc="upper left", fontsize=8); a.set_title("(b) MU sizes and recruitment thresholds")
a = ax[1, 0]
Eg = np.linspace(0, 100, 400)
for k, c in ((0, "C0"), (100, "C2"), (199, "C1")):
    rt = mu.threshold[k]
    fr = np.where(Eg >= rt, M.FR_RANGE[0] + (M.FR_RANGE[1] - M.FR_RANGE[0]) * (Eg - rt) / (100 - rt), np.nan)
    a.plot(Eg, fr, c, label=f"MU {k} (size {mu.size[k]}, threshold {rt:.1f}%)")
a.set_xlabel("excitation (% MVC)"); a.set_ylabel("mean firing rate (Hz)"); a.legend(fontsize=8)
a.set_title("(c) Rate coding: 8 Hz at threshold -> 35 Hz at 100%"); a.grid(alpha=0.3)
a = ax[1, 1]
for k, c in ((5, "C0"), (100, "C2"), (199, "C1")):
    a.plot(t_muap * 1e3, 1e3 * muap[k, 0], c, label=f"MU {k} ({mu.size[k]} fibres)")
a.set_xlabel("time after the impulse (ms)"); a.set_ylabel("MUAP on channel 1 (mV)")
a.set_title("(d) Motor-unit action potentials: bigger MU -> bigger MUAP"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "stage4_model.png"), dpi=130, bbox_inches="tight")

# ---- Figure 2: experiment B ----------------------------------------------------
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, height_ratios=[1, 1.7, 1.2], figure=fig)
a = fig.add_subplot(gs[0, 0])
a.plot(t_grid, E, "k"); a.set_ylabel("excitation (% MVC)"); a.set_xlabel("time (s)")
n_active = np.array([(first <= tt).sum() for tt in t_grid])
a2 = a.twinx(); a2.plot(t_grid, n_active, "C3"); a2.set_ylabel("MUs recruited so far", color="C3")
a.set_title("(a) Excitation ramp and recruitment")
a = fig.add_subplot(gs[0, 1])
for k, s in enumerate(trains):
    a.plot(s, np.full(len(s), k), "k.", ms=0.6)
a.set_xlabel("time (s)"); a.set_ylabel("MU rank"); a.set_title(f"(b) Discharge times of all {mu.n_mu} MUs ({n_sp:,} spikes)")

a = fig.add_subplot(gs[1, :])
spc = np.abs(emg).max() * 0.55
for j in range(M.N_RING):
    a.plot(t_grid, 1e3 * emg[j] + 1e3 * spc * (7 - j), lw=0.5, color=f"C{j}")
    a.text(-0.02, 1e3 * spc * (7 - j), f"ch {j+1}\n({ang[j]}°)", ha="right", va="center", fontsize=8)
a.plot(t_grid, 1e3 * spc * 7 + 0 * t_grid, alpha=0)           # keep limits tidy
a.plot([1.85, 1.85], [-1e3 * spc * 0.9, -1e3 * spc * 0.9 + 1.0], "k", lw=2)
a.text(1.83, -1e3 * spc * 0.9 + 0.5, "1 mV", ha="right", va="center", fontsize=8)
a.set_yticks([]); a.set_xlabel("time (s)"); a.set_xlim(0, 2)
a.spines[["left", "right", "top"]].set_visible(False)
a.set_title("(c) Simulated sEMG, 8 bipolar channels (channel 1 is nearest the muscle)")

a = fig.add_subplot(gs[2, 0])
for j in range(M.N_RING):
    a.plot(t_win, 1e3 * rms[j], color=f"C{j}", label=f"ch {j+1}")
a.set_xlabel("time (s)"); a.set_ylabel("RMS (mV), 100 ms windows"); a.legend(fontsize=7, ncol=2)
a.set_title("(d) Amplitude follows the excitation"); a.grid(alpha=0.3)
a = fig.add_subplot(gs[2, 1])
a.bar(np.arange(1, 9), 1e3 * rms_pk, color=[f"C{j}" for j in range(8)])
a.set_xticks(np.arange(1, 9)); a.set_xticklabels([f"{x}\n{y}°" for x, y in zip(range(1, 9), ang)], fontsize=8)
a.set_ylabel("RMS at peak excitation (mV)"); a.set_xlabel("electrode (angle around the arm)")
a.set_title("(e) Larger at electrodes nearer the muscle")
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "stage4_exp_B.png"), dpi=130, bbox_inches="tight")
print("saved stage4_model.png, stage4_exp_B.png")
