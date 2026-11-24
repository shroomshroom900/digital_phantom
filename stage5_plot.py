"""
stage5_plot.py -- draws the Stage 5 figure from the saved results (cache/cache_stage5.pkl).
stage5_timing.py calls make_figure() at its end; you can also run this file alone
(python3 stage5_plot.py) to redraw the figure without redoing the 10-minute timing run.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# The paper's Table 1 (seconds).  Its mesh has 2.1 M vertices and it used FEniCS.
PAPER = {"forward": 7 * 60.0, "basis_pts": 120.0, "response": 30.0, "muap": 0.8, "semg": 2.6}


def fmt_t(x):
    return f"{x:.2f} s" if x < 10 else f"{x:.1f} s"


HERE = os.path.dirname(os.path.abspath(__file__))


def make_figure(R, path=os.path.join(HERE, "figures", "stage5_timing.png")):
    Tb, LABEL, SCEN, meas = R["Tb"], R["LABEL"], R["SCENARIOS"], R["meas"]
    Tb_disp = dict(Tb)
    Tb_disp["forward"] = Tb["mesh"] + Tb["A"] + Tb["forward"]   # paper's step 1 = everything on the volume conductor
    rows_disp = ["forward", "mu", "basis_pts", "response", "muap", "spikes", "semg"]
    names_disp = ["general forward\nsolution", "MU distribution\n(not in paper's table)", "fibres\nbasis points",
                  "fibres EMG\nresponse", "MUAPs\nassembling", "recruitment + firing\n(not in paper's table)",
                  "raw sEMG\nassembling"]

    fig = plt.figure(figsize=(17, 11.5))
    gs = GridSpec(2, 6, height_ratios=[1.15, 1], hspace=0.5, wspace=0.9, figure=fig)

    # ---- (a) this code vs the paper's Table 1 ------------------------------------
    a = fig.add_subplot(gs[0, 0:2])
    y = np.arange(len(rows_disp))[::-1]
    a.barh(y + 0.2, [max(Tb_disp[r], 1e-3) for r in rows_disp], 0.38, color="C0",
           label="this code (116k-vertex mesh)")
    a.barh(y - 0.2, [PAPER.get(r, np.nan) for r in rows_disp], 0.38, color="C1",
           label="paper's Table 1 (2.1 M vertices)")
    a.set_xscale("log"); a.set_yticks(y); a.set_yticklabels(names_disp, fontsize=8)
    a.set_xlabel("time (s), log scale"); a.grid(alpha=0.3, axis="x")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    a.set_title("(a) Time of each pipeline step")
    for yy, r in zip(y, rows_disp):
        a.text(max(Tb_disp[r], 1e-3) * 1.15, yy + 0.2, f"{Tb_disp[r]:.2f}", va="center", fontsize=7)

    # ---- (b) what must be recomputed ----------------------------------------------
    a = fig.add_subplot(gs[0, 2:6])
    names = list(SCEN)
    mat_rows = ["forward", "place", "mu", "basis_pts", "response", "muap", "spikes", "semg"]
    for j, sn in enumerate(names):
        reuse = SCEN[sn]
        for i, r in enumerate(mat_rows):
            grp = ("mesh", "A", "forward") if r == "forward" else (r,)
            reused = all(g in reuse for g in grp)
            tt = sum(meas[sn][3][g] for g in grp)
            a.add_patch(plt.Rectangle((j, len(mat_rows) - 1 - i), 1, 1,
                                      fc="#d9d9d9" if reused else "#f28e2b", ec="w", lw=2))
            a.text(j + 0.5, len(mat_rows) - 1 - i + 0.5, "reused" if reused else fmt_t(tt),
                   ha="center", va="center", fontsize=8, color="#555" if reused else "k")
        w_part, w_full = meas[sn][0], meas[sn][1]
        a.text(j + 0.5, -0.45, fmt_t(w_part), ha="center", fontsize=11, fontweight="bold")
        sp = w_full / w_part
        a.text(j + 0.5, -1.0, "baseline" if j == 0 else (f"{sp:.1f}x faster" if sp < 10 else f"{sp:,.0f}x faster"),
               ha="center", fontsize=9, color="k" if j == 0 else "C2")
    a.set_xlim(0, len(names)); a.set_ylim(-1.3, len(mat_rows))
    a.set_xticks(np.arange(len(names)) + 0.5); a.set_xticklabels(names, fontsize=8)
    a.xaxis.tick_top()
    a.set_yticks(np.arange(len(mat_rows)) + 0.5)
    a.set_yticklabels([("forward solution\n(mesh + A + solves)" if r == "forward" else LABEL[r].replace("\n", " "))
                       for r in mat_rows][::-1], fontsize=8)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.legend(handles=[Patch(color="#f28e2b", label="recomputed (time taken)"),
                      Patch(color="#d9d9d9", label="reused from memory")],
             fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    a.set_title("(b) What must be recomputed when one thing changes\n"
                "(bold: measured wall-clock time of the short path;  green: speed-up vs. recomputing everything)",
                fontsize=9, pad=42)

    # ---- (c) solve time vs mesh size ------------------------------------------------
    a = fig.add_subplot(gs[1, 0:2])
    nv, ts, slope, icpt = R["nv"], R["ts"], R["slope"], R["icpt"]
    a.loglog(nv, ts, "o-", label="measured (one CG solve)")
    xx = np.array([nv[2], 2.1e6])
    a.loglog(xx, np.exp(icpt) * xx ** slope, "k--", lw=1, label=f"fit ~ n_v^{slope:.2f}, extrapolated")
    a.plot(2.1e6, 13, "C1s", ms=9, label="paper: 13 s / electrode")
    a.axvline(R["n_vertices_main"], color="grey", lw=0.6, ls=":")
    a.text(R["n_vertices_main"] * 1.1, ts.min() * 1.3, "our\nmain mesh", fontsize=7)
    a.set_xlabel("mesh vertices n_v"); a.set_ylabel("time per electrode (s)")
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3, which="both")
    a.set_title("(c) Cost of one solve grows with mesh size")

    # ---- (d) fibres ------------------------------------------------------------------
    a = fig.add_subplot(gs[1, 2:4])
    a.plot(R["nf_list"], R["t_bp"], "o-", label="fibres basis points")
    a.plot(R["nf_list"], R["t_rs"], "s-", label="fibres EMG response")
    a.set_xlabel("number of fibres"); a.set_ylabel("time (s)"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_title("(d) Fibre steps grow linearly with fibres")

    # ---- (e) electrodes ----------------------------------------------------------------
    a = fig.add_subplot(gs[1, 4:6])
    a.plot(R["n_ch_list"], R["t_fw"], "o-")
    a.set_ylim(0, max(R["t_fw"]) * 1.15)
    a.set_xlabel("number of bipolar channels n_e"); a.set_ylabel("forward solution time (s)")
    a.set_title(f"(e) Adjoint: one solve per channel ({R['t_fw'][-1]:.0f} s for 8).\n"
                f"Brute force: {R['n_s'] * R['t_solve'] / 3600:.0f} h, whatever n_e is", fontsize=10)
    a.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    with open(os.path.join(HERE, "cache", "cache_stage5.pkl"), "rb") as f:
        make_figure(pickle.load(f))
