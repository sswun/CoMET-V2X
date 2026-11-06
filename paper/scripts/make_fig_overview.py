"""Figure 1: (a) receiver-side geometry of the common-mode error, (b) tracking it over a real
validation message stream (archived CoMET states, results/mechanism/KF_trajectories), (c) late
fusion with a protected ego prefix and a frozen readout head.
Usage: python3 make_fig_overview.py"""
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge, Circle
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
C_GT, C_EGO, C_REM, C_COR, C_ANC = "#9e9e9e", "#1f5fa8", "#c62828", "#2e7d32", "#ef6c00"
C_ROAD, C_BLD = "#ececec", "#d7ccc8"


def corners(cx, cy, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return p @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def box(ax, cx, cy, yaw, ec, fc="none", ls="-", lw=0.9, l=4.4, w=2.0, z=3, alpha=1.0):
    ax.add_patch(Polygon(corners(cx, cy, l, w, yaw), closed=True, ec=ec, fc=fc, ls=ls, lw=lw, zorder=z, alpha=alpha))


def arrow(ax, p, q, color, lw=0.9, style="-|>", ms=6, z=5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, t, color="black", fs=6.0, ha="left"):
    ax.text(x, y, t, fontsize=fs, color=color, ha=ha, va="center", zorder=12,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def panel_a(ax):
    ax.set_xlim(0, 60); ax.set_ylim(-4, 40); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 14), 60, 12, fc=C_ROAD, ec="none", zorder=0))
    ax.add_patch(Rectangle((30, 0), 12, 40, fc=C_ROAD, ec="none", zorder=0))
    for x0, y0, w, h in [(0, 26.6, 29.4, 13.4), (42.6, 26.6, 17.4, 13.4), (0, 0, 29.4, 13.4), (42.6, 0, 17.4, 13.4)]:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=C_BLD, ec="#bcaaa4", lw=0.5, zorder=1))
    for (a, b) in [((0, 20), (30, 20)), ((42, 20), (60, 20)), ((36, 26), (36, 40)), ((36, 0), (36, 14))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.8, ls=(0, (4, 3)), zorder=1)
    rsu = np.array([43.4, 27.2])
    ax.add_patch(Wedge(rsu, 34, 140, 290, fc="#ffe0b2", ec="none", alpha=0.45, zorder=0.5))
    ax.add_patch(Rectangle((rsu[0] - 0.35, rsu[1]), 0.7, 3.0, fc="#424242", ec="none", zorder=6))
    ax.add_patch(Circle(rsu + [0, 3.4], 1.0, fc="#ff8f00", ec="#424242", lw=0.5, zorder=7))
    label(ax, 45.2, 34.2, "infrastructure", fs=6.2)
    ego = np.array([5.5, 16.9])
    ax.add_patch(Polygon(corners(*ego, 4.6, 2.1, 0.0), closed=True, fc=C_EGO, ec="black", lw=0.5, zorder=6))
    ax.add_patch(Polygon(corners(ego[0] + 1.0, ego[1], 1.0, 1.7, 0.0), closed=True, fc="#bbdefb", ec="none", zorder=7))
    label(ax, ego[0] - 1.6, 12.2, "ego", color=C_EGO, fs=6.4)
    ax.add_patch(Polygon(corners(21.0, 23.2, 9.0, 2.8, 0.0), closed=True, fc="#616161", ec="black", lw=0.4, zorder=4))
    ax.text(18.3, 23.2, "truck", fontsize=5.5, color="white", va="center", zorder=8)
    arrow(ax, rsu + [-1.0, 3.6], ego + [1.0, 1.5], "#6d4c41", lw=0.7, ls=(0, (3, 2)), rad=0.32, ms=5)
    label(ax, 2.0, 37.3, r"message $m_\tau$: boxes, scores, $\tau$", color="#6d4c41", fs=5.9)
    label(ax, 2.0, 34.6, r"admissible if $\tau\leq t-d$; online pose", color="#6d4c41", fs=5.9)
    delta = np.array([-1.8, 0.8])
    co_vis = [(47.5, 22.8, 0.0), (55.0, 17.1, np.pi)]
    excl = [(38.7, 33.5, np.pi / 2), (33.2, 7.0, -np.pi / 2), (27.6, 23.1, 0.0)]
    for cx, cy, yaw in co_vis + excl:
        box(ax, cx, cy, yaw, C_GT, fc="#e0e0e0", lw=0.5, z=2)
    for cx, cy, yaw in co_vis:
        box(ax, cx + 0.12, cy - 0.1, yaw, C_EGO, lw=1.0, z=4)
        r = np.array([cx, cy]) - delta
        box(ax, *r, yaw, C_REM, ls=(0, (2.5, 1.5)), lw=0.9, z=4)
        arrow(ax, r, (cx + 0.12, cy - 0.1), C_ANC, lw=1.1, ms=6, z=7)
    for cx, cy, yaw in excl:
        r = np.array([cx, cy]) - delta
        box(ax, *r, yaw, C_REM, ls=(0, (2.5, 1.5)), lw=0.9, z=4)
        box(ax, cx, cy, yaw, C_COR, lw=1.1, z=5)
        arrow(ax, r, (cx, cy), C_COR, lw=0.9, ms=5, z=7)
    label(ax, 44.0, 10.6, "co-visible anchors:", color=C_ANC, fs=5.9)
    label(ax, 44.0, 7.6, r"$\mathbf{z}_k=\mathbf{x}^{\rm ego}_k-\tilde{\mathbf{x}}_k$", color=C_ANC, fs=6.4)
    label(ax, 3.0, 9.6, "roadside-only (occluded):", color=C_COR, fs=5.9)
    label(ax, 3.0, 6.9, r"$\hat{\mathbf{x}}_j=\tilde{\mathbf{x}}_j+\hat{\boldsymbol{\delta}}_t$", color=C_COR, fs=6.4)
    # common-mode glyph
    g1 = np.array([10.2, 1.4]); g0 = g1 + 3.4 * delta
    arrow(ax, g1, g0, "black", lw=1.2, ms=8, z=9)
    ax.text(g1[0] + 1.0, g1[1] + 0.9, r"one $\boldsymbol{\delta}_t$ shared by all objects of $m_\tau$", fontsize=5.9, va="center", zorder=9)
    ly = -2.6
    items = [(C_GT, "GT", "-", "#e0e0e0"), (C_EGO, "ego det.", "-", "none"), (C_REM, "remote, online pose", (0, (2.5, 1.5)), "none"),
             (C_COR, "remote + CoMET", "-", "none")]
    xs = [0.6, 7.6, 18.6, 39.6]
    for (c, t, ls, fc), x in zip(items, xs):
        ax.add_patch(Rectangle((x, ly - 0.7), 2.6, 1.4, ec=c, fc=fc, ls=ls, lw=0.8, zorder=9))
        ax.text(x + 3.2, ly, t, fontsize=5.8, va="center", zorder=9)
    ax.set_title(r"(a) a shared offset $\boldsymbol{\delta}_t$ displaces every remote box", fontsize=7.5, pad=2)


def load_trace(sequence="0071", wait="0", lo=68, hi=152):
    path = Path(__file__).resolve().parents[2] / "results/mechanism/KF_trajectories/KF_trajectories_author.csv.gz"
    rows = [r for r in csv.DictReader(gzip.open(path, "rt")) if r["role"] == "val21" and r["sequence"] == sequence and r["wait_ms"] == wait]
    rows.sort(key=lambda r: int(r["receiver_timestamp_us"]))
    return rows[lo:hi]


def panel_b(ax_m, ax_d, T):
    n = len(T)
    idx = np.arange(n)
    m = np.array([int(t["anchors"]) for t in T])
    x = np.array([float(t["delta_x_m"]) for t in T])
    sd = np.sqrt(np.array([float(t["posterior_variance_m2"]) for t in T]))
    zx = np.array([float(t["observation_x_m"]) if t["observed"] == "True" else np.nan for t in T])
    ax_m.bar(idx, m, width=0.8, color="#90a4ae", zorder=2)
    zero = m == 0
    ax_m.scatter(idx[zero], np.full(zero.sum(), 0.45), marker="x", s=6, color=C_REM, lw=0.5, zorder=3)
    ax_m.set_ylabel("$n_t$", fontsize=7, labelpad=1)
    top = int(np.ceil(m.max() / 5.0) * 5)
    ax_m.set_xlim(-1, n); ax_m.set_ylim(0, top * 1.45); ax_m.set_xticks([]); ax_m.set_yticks([0, top // 2, top])
    ax_m.tick_params(labelsize=5.5, length=1.5, pad=1)
    ax_m.spines[["top", "right"]].set_visible(False)
    ax_m.text(0.99, 0.97, r"anchors per message (Hungarian + mutual NN $\leq g$; ${\times}$: none)", transform=ax_m.transAxes, ha="right", va="top", fontsize=5.5)
    ax_m.set_title(r"(b) tracking $\boldsymbol{\delta}_t$ over a real message stream", fontsize=7.5, pad=2)
    runs, start = [], None
    for i in range(n):
        if zero[i] and start is None:
            start = i
        if (not zero[i] or i == n - 1) and start is not None:
            runs.append((start, i if not zero[i] else i + 1)); start = None
    for a_, b_ in runs:
        ax_d.axvspan(a_ - 0.5, b_ - 0.5, color="#eceff1", zorder=0, lw=0)
    ax_d.fill_between(idx, x - 2 * sd, x + 2 * sd, color="#bbdefb", alpha=0.9, lw=0, zorder=1, label=r"$\hat\delta_{x}\pm2\sqrt{p_t}$")
    ax_d.scatter(idx, zx, s=3.5, color=C_ANC, zorder=3, label=r"current message only, $\bar z_{x}$")
    ax_d.plot(idx, x, color=C_EGO, lw=1.2, zorder=4, label=r"CoMET $\hat\delta_{x}$")
    ax_d.set_xlim(-1, n)
    ax_d.set_xlabel("message index (10 Hz)", fontsize=6.5, labelpad=1)
    ax_d.set_ylabel("offset $x$ (m)", fontsize=6.5, labelpad=1)
    ax_d.tick_params(labelsize=5.5, length=1.5, pad=1)
    ax_d.spines[["top", "right"]].set_visible(False)
    lo = np.nanmin(np.r_[x - 2 * sd, zx[~np.isnan(zx)]]); hi = np.nanmax(np.r_[x + 2 * sd, zx[~np.isnan(zx)]])
    ax_d.set_ylim(lo - 0.12, hi + 0.42)
    ax_d.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, 1.0), framealpha=0.95, handlelength=1.1, borderpad=0.25, labelspacing=0.15, columnspacing=0.8, ncol=3, handletextpad=0.3)
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        k = min(b - 1, n - 1)
        km = min(a + 12, b - 1)
        ax_d.annotate("no anchors:\nestimate kept,\nuncertainty grows", xy=(km, x[km] + 2 * sd[km]), xytext=(km - 2, hi + 0.05),
                      fontsize=5.2, ha="left", va="top", arrowprops=dict(arrowstyle="-|>", lw=0.5, color="black", mutation_scale=5))


def panel_c(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(c) late fusion and frozen readout", fontsize=7.5, pad=2)
    rows = [("ego", C_EGO, "-", "#bbdefb", "E1"), ("ego", C_EGO, "-", "#bbdefb", "E2"),
            ("remote", C_COR, "-", "none", "R1"), ("remote (dup.)", C_COR, "-", "none", "D"),
            ("remote", C_COR, "-", "none", "R2"), ("ego, weak", C_EGO, (0, (2, 1.5)), "none", "W")]
    ys = {}
    for k, (t, c, ls, fc, key) in enumerate(rows):
        y = 8.55 - k * 1.0
        ys[key] = y
        ax.add_patch(Rectangle((0.55, y - 0.3), 0.75, 0.6, ec=c, fc=fc, ls=ls, lw=0.9))
        ax.text(1.45, y, t, fontsize=5.7, va="center")
    ax.plot([0.35, 0.25, 0.25, 0.35], [ys["E1"] + 0.38, ys["E1"] + 0.38, ys["E2"] - 0.38, ys["E2"] - 0.38], color=C_EGO, lw=0.8)
    ax.text(0.12, (ys["E1"] + ys["E2"]) / 2, r"$\mathcal{E}^+_t$", fontsize=6.5, color=C_EGO, ha="right", va="center")
    ax.plot([0.35, 0.25, 0.25, 0.35], [ys["R1"] + 0.38, ys["R1"] + 0.38, ys["W"] - 0.38, ys["W"] - 0.38], color="#455a64", lw=0.8)
    ax.text(0.12, (ys["R1"] + ys["W"]) / 2, r"$\mathcal{C}_t$", fontsize=6.5, color="#455a64", ha="right", va="center")
    mx = [4.45, 5.25, 6.05]
    layers = [np.linspace(3.2, 6.0, 5), np.linspace(3.55, 5.65, 4), np.array([4.6])]
    for a_ in range(2):
        for ya in layers[a_]:
            for yb in layers[a_ + 1]:
                ax.plot([mx[a_], mx[a_ + 1]], [ya, yb], color="#b0bec5", lw=0.35, zorder=1)
    for x_, L in zip(mx, layers):
        for y_ in L:
            ax.add_patch(Circle((x_, y_), 0.12, fc="white", ec="#455a64", lw=0.6, zorder=2))
    ax.text(3.25, 7.05, r"$\mathbf{f}_c$: 39-D, no position", fontsize=5.2, ha="left")
    ax.text(mx[1], 2.85, "32", fontsize=5.5, ha="center")
    ax.text(mx[2] + 0.1, 5.05, r"$u_c$", fontsize=6.4, ha="center")
    for key in ("R1", "D", "R2", "W"):
        arrow(ax, (3.3, ys[key]), (mx[0] - 0.22, 4.6), "#90a4ae", lw=0.45, ms=4, z=1)
    ax.text(7.0, 9.25, "ranked output", fontsize=5.9)
    ranked = [(C_EGO, "#bbdefb", "-", None), (C_EGO, "#bbdefb", "-", None), (C_COR, "none", "-", 0.92),
              (C_COR, "none", "-", 0.81), (C_EGO, "none", (0, (2, 1.5)), 0.33), (C_COR, "none", "-", 0.06)]
    for k, (c, fc, ls, u) in enumerate(ranked):
        y = 8.55 - k * 1.0
        ax.text(6.85, y, f"{k + 1}", fontsize=5.4, va="center", color="#757575")
        ax.add_patch(Rectangle((7.1, y - 0.28), 0.65, 0.56, ec=c, fc=fc, ls=ls, lw=0.9))
        if u is None:
            ax.text(7.9, y, "by score $s$", fontsize=5.4, va="center", color=C_EGO)
        else:
            ax.add_patch(Rectangle((7.9, y - 0.17), 1.45 * u, 0.34, fc="#ffcc80", ec="none"))
            ax.text(7.95 + 1.45 * u, y, f"{u:.2f}", fontsize=5.0, va="center")
    arrow(ax, (mx[2] + 0.2, 4.6), (6.95, 5.6), "#90a4ae", lw=0.5, ms=5)
    ax.text(0.0, 1.6, r"$y_c=\mathbb{1}\!\left[\max_{g\,\notin\,\mathcal{M}(\mathcal{E}^+_t)}\mathrm{IoU}(c,g)\geq0.7\right]$", fontsize=6.1)
    ax.text(0.0, 0.55, "trained once on the fit split, then frozen", fontsize=5.6, color="#455a64")


def main():
    tr = load_trace()
    fig = plt.figure(figsize=(6.9, 2.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1.05, 1.0], height_ratios=[0.40, 1.0], wspace=0.17, hspace=0.10,
                          left=0.005, right=0.995, top=0.93, bottom=0.11)
    panel_a(fig.add_subplot(gs[:, 0]))
    panel_b(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), tr)
    panel_c(fig.add_subplot(gs[:, 2]))
    out = Path(__file__).resolve().parents[1] / "fig/overview.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
