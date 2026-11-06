"""Draw the manuscript figures from reported result files only (../../results and paper/numbers.json)."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parent / "results"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
N = json.load(open(HERE / "numbers.json"))
WAITS = ("0", "100", "200", "400")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
C = dict(comet="#1f5fa8", ref="#8c8c8c", base="#d17a22", ego="#4d4d4d", fusion="#7a4fa0", late="#b5452b", ext="#2f8f6f")


def audit_figure():
    rows = list(csv.DictReader(open(RES / "dataset_audit/offset_grid/boundary.csv")))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.95], wspace=0.42))
    for k, axis in enumerate(("x", "y")):
        r = [x for x in rows if x["axis"] == axis]
        v = np.array([float(x["official_value"]) for x in r])
        med = np.array([float(x["displacement_median"]) for x in r])
        lo = np.array([float(x["p10"]) for x in r])
        hi = np.array([float(x["p90"]) for x in r])
        n = np.array([int(x["n"]) for x in r])
        a = ax[k]
        lim = (-2.1, 1.6)
        a.plot(lim, lim, color=C["ref"], lw=0.6, ls="--", zorder=0, label="identity")
        bound = np.isclose(v, -1.1) | np.isclose(v, 0.66)
        a.errorbar(v[~bound], med[~bound], yerr=[med[~bound] - lo[~bound], hi[~bound] - med[~bound]], fmt="o", ms=2.4,
                   color=C["comet"], ecolor=C["comet"], elinewidth=0.5, capsize=0, label="interior value")
        a.errorbar(v[bound], med[bound], yerr=[med[bound] - lo[bound], hi[bound] - med[bound]], fmt="s", ms=3.4,
                   color=C["late"], ecolor=C["late"], elinewidth=0.8, capsize=0, label="search bound")
        for xb in (-1.1, 0.66):
            a.axvline(xb, color=C["late"], lw=0.5, ls=":")
        for vv, mm, nn in zip(v[bound], med[bound], n[bound]):
            if vv < 0:
                a.annotate(f"n = {nn}", (vv, mm), xytext=(-0.85, -1.9), fontsize=6.2, color=C["late"],
                           arrowprops=dict(arrowstyle="-", lw=0.4, color=C["late"]))
            else:
                a.annotate(f"n = {nn}", (vv, mm), xytext=(0.82, 0.35), fontsize=6.2, color=C["late"],
                           arrowprops=dict(arrowstyle="-", lw=0.4, color=C["late"]))
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xticks([-2, -1, 0, 1]); a.set_yticks([-2, -1, 0, 1])
        a.set_xlabel(f"official offset, {axis} (m)")
        a.set_ylabel("co-labelled displacement (m)")
        a.set_title(f"({'ab'[k]}) {axis} axis", loc="left")
        if k == 0:
            a.legend(frameon=False, loc="upper left", fontsize=6.2, handlelength=1.2)
    a = ax[2]
    splits = [("fit24", "fit"), ("calibration8", "cal."), ("mechanism_holdout14", "held-out"), ("val21", "val.")]
    z = np.array([N["census"][s]["zero"] / N["census"][s]["messages"] for s, _ in splits])
    sat = np.array([N["saturation"][s]["saturated"] / N["saturation"][s]["frames"] for s, _ in splits])
    inner = 1 - z - sat
    xs = np.arange(len(splits))
    a.bar(xs, z, color=C["ref"], width=0.62, label="exactly zero (placeholder)")
    a.bar(xs, sat, bottom=z, color=C["late"], width=0.62, label="non-zero, on a bound")
    a.bar(xs, inner, bottom=z + sat, color=C["comet"], width=0.62, alpha=0.75, label="non-zero, interior")
    for i, s_ in enumerate(splits):
        a.text(i, z[i] / 2, f"{100*z[i]:.0f}%", ha="center", va="center", fontsize=5.8, color="white")
        a.text(i, z[i] + sat[i] / 2, f"{100*N['saturation'][s_[0]]['fraction']:.0f}%$^*$", ha="center", va="center", fontsize=5.8, color="white")
    a.set_xticks(xs, [s[1] for s in splits])
    a.set_ylim(0, 1); a.set_ylabel("fraction of messages")
    a.set_title("(c) offset per message", loc="left")
    a.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, handlelength=1.0)
    fig.savefig(FIG / "offset_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def trajectory_figure(sequence="0007"):
    r = [x for x in csv.DictReader(gzip.open(RES / "mechanism/KF_trajectories/KF_trajectories_author.csv.gz", "rt"))
         if x["role"] == "val21" and x["sequence"] == sequence]
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 3.6), sharex="col", gridspec_kw=dict(height_ratios=[1, 1, 0.42], hspace=0.12, wspace=0.18))
    for col, wait in enumerate(("0", "400")):
        s = sorted([x for x in r if x["wait_ms"] == wait], key=lambda x: int(x["receiver_timestamp_us"]))
        t = (np.array([int(x["receiver_timestamp_us"]) for x in s]) - int(s[0]["receiver_timestamp_us"])) / 1e6
        obs = np.array([x["observed"] == "True" for x in s])
        na = np.array([int(x["anchors"]) for x in s])
        var = np.array([float(x["posterior_variance_m2"]) for x in s])
        gaps = []
        start = None
        for i in range(len(obs)):
            if not obs[i] and start is None:
                start = i
            if (obs[i] or i == len(obs) - 1) and start is not None:
                gaps.append((t[start], t[i])); start = None
        for row, axis in enumerate(("x", "y")):
            a = ax[row, col]
            for g0, g1 in gaps:
                a.axvspan(g0, g1, color="#eceff1", lw=0, zorder=0)
            z = np.array([float(x[f"observation_{axis}_m"]) for x in s])
            d = np.array([float(x[f"delta_{axis}_m"]) for x in s])
            a.fill_between(t, d - 2 * np.sqrt(var), d + 2 * np.sqrt(var), color=C["comet"], alpha=0.16, lw=0, label="CoMET $\\pm2\\sigma$")
            a.scatter(t[obs], z[obs], s=3.5, color=C["base"], label="current message only", zorder=2)
            a.plot(t, d, color=C["comet"], lw=1.2, label="CoMET estimate", zorder=3)
            a.set_ylim(-2.6, 1.6); a.set_yticks([-2, -1, 0, 1])
            if col == 0:
                a.set_ylabel(f"offset {axis} (m)")
            if row == 0:
                a.set_title(f"latency budget $d$ = {wait} ms", loc="left")
        a = ax[2, col]
        a.bar(t, na, width=0.085, color=C["ego"])
        a.set_ylim(0, max(4, na.max() + 1)); a.set_yticks([0, 2, 4])
        if col == 0:
            a.set_ylabel("anchors")
        a.set_xlabel("time in sequence (s)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.savefig(FIG / "trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def latency_figure():
    t1 = N["table1"]; a1 = N["author_absolute_A1"]; r4 = N["robust_external_val21"]
    series = [
        ("Ego detector only", [a1["EGO"]["AP70"][w] for w in WAITS], C["ego"], ":", "o"),
        ("Late fusion, online pose", [a1["PUBLIC"]["AP70"][w] for w in WAITS], C["late"], "-", "v"),
        ("Official late fusion (best of two)", [t1["stronger_late_per_column"]["AP70"][w] for w in WAITS], C["late"], "--", "^"),
        ("V2X-ViT (3 seeds)", [t1["V2X-ViT_three_detector_seeds"]["AP70"][w] for w in WAITS], C["fusion"], "-", "s"),
        ("F-Cooper", [t1["fcooper_native_author_weight"]["AP70"][w] for w in WAITS], C["fusion"], "--", "D"),
        ("TraF-Align", [t1["traf_native_author_weight"]["AP70"][w] for w in WAITS], C["fusion"], ":", "P"),
        ("CBM + temporal median", [r4["cbm_median"]["AP70"][w] for w in WAITS], C["ext"], "-", "X"),
        ("CoMET (ours)", [t1["author|val21"]["AP70"][w] for w in WAITS], C["comet"], "-", "o"),
        ("Upper reference: dataset offset\n(uses annotations, not deployable)", [a1["OFFICIAL"]["AP70"][w] for w in WAITS], C["ref"], "--", None),
    ]
    fig, a = plt.subplots(figsize=(5.6, 2.6))
    x = [0, 1, 2, 3]
    for name, y, c, ls, m in series:
        a.plot(x, y, color=c, ls=ls, marker=m, ms=3.6, lw=1.8 if "CoMET" in name else 1.0, label=name, zorder=3 if "CoMET" in name else 2)
    a.set_xticks(x, ["0", "100", "200", "400"]); a.set_xlim(-0.2, 3.2)
    a.set_xlabel("latency budget $d$ (ms)"); a.set_ylabel("AP@0.7 (%)")
    a.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    a.legend(frameon=False, fontsize=6.6, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=2.2, labelspacing=0.45)
    fig.subplots_adjust(left=0.1, right=0.6, bottom=0.17, top=0.97)
    fig.savefig(FIG / "latency.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    audit_figure()
    trajectory_figure()
    latency_figure()
    print("figures written to", FIG)
