"""Robustness figure for the R8c/R8d pose-error stress test (results/research_v2 only).

Writes fig/robustness.pdf (Figure "robustness" of the manuscript).
(a) Late fusion, val21 AP70 against a constant injected translation for the
    author and strict60 detectors: online pose, CoMET and CoMET-A, with the clean CoMET level.
(b) Native TraF-Align, val21 AP70 with the injected pose versus the pose corrected by CoMET-A.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parent / "results" / "research_v2" / "R8c_stress"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
C = dict(comet="#1f5fa8", acq="#2f8f6f", base="#d17a22", ref="#8c8c8c")
METHODS = (("PUBLIC", "online pose", C["base"], "s"), ("H1", "CoMET", C["comet"], "o"), ("CoMET_A", "CoMET + acquisition", C["acq"], "D"))


def late(domain):
    rows = [r for r in csv.DictReader(open(RES / "late" / domain / "results.csv"))
            if r["role"] == "val21" and r["threshold"] == "70" and r["kind"] == "T1"]
    amps = sorted({float(r["amplitude"]) for r in rows})
    out = {m: np.array([float(next(r["AP"] for r in rows if r["method"] == m and float(r["amplitude"]) == a)) for a in amps])
           for m, *_ in METHODS}
    ref = next(r for r in rows if r["method"] == "CoMET_A" and float(r["amplitude"]) == amps[-1])
    clean = float(ref["AP"]) / float(ref["ratio_clean_H1"])
    return np.array(amps), out, clean


def main():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.32))
    a = ax[0]
    for domain, ls, label in (("author", "-", "released"), ("strict60", "--", "strict")):
        amps, out, clean = late(domain)
        for m, name, color, marker in METHODS:
            a.plot(amps, out[m], ls=ls, color=color, marker=marker, ms=2.8, lw=1.0,
                   label=name if domain == "author" else None)
        a.axhline(clean, color=C["ref"], lw=0.6, ls=":")
        a.annotate(f"CoMET without injected error ({label})", (0.35 if domain == "author" else 2.75, clean + 0.15), fontsize=6,
                   color=C["ref"], va="bottom")
    a.axvline(2.5, color=C["ref"], lw=0.5, ls="-.")
    a.annotate("2.5 m gate", (2.55, 60.3), fontsize=6, color=C["ref"])
    a.set_xlim(0.3, 5.5); a.set_ylim(54.5, 67.6)
    a.set_xticks([0.5, 1, 2, 3, 5]); a.set_xticklabels(["0.5", "1", "2", "3", "5"])
    a.set_xlabel("injected constant translation (m)")
    a.set_ylabel("validation AP@0.7")
    a.set_title("(a) late fusion, TraF-Align boxes", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=mk, ms=2.8, lw=1.0, label=n) for _, n, c, mk in METHODS]
    handles += [Line2D([], [], color="k", lw=0.8, ls="-", label="released"), Line2D([], [], color="k", lw=0.8, ls="--", label="strict")]
    a.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5, columnspacing=1.2)

    b = ax[1]
    rows = [r for r in csv.DictReader(open(RES / "native" / "results.csv")) if r["threshold"] == "70"]
    width = 0.18
    for k, (model, name) in enumerate((("PUBLIC", "online-pose retraining"), ("A", "second tier"))):
        sel = sorted((r for r in rows if r["model"] == model), key=lambda r: float(r["amplitude_m"]))
        x = np.array([float(r["amplitude_m"]) for r in sel])
        un = np.array([float(r["uncorrected_AP"]) for r in sel]); co = np.array([float(r["corrected_AP"]) for r in sel])
        off = (-1.5 + 2 * k) * width
        b.bar(x + off, un, width, color=C["base"], alpha=0.45 if k == 0 else 0.85, label=f"{name}, injected pose")
        b.bar(x + off + width, co, width, color=C["acq"], alpha=0.45 if k == 0 else 0.85, label=f"{name}, corrected pose")
    b.set_ylim(60, 67.5); b.set_xticks([1, 2, 3])
    b.set_xlabel("injected constant translation (m)")
    b.set_ylabel("validation AP@0.7")
    b.set_title("(b) retrained TraF-Align", loc="left")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, ncol=1, handlelength=1.2)
    fig.savefig(FIG / "robustness.pdf", bbox_inches="tight")
    if "--png" in __import__("sys").argv:
        fig.savefig(FIG.parent.parent / "robustness_preview.png", bbox_inches="tight", dpi=170)
    print("wrote", FIG / "robustness.pdf")


if __name__ == "__main__":
    main()
