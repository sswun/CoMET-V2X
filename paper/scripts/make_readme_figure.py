"""Render the README trace from the same validation replay as manuscript Fig. 1."""
from pathlib import Path

import numpy as np

from make_fig_overview import load_trace
import matplotlib.pyplot as plt


def main():
    rows = load_trace(sequence="0071", wait="0", lo=68, hi=152)
    index = np.arange(68, 68 + len(rows))
    anchors = np.array([int(row["anchors"]) for row in rows])
    observed = anchors > 0
    raw = np.array([float(row["observation_x_m"]) for row in rows])
    mean = np.array([float(row["delta_x_m"]) for row in rows])
    sd = np.sqrt([float(row["posterior_variance_m2"]) for row in rows])
    last = np.flatnonzero(observed)[-1]

    ink, teal, orange = "#182d3b", "#08696b", "#ac511f"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, (counts, trace) = plt.subplots(2, 1, sharex=True, figsize=(9.6, 3.75),
                                        gridspec_kw={"height_ratios": [1, 2.6]})
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.23, top=0.85, hspace=0.14)
    fig.text(0.022, 0.944, "Tracking a roadside message stream", color=ink,
             weight="bold", size=15, va="center")
    fig.text(0.975, 0.942, "V2X-Seq SPD  /  seq. 0071  /  0 ms", color="#435967",
             size=10, ha="right", va="center")
    for ax in (counts, trace):
        ax.axvspan(index[last] + 0.5, index[-1] + 0.5, color="#f0f4f6", zorder=0)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color("#aab7bb")
        ax.tick_params(length=0, pad=4)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e0e6e8", lw=0.7)
        ax.yaxis.set_label_coords(-0.085, 0.5)
    counts.bar(index, anchors, color="#64828b", width=0.74)
    counts.set_ylabel("Anchors")
    counts.set_yticks([0, 5, 10])
    counts.text((index[last] + index[-1] + 1) / 2, 7,
                f"{len(rows) - last - 1} messages without anchors",
                ha="center", va="center", color=ink, fontsize=10)
    band = trace.fill_between(index, mean - 2 * sd, mean + 2 * sd, color="#b7dbd9",
                              label="±2 posterior SD")
    dots = trace.scatter(index[observed], raw[observed], s=17, color=orange,
                         linewidths=0, label="Single-message observation", zorder=2)
    line, = trace.plot(index, mean, color=teal, lw=2.6, label="CoMET estimate")
    trace.set(xlim=(index[0] - 0.7, index[-1] + 0.5), ylim=(-1.8, 0.65),
              ylabel="x offset (m)", xlabel="Message index")
    trace.set_yticks([-1.5, -1, -0.5, 0, 0.5])
    fig.legend(handles=[line, dots, band], loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.53, 0.005))
    output = Path(__file__).resolve().parents[2] / "docs/figures/message-stream.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=185, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
