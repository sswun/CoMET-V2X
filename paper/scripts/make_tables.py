"""Generate LaTeX table bodies (paper/tables/*.tex) from paper/numbers.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)
N = json.load(open(HERE / "numbers.json"))
W = ("0", "100", "200", "400")


def f2(x):
    return f"{x:.2f}"


def sgn(x, nd=2):
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.{nd}f}"


def ci(lo, hi, nd=2):
    def g(v):
        return ("$-$" if v < 0 else "") + f"{abs(v):.{nd}f}"
    return f"[{g(lo)}, {g(hi)}]"


def write(name, text):
    (OUT / name).write_text(text.strip() + "\n")


def row4(label, d70, d50, bold=False, gray=False):
    vals = [d70[w] for w in W] + [d70["mean"] if "mean" in d70 else d70["four_wait_mean"]] + [d50[w] for w in W] + [d50["mean"] if "mean" in d50 else d50["four_wait_mean"]]
    cells = [f2(v) for v in vals]
    if bold:
        cells = [r"\textbf{" + c + "}" for c in cells]
    line = label + " & " + " & ".join(cells) + r" \\"
    if gray:
        line = r"\rowcolor{gray!12} " + line
    return line


def main_table():
    t1, a1, r4 = N["table1"], N["author_absolute_A1"], N["robust_external_val21"]
    L = []
    L.append(row4("Ego detector only", a1["EGO"]["AP70"], a1["EGO"]["AP50"]))
    L.append(r"\midrule")
    L.append(row4("F-Cooper~\\cite{chen2019fcooper}", t1["fcooper_native_author_weight"]["AP70"], t1["fcooper_native_author_weight"]["AP50"]))
    L.append(row4("TraF-Align~\\cite{song2025trafalign}", t1["traf_native_author_weight"]["AP70"], t1["traf_native_author_weight"]["AP50"]))
    L.append(row4("V2X-ViT~\\cite{xu2022v2xvit} (3 seeds)", t1["V2X-ViT_three_detector_seeds"]["AP70"], t1["V2X-ViT_three_detector_seeds"]["AP50"]))
    L.append(r"\midrule")
    L.append(row4("Official late fusion~\\cite{yu2022dairv2x} (best of two)", t1["stronger_late_per_column"]["AP70"], t1["stronger_late_per_column"]["AP50"]))
    L.append(row4("Late fusion, online pose$^\\dagger$", a1["PUBLIC"]["AP70"], a1["PUBLIC"]["AP50"]))
    for k, lab in (("freealign_instant", "FreeAlign~\\cite{lei2024freealign}$^\\dagger$"), ("roco_instant", "RoCo~\\cite{huang2024roco}$^\\dagger$"), ("cbm_instant", "CBM~\\cite{song2023cbm}$^\\dagger$"), ("cbm_median", "CBM + temporal median$^{\\dagger\\ddagger}$")):
        L.append(row4(lab, r4[k]["AP70"], r4[k]["AP50"]))
    L.append(r"\midrule")
    L.append(row4("\\textbf{CoMET (ours)}$^\\dagger$", t1["author|val21"]["AP70"], t1["author|val21"]["AP50"], bold=True))
    write("main.tex", "\n".join(L))


def cross_detector():
    names = dict(author="TraF-Align (released)", strict60="TraF-Align (strict retrain)", v2xvit_seed2024="V2X-ViT, seed A", v2xvit_seed2025="V2X-ViT, seed B", where2comm="Where2comm")
    L = []
    for d in ("author", "strict60", "v2xvit_seed2024", "v2xvit_seed2025", "where2comm"):
        cd = N["comet_cross_detector"][d]
        P = N["population_split"][d]
        cells = [names[d]]
        for role in ("val21", "mechanism_holdout14"):
            cells.append(f2(cd[role]["AP70"]["mean"]))
            for comp in ("H1_minus_PUBLIC", "H1_minus_single_mean"):
                e = P[f"{role}|{comp}|all|70"]
                cells.append(sgn(e["delta"]))
        L.append(" & ".join(cells) + r" \\")
    write("cross_detector.tex", "\n".join(L))


def temporal():
    lab = [("single_mean", "Current message, mean"), ("single_median", "Current message, median"), ("hold_last", "Hold last observation"), ("EMA", "Exponential moving average"), ("window", "Sliding-window mean"), ("KF", "\\textbf{Common-mode KF (CoMET)}")]
    L = []
    for k, name in lab:
        cells = [name]
        for role in ("val21", "mechanism_holdout14"):
            d = N["temporal"][role][k]
            cells += [f2(d["ap70"]["mean"]), f2(d["ap50"]["mean"])]
        L.append(" & ".join(cells) + r" \\")
    write("temporal.tex", "\n".join(L))


def ablation():
    desc = {
        "N0": ("confident ego anchors (score $\\ge\\tau_p$)", "Hungarian + mutual NN", "KF"),
        "U0": ("all ego boxes", "Sinkhorn, $\\sigma{=}1$\\,m, centred on estimate", "KF"),
        "U1": ("all ego boxes", "Sinkhorn, $\\sigma{=}1$\\,m, uncentred", "KF"),
        "U2": ("confident ego anchors", "Sinkhorn", "KF"),
        "U3": ("confident ego anchors", "Hungarian, prediction-gated", "KF"),
        "U5": ("all ego boxes", "Sinkhorn, $\\sigma{=}0.5$\\,m", "KF"),
        "U6": ("all ego boxes", "Sinkhorn, $\\sigma{=}2$\\,m", "KF"),
        "N2": ("all ego boxes", "learned soft assignment", "KF"),
        "H1": ("\\textbf{all ego boxes (\\method)}", "\\textbf{Hungarian + mutual NN}", "\\textbf{KF}"),
        "H2": ("all ego boxes", "Hungarian, prediction-gated (2.5\\,m)", "KF"),
        "H3": ("all ego boxes", "Hungarian, prediction-gated (1.5\\,m)", "KF"),
    }
    L = []
    for k in ("N0", "U2", "U3", "U0", "U1", "U5", "U6", "N2", "H2", "H3", "H1"):
        a, b, c = desc[k]
        cells = [a, b]
        for role in ("calibration8", "val21", "mechanism_holdout14"):
            cells.append(f2(N["ablation"][role][k]["ap70"]["mean"]))
        L.append(" & ".join(cells) + r" \\")
    write("ablation.tex", "\n".join(L))


def sensitivity():
    s = N["sensitivity_author_AP70"]
    thr = [("score_all", "all"), ("score_0.1", "0.1"), ("score_0.2", "0.2"), ("score_0.3", "0.3"), ("score_0.4", "0.4"), ("score_0.5", "0.5")]
    gate = [("gate_1.5", "1.5"), ("gate_2.0", "2.0"), ("gate_2.5", "2.5"), ("gate_3.0", "3.0")]
    l1 = "Anchor score threshold & " + " & ".join(x for _, x in thr) + r" \\"
    l2 = "AP@0.7 & " + " & ".join(f2(s["?|" + k]) for k, _ in thr) + r" \\"
    l3 = "Association gate (m) & " + " & ".join(x for _, x in gate) + r" & & \\"
    l4 = "AP@0.7 & " + " & ".join(f2(s["?|" + k]) for k, _ in gate) + r" & & \\"
    write("sensitivity.tex", "\n".join([l1, r"\midrule", l2, r"\midrule", l3, r"\midrule", l4]))


def robust():
    r = N["robust_external_val21"]
    L = []
    for mod, name in (("freealign", "FreeAlign"), ("roco", "RoCo"), ("cbm", "CBM")):
        cells = [name]
        for flt in ("instant", "median", "gated_KF"):
            cells.append(f2(r[f"{mod}_{flt}"]["AP70"]["mean"]))
        L.append(" & ".join(cells) + r" \\")
    write("robust.tex", "\n".join(L))


def coalign():
    c = N["coalign_R3_common"]
    L = []
    for k, name in (("GRAPH", "CoAlign pose graph~\\cite{lu2023coalign}"), ("K1", "\\method, confident anchors only"), ("H1", "\\textbf{CoMET}")):
        cells = [name] + [f2(c[k]["AP70"][w]) for w in W] + [f2(c[k]["AP70"]["mean"]), f2(c[k]["AP50"]["mean"])]
        L.append(" & ".join(cells) + r" \\")
    write("coalign.tex", "\n".join(L))


def census():
    L = []
    for role, name in (("fit24", "fit (24 seq.)"), ("calibration8", "calibration (8)"), ("mechanism_holdout14", "held-out (14)"), ("val21", "validation (21)")):
        c = N["census"][role]; s = N["saturation"][role]; ct = c["contingency"]
        zc = ct.get("zero_coop", 0)
        L.append(" & ".join([name, str(c["messages"]), f"{c['zero']} ({100*c['zero']/c['messages']:.1f}\\%)", str(zc), str(ct.get("nonzero_no_coop", 0)), f"{s['saturated']} ({100*s['fraction']:.1f}\\%)"]) + r" \\")
    write("census.tex", "\n".join(L))


def population():
    names = dict(author="TraF-Align (released)", strict60="TraF-Align (strict)", v2xvit_seed2024="V2X-ViT A", v2xvit_seed2025="V2X-ViT B", where2comm="Where2comm")
    L = []
    for d in names:
        P = N["population_split"][d]
        cells = [names[d]]
        for comp in ("H1_minus_PUBLIC", "H1_minus_single_mean"):
            for pop in ("all", "colabelled", "zero"):
                cells.append(sgn(P[f"val21|{comp}|{pop}|70"]["delta"]))
        L.append(" & ".join(cells) + r" \\")
    write("population.tex", "\n".join(L))


def aux_gt():
    f = {(r["domain"], r["role"], r["comparison"]): r for r in N["R6f_summary"] if r["population"] == "all" and r["threshold"] == 70 and r["wait_ms"] == "four_wait_mean"}
    d = {(r["role"], r["comparison"]): r for r in N["R6d_author"] if r["tier"] == "all_gaps" and r["threshold"] == 70}
    L = []
    for role, rn in (("val21", "validation"), ("mechanism_holdout14", "held-out")):
        for comp, cn in (("H1_minus_PUBLIC", "CoMET $-$ online pose"), ("H1_minus_single_mean", "CoMET $-$ current message")):
            cells = [rn, cn]
            e = d[(role, comp)]
            cells.append(f"{sgn(e['delta_pp'])} {ci(*e['CI95'])}")
            for dom in ("author", "strict60"):
                e = f.get((dom, role, comp))
                cells.append(f"{sgn(e['delta_pp'])} {ci(*e['CI95'])}" if e else "--")
            L.append(" & ".join(cells) + r" \\")
    write("aux_gt.tex", "\n".join(L))


def tier2():
    A = N["R7b_absolute"]; C = N["R7b_contrasts_A"]
    rows = (("Online pose, original labels", "PUBLIC_finetuned"), ("\\method pose (confident anchors), original labels", "original_K4"), ("\\method pose (confident anchors), cleaned labels", "clean_K1"), ("\\textbf{\\method pose, cleaned labels (second tier)}", None), (None, None), ("\\method without retraining (first tier)", "H1_zero_training"))
    def val(role, th, ref):
        base = A[f"A|{role}|all|{th}|four_wait_mean"]
        return base if ref is None else base - C[f"{role}|all|{th}|{ref}|four_wait_mean"]["delta"]
    for name_file, roles in (("tier2_abs.tex", ("val21",)), ("tier2_heldout.tex", ("mechanism_holdout14",))):
        L = []
        for name, ref in rows:
            if name is None:
                L.append(r"\midrule"); continue
            cells = [name] + [f2(val(role, th, ref)) for role in roles for th in ("70", "50")]
            L.append(" & ".join(cells) + r" \\")
        write(name_file, "\n".join(L))
    L = []
    for ref, name in (("PUBLIC_finetuned", "retraining with the online pose"), ("H1_zero_training", "\\method without retraining (first tier)")):
        cells = [name]
        for th in ("70", "50"):
            e = C[f"val21|all|{th}|{ref}|four_wait_mean"]
            cells.append(f"{sgn(e['delta'])} {ci(*e['ci'])}")
        L.append(" & ".join(cells) + r" \\")
    write("tier2_ci.tex", "\n".join(L))
    R = N["R7c_absolute"]; RC = N["R7c_contrasts"]
    L = []
    for arch, name, t1 in (("where", "Where2comm", "where2comm"), ("coalign", "CoAlign", None)):
        cells = [name]
        for m in ("PUBLIC", "old_ONLINE", "clean_H1"):
            cells.append(f"{R[f'{arch}|{m}|all|70|four_wait_mean']:.2f} / {R[f'{arch}|{m}|all|50|four_wait_mean']:.2f}")
        e70 = RC[f"{arch}|clean_H1_minus_PUBLIC|all|70|four_wait_mean"]; e50 = RC[f"{arch}|clean_H1_minus_PUBLIC|all|50|four_wait_mean"]
        cells.append(f"{sgn(e70['delta'])} {ci(*e70['ci'])}")
        cells.append(f"{sgn(e50['delta'])} {ci(*e50['ci'])}")
        L.append(" & ".join(cells) + r" \\")
    write("tier2_cross.tex", "\n".join(L))


def opv():
    o = N["opv_boundary"]
    L = []
    for k, name in (("SPD_structured", "structured, fitted to V2X-Seq"), ("iid_02m_02deg", "i.i.d.\\ per frame (0.2\\,m, 0.2$^\\circ$)"), ("ego_pose_time_050ms", "ego pose time error 50\\,ms"), ("ego_pose_time_100ms", "ego pose time error 100\\,ms"), ("ego_pose_time_150ms", "ego pose time error 150\\,ms")):
        cells = [name]
        for w in ("0", "400"):
            for m in ("stale", "H1", "oracle"):
                cells.append(f2(o[f"{k}|{m}|{w}|AP70"]))
        L.append(" & ".join(cells) + r" \\")
    write("opv.tex", "\n".join(L))
    v = N["v2v4real_boundary"]
    L = []
    for role, name in (("val21", "validation"), ("mechanism_holdout14", "held-out")):
        cells = [name]
        for w in ("0", "400"):
            for m in ("stale", "H1"):
                cells.append(f2(v[f"{role}|{m}|{w}|ap70"]))
        L.append(" & ".join(cells) + r" \\")
    write("v2v4real.tex", "\n".join(L))


def appendix_tables():
    an = N["anchors_author"]
    L = []
    for role, name in (("fit24", "fit"), ("calibration8", "calibration"), ("mechanism_holdout14", "held-out"), ("val21", "validation")):
        cells = [name]
        for w in W:
            a = an[f"{role}|{w}"]
            cells.append(f"{a['mean_anchors']:.2f} / {100*a['no_anchor_fraction']:.1f}\\%")
        L.append(" & ".join(cells) + r" \\")
    write("anchors.tex", "\n".join(L))
    # R6f full by population
    f = {(r["domain"], r["role"], r["population"], r["comparison"]): r for r in N["R6f_summary"] if r["threshold"] == 70 and r["wait_ms"] == "four_wait_mean"}
    L = []
    for dom, dn in (("author", "TraF-Align (released)"), ("strict60", "TraF-Align (strict)")):
        for role, rn in (("val21", "val."), ("mechanism_holdout14", "held-out")):
            cells = [dn, rn]
            for comp in ("H1_minus_PUBLIC", "H1_minus_single_mean"):
                for pop in ("all", "colabelled", "zero"):
                    e = f[(dom, role, pop, comp)]
                    cells.append(sgn(e["delta_pp"]) + ("" if e["CI95"][0] > 0 or e["CI95"][1] < 0 else "$^\\circ$"))
            L.append(" & ".join(cells) + r" \\")
    write("aux_full.tex", "\n".join(L))
    d = {(r["role"], r["tier"], r["comparison"]): r for r in N["R6d_author"] if r["threshold"] == 70}
    L = []
    for tier, tn in (("gap1s", "gaps $\\le$1\\,s"), ("gap3s", "gaps $\\le$3\\,s"), ("all_gaps", "all gaps")):
        cells = [tn]
        for role in ("val21", "mechanism_holdout14"):
            for comp in ("H1_minus_PUBLIC", "H1_minus_single_mean"):
                e = d[(role, tier, comp)]
                cells.append(f"{sgn(e['delta_pp'])} {ci(*e['CI95'])}")
        L.append(" & ".join(cells) + r" \\")
    write("aux_interp.tex", "\n".join(L))
    B = N["R7b_contrasts_B"]; A = N["R7b_contrasts_A"]
    L = []
    for ref, name in (("PUBLIC_finetuned", "online-pose retraining"),):
        for tier, TT, label in (("A", A, "Interpolated labels (selected)"), ("B", B, "Interpolated and bound-corrected labels")):
            cells = [f"{label} $-$ {name}"]
            for role in ("val21", "mechanism_holdout14"):
                for th in ("70", "50"):
                    e = TT[f"{role}|all|{th}|{ref}|four_wait_mean"]
                    cells.append(f"{sgn(e['delta'])} {ci(*e['ci'])}")
            L.append(" & ".join(cells) + r" \\")
    write("tier_ab.tex", "\n".join(L))
    h = [r for r in N["R6h_contrasts"] if r["wait_ms"] == "four_wait_mean" and r["population"] == "all"]
    hh = {(r["role"], r["tier"], r["comparison"], r["threshold"]): r for r in h}
    L = []
    for tier, tn in (("all_gaps", "interpolated"), ("label_median", "label-median")):
        for comp, cn in (("A_minus_PUBLIC", "second tier $-$ online-pose retraining"),):
            cells = [tn, cn]
            for role in ("val21", "mechanism_holdout14"):
                for th in (70, 50):
                    e = hh[(role, tier, comp, th)]
                    cells.append(f"{sgn(e['delta_pp'])} {ci(*e['CI95'])}")
            L.append(" & ".join(cells) + r" \\")
    write("tier2_aux.tex", "\n".join(L))


def combination():
    c = N["combination"]
    cols = [("val21", "70"), ("val21", "50"), ("mechanism_holdout14", "70"), ("mechanism_holdout14", "50")]
    L = []
    L.append(" & ".join(["\\method late fusion (first tier)"] + [f2(c[f"{r}|{th}|tier1"]["baseline"]) for r, th in cols]) + r" \\")
    L.append(" & ".join(["Retrained TraF-Align (second tier)"] + [f2(c[f"{r}|{th}|tier2"]["baseline"]) for r, th in cols]) + r" \\")
    L.append(" & ".join(["\\textbf{Two-tier combination}"] + [r"\textbf{" + f2(c[f"{r}|{th}|tier1"]["AP"]) + "}" for r, th in cols]) + r" \\")
    L.append(r"\midrule")
    for ref, name in (("tier1", "Combination $-$ first tier"), ("tier2", "Combination $-$ second tier")):
        L.append(" & ".join([name] + [f"{sgn(c[f'{r}|{th}|{ref}']['delta'])} {ci(*c[f'{r}|{th}|{ref}']['CI'])}" for r, th in cols]) + r" \\")
    write("combination.tex", "\n".join(L))


def acquisition():
    s = N["stress_late"]
    names = dict(author="TraF-Align (released)", strict60="TraF-Align (strict)", v2xvit_seed2024="V2X-ViT A", v2xvit_seed2025="V2X-ViT B", where2comm="Where2comm")
    L = []
    for d, amps in (("author", (3, 5)), ("strict60", (3, 5)), ("v2xvit_seed2024", (3,)), ("v2xvit_seed2025", (3,)), ("where2comm", (3,))):
        for a in amps:
            k = lambda m: s[f"{d}|val21|T1|{a}|{m}|70"]
            e = k("CoMET_A")
            L.append(" & ".join([names[d] if a == amps[0] else "", f"{a}\\,m", f2(k("PUBLIC")["AP"]), f2(k("H1")["AP"]),
                                 r"\textbf{" + f2(e["AP"]) + "}", f"{sgn(e['delta_H1'])} {ci(*e['CI'])}", f"{100 * e['retained']:.1f}\\%"]) + r" \\")
    write("acquisition.tex", "\n".join(L))


def budget():
    b = N["budget"]
    L = []
    for u in ("2000", "8000", "20000"):
        cells = [f"{int(u):,}" + (" (selected)" if u == "2000" else ""), f2(b[f"{u}|calibration8|A|70"]), f2(b[f"{u}|val21|A|70"]),
                 f2(b[f"{u}|val21|A|50"]), f2(b[f"{u}|mechanism_holdout14|A|70"]), f2(b[f"{u}|val21|PUBLIC|70"])]
        L.append(" & ".join(cells) + r" \\")
    write("budget.tex", "\n".join(L))


if __name__ == "__main__":
    for fn in (main_table, cross_detector, temporal, ablation, sensitivity, robust, coalign, census, population, aux_gt, tier2, opv, appendix_tables, combination, acquisition, budget):
        fn()
    print("tables written to", OUT)
