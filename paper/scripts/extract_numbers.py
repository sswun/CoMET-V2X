"""Collect every number used in the manuscript from the reported result files in ../../results.

Writes paper/numbers.json. Tables in paper/tables/*.tex are generated from it by make_tables.py.
Only reported results are read; nothing is recomputed except means over waits, heads or seeds.
"""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parent / "results"
T3 = RES / "tables_v3"
WAITS = ("0", "100", "200", "400")
DOMAINS = ("author", "strict60", "v2xvit_seed2024", "v2xvit_seed2025", "where2comm")


def rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def jload(path):
    p = Path(path)
    if p.suffix == ".gz":
        return json.load(gzip.open(p, "rt"))
    return json.load(open(p))


def fields(row):
    return row.split("|")


def panel_means(path, role, name_filter, key_index, metrics=("value__ap50", "value__ap70"), domain=None):
    """Average AP over head/seed variants (and detectors, unless ``domain`` is given) per method and wait."""
    acc = defaultdict(list)
    for r in rows(path):
        if r["metric"] not in metrics:
            continue
        if domain is not None and f"/{domain}/" not in r["row"]:
            continue
        f = fields(r["row"])
        if role not in f:
            continue
        name = f[key_index]
        if not name_filter(name):
            continue
        wait = f[-2] if f[-1] == "" else f[-1]
        method = name.rsplit("_seed", 1)[0]
        acc[(method, wait, r["metric"][-4:])].append(float(r["value"]))
    out = defaultdict(dict)
    for (m, w, th), v in acc.items():
        out[m].setdefault(th, {})[w] = float(np.mean(v))
    for m in out:
        for th in out[m]:
            out[m][th]["mean"] = float(np.mean([out[m][th][w] for w in WAITS if w in out[m][th]]))
    return {k: dict(v) for k, v in out.items()}


def main():
    N = {}
    # ---------------- main table baselines (table1) ----------------
    t1 = defaultdict(lambda: defaultdict(dict))
    for r in rows(T3 / "table1.csv"):
        name, wait = r["row"].rsplit("|", 1)
        metric = r["metric"].replace("_mean", "")
        if metric in ("AP50", "AP70"):
            t1[name][metric][wait] = float(r["value"])
    for name in t1:
        for m in t1[name]:
            t1[name][m]["mean"] = float(np.mean([t1[name][m][w] for w in WAITS]))
    N["table1"] = {k: dict(v) for k, v in t1.items()}

    # ---------------- cross-detector CoMET absolute (three heads) ----------------
    cd = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows(T3 / "H1_cross_detector_summary.csv"):
        dom, role, wait = r["row"].split("|")
        if r["metric"] in ("AP50_mean", "AP70_mean"):
            cd[dom][role][r["metric"][:4]][wait] = float(r["value"])
    for dom in cd:
        for role in cd[dom]:
            for m in cd[dom][role]:
                cd[dom][role][m]["mean"] = float(np.mean([cd[dom][role][m][w] for w in WAITS]))
    # The author detector's val21 rows live in table1 (same three-head recipe).
    cd["author"]["val21"]["AP50"] = dict(N["table1"]["author|val21"]["AP50"])
    cd["author"]["val21"]["AP70"] = dict(N["table1"]["author|val21"]["AP70"])
    N["comet_cross_detector"] = json.loads(json.dumps(cd))

    # ---------------- A1: EGO / PUBLIC / CoMET / OFFICIAL / MA5 absolute (author) ----------------
    a1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows(T3 / "H1_A1_retention.csv"):
        if not r["metric"].startswith("value__AP__"):
            continue
        f = fields(r["row"])
        if f[4] != "PUBLIC":
            continue
        meth = r["metric"].split("__")[-1]
        wait, th = f[-2], f[-1]
        a1[meth]["AP" + th][wait].append(float(r["value"]))
    N["author_absolute_A1"] = {m: {th: {w: float(np.mean(v)) for w, v in d.items()} for th, d in a1[m].items()} for m in a1}

    # ---------------- temporal aggregation (author, three heads) ----------------
    N["temporal"] = {}
    for role in ("val21", "mechanism_holdout14"):
        N["temporal"][role] = panel_means(T3 / "H1_temporal_aggregation.csv", role, lambda n: "_seed" in n, 4)

    # ---------------- learned vs untrained ablation (author) ----------------
    # Diagnostic ablation: original per-detector q and tau, three frozen heads; mean over the five detectors.
    N["ablation"] = {}
    N["ablation_author"] = {}
    for role in ("calibration8", "val21", "mechanism_holdout14"):
        N["ablation"][role] = panel_means(T3 / "learning_untrained_H1_ablation.csv.gz", role, lambda n: "_seed" in n, 4)
        N["ablation_author"][role] = panel_means(T3 / "learning_untrained_H1_ablation.csv.gz", role, lambda n: "_seed" in n, 4, domain="author")
    cnt = defaultdict(int)
    for r in rows(T3 / "learning_untrained_H1_ablation.csv.gz"):
        if r["metric"] == "value__ap70" and "|val21|H1_seed" in r["row"]:
            cnt[r["row"].split("/recipe_v3_interface_recovery_v2/")[1].split("/")[0]] += 1
    N["ablation_domains"] = dict(cnt)

    # ---------------- sensitivity (score threshold / gate), author val21 ----------------
    sens = defaultdict(list)
    for r in rows(T3 / "H1_sensitivity.csv.gz"):
        if r["metric"] != "value__ap70":
            continue
        f = fields(r["row"])
        name = next((x for x in f if x.startswith(("score_", "gate_"))), None)
        if not name or "author" not in r["row"]:
            continue
        sens[(f[3] or "?", name.rsplit("_frozen", 1)[0])].append(float(r["value"]))
    N["sensitivity_author_AP70"] = {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in sens.items()}

    # ---------------- robust external (R4), val21 ----------------
    rob = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows(T3 / "R4_independent_read/AP.csv"):
        v = json.loads(r["values"])
        if v.get("role") != "val21":
            continue
        key = v["module"] + "_" + v["filter"]
        rob[key]["AP70"][str(v["wait_ms"])].append(v["AP70"])
        rob[key]["AP50"][str(v["wait_ms"])].append(v["AP50"])
    R4 = {}
    for k in rob:
        R4[k] = {}
        for th in rob[k]:
            d = {w: float(np.mean(x)) for w, x in rob[k][th].items()}
            d["mean"] = float(np.mean([d[w] for w in WAITS]))
            R4[k][th] = d
    N["robust_external_val21"] = R4

    # ---------------- CoAlign R3 ----------------
    co = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows(T3 / "CoAlign_R3_independent/AP.csv"):
        v = json.loads(r["values"])
        if v["roi"] != "common":
            continue
        co[v["method"]]["AP%d" % v["threshold"]][str(v["wait_ms"])].append(v["AP"])
    N["coalign_R3_common"] = {m: {th: dict({w: float(np.mean(x)) for w, x in d.items()}, mean=float(np.mean([np.mean(d[w]) for w in WAITS])), seeds=len(d["0"])) for th, d in co[m].items()} for m in co}

    # ---------------- anchors ----------------
    an = defaultdict(dict)
    for r in rows(T3 / "H1_anchor_statistics.csv"):
        dom, role, wait = r["row"].split("|")
        if dom == "author" and r["metric"] in ("mean_anchors", "no_anchor_fraction", "mean_anchors_given_observed"):
            an[f"{role}|{wait}"][r["metric"]] = float(r["value"])
    N["anchors_author"] = an

    # ---------------- source AP (author CoMET, three heads) ----------------
    src = defaultdict(list)
    for r in rows(T3 / "H1_source_AP.csv"):
        if r["metric"] != "value__AP":
            continue
        f = fields(r["row"])
        src[(f[10], f[-2], f[-1])].append(float(r["value"]))
    N["source_AP_author"] = {"|".join(k): float(np.mean(v)) for k, v in src.items()}

    # ---------------- A3 external/single-message estimator error vs reference ----------------
    a3 = {}
    for r in rows(T3 / "H1_A3.csv"):
        f = fields(r["row"])
        if r["metric"] in ("value__metrics__radial_RMSE_m__value", "value__metrics__failure_gt1m_fraction__value", "value__metrics__demeaned_lag1_correlation__value"):
            a3[f"{f[4]}|{f[11]}|{f[-2]}|{r['metric'].split('__')[2]}"] = float(r["value"])
    N["A3"] = a3

    # ---------------- boundaries ----------------
    opv = defaultdict(list)
    for r in rows(T3 / "H1_opv_boundary.csv"):
        if r["metric"] not in ("value__AP70", "value__AP50"):
            continue
        f = fields(r["row"])
        opv[(f[6], f[4].rsplit("_seed", 1)[0], f[-2], r["metric"][-4:])].append(float(r["value"]))
    N["opv_boundary"] = {"|".join(k): float(np.mean(v)) for k, v in opv.items()}
    v2 = defaultdict(list)
    for r in rows(T3 / "H1_v2_boundary.csv"):
        if r["metric"] not in ("value__ap70", "value__ap50"):
            continue
        f = fields(r["row"])
        v2[(f[3], f[4].rsplit("_seed", 1)[0], f[-2], r["metric"][-4:])].append(float(r["value"]))
    N["v2v4real_boundary"] = {"|".join(k): float(np.mean(v)) for k, v in v2.items()}

    # ---------------- population split contrasts, five detectors ----------------
    pop = {}
    for dom in DOMAINS:
        d = jload(RES / "population_split" / dom / "contrasts.json.gz")
        acc = defaultdict(list)
        for r in d:
            if r["wait_ms"] != "four_wait_mean":
                continue
            acc[(r["role"], r["comparison"], r["population"], r["threshold"])].append((r["delta_pp"], r["CI95"][0], r["CI95"][1]))
        pop[dom] = {"|".join(map(str, k)): dict(delta=float(np.mean([x[0] for x in v])), ci_low_min=float(min(x[1] for x in v)), ci_high_max=float(max(x[2] for x in v)), heads=len(v)) for k, v in acc.items()}
    N["population_split"] = pop

    # ---------------- dataset audit ----------------
    sat = rows(RES / "dataset_audit/offset_grid/saturation.csv")
    N["saturation"] = {r["role"]: dict(frames=int(r["frames"]), nonzero=int(r["nonzero"]), saturated=int(r["saturated"]), fraction=float(r["fraction_of_nonzero"])) for r in sat}
    N["bound_displacement"] = rows(RES / "dataset_audit/offset_grid/boundary.csv")
    census = jload(RES / "dataset_audit/R6a_raw_label_audit/summary.json")["census"]
    N["census"] = {role: dict(messages=c["messages"], zero=c["zero"], contingency=c.get("contingency")) for role, c in census.items()}

    # ---------------- auxiliary GT ----------------
    N["R6d_author"] = jload(RES / "aux_gt/R6d_all_gaps/author_paired_summary.json")
    N["R6f_summary"] = jload(RES / "aux_gt/R6f_label_median/summary.json")
    N["R6h_contrasts"] = jload(RES / "aux_gt/R6h_tier2/contrasts.json")

    # ---------------- second tier ----------------
    both = jload(RES / "tier2/R7b_TraF/contrasts_both_tiers.json")
    N["R7b_contrasts_A"] = {f"{r['role']}|{r['population']}|{r['threshold']}|{r['reference']}|{r['wait_ms']}": dict(delta=r["delta_pp"], ci=r["CI95"]) for r in both if r["tier"] == "A"}
    N["R7b_contrasts_B"] = {f"{r['role']}|{r['population']}|{r['threshold']}|{r['reference']}|{r['wait_ms']}": dict(delta=r["delta_pp"], ci=r["CI95"]) for r in both if r["tier"] == "B"}
    per = defaultdict(list)
    for p in sorted((RES / "tier2/R7b_TraF/per_seed").glob("*.json")):
        name = p.stem
        if name.endswith("_independent"):
            continue
        tier, seed, role = name.split("_", 2)
        for r in jload(p):
            per[(tier, role, r["population"], r["threshold"], str(r["wait_ms"]))].append(r["AP"])
    N["R7b_absolute"] = {"|".join(map(str, k)): float(np.mean(v)) for k, v in per.items()}
    N["R7b_decision"] = jload(RES / "tier2/R7b_TraF/decision.json")["checks"]
    N["R7b_calibration"] = jload(RES / "tier2/R7b_TraF/calibration_choice.json")["means"]
    c7 = jload(RES / "tier2/R7c_cross_architecture/contrasts.json")
    N["R7c_contrasts"] = {f"{r['architecture']}|{r['comparison']}|{r['population']}|{r['threshold']}|{r['wait_ms']}": dict(delta=r["delta_pp"], ci=r["CI95"]) for r in c7}
    per = defaultdict(list)
    for p in sorted((RES / "tier2/R7c_cross_architecture/per_seed").glob("*.json")):
        if p.stem.endswith("_independent"):
            continue
        arch, seed, meth = p.stem.split("_", 2)
        for r in jload(p):
            per[(arch, meth, r["population"], r["threshold"], str(r["wait_ms"]))].append(r["AP"])
    N["R7c_absolute"] = {"|".join(map(str, k)): float(np.mean(v)) for k, v in per.items()}
    N["R6e"] = jload(RES / "ablations/R6e_readout_head_decontamination/decision.json")["means"]
    N["verification"] = jload(RES / "verification/verify_against_recipe.json")["_all_equal"]

    # ---------------- second tier: training-budget sweep ----------------
    bud = {}
    for r in jload(RES / "tier2/R7d_TraF_budget/panels.json"):
        if r["population"] == "all" and r["wait_ms"] == "four_wait_mean":
            bud[f"{r['budget']}|{r['role']}|{r['method']}|{r['threshold']}"] = r["AP"]
    cal = defaultdict(list)
    for r in rows(RES / "tier2/tables/R7d_A_calibration.csv.gz"):
        if r["population"] == "all" and r["threshold"] == "70" and r["wait_ms"] == "four_wait_mean" and r["seed"]:
            cal[r["method"].replace("A_budget", "")].append(float(r["AP"]))
    for b, v in cal.items():
        bud[f"{b}|calibration8|A|70"] = float(np.mean(v))
    N["budget"] = bud

    # ---------------- lock acquisition (pre-registered extension) ----------------
    V2 = RES / "research_v2"
    acq = {}
    for r in rows(V2 / "R8d_acquisition/results.csv"):
        key = f"{r['domain']}|{r['role']}|{r['kind']}|{float(r['amplitude']):g}|{r['method']}|{r['threshold']}"
        acq[key] = dict(AP=float(r["AP"]), delta_H1=float(r["delta_H1"]) if r["delta_H1"] else None,
                        CI=json.loads(r["delta_H1_CI95"]) if r["delta_H1_CI95"] else None,
                        retained=float(r["ratio_clean_H1"]) if r["ratio_clean_H1"] else None)
    N["acquisition"] = acq
    N["acquisition_gate"] = jload(V2 / "R8d_acquisition/adoption_gate.json")["checks"]

    # ---------------- pose-error stress test ----------------
    st = {}
    for d in DOMAINS:
        for r in rows(V2 / "R8c_stress/late" / d / "results.csv"):
            key = f"{d}|{r['role']}|{r['kind']}|{float(r['amplitude']):g}|{r['method']}|{r['threshold']}"
            st[key] = dict(AP=float(r["AP"]), delta_H1=float(r["delta_H1"]) if r["delta_H1"] else None,
                           CI=json.loads(r["delta_H1_CI95"]) if r["delta_H1_CI95"] else None,
                           retained=float(r["ratio_clean_H1"]) if r["ratio_clean_H1"] else None)
    N["stress_late"] = st
    N["stress_native"] = {f"{r['model']}|{float(r['amplitude_m']):g}|{r['threshold']}": dict(
        uncorrected=float(r["uncorrected_AP"]), corrected=float(r["corrected_AP"]), delta=float(r["delta"]),
        CI=json.loads(r["paired_CI95"])) for r in rows(V2 / "R8c_stress/native/results.csv")}
    rec = defaultdict(lambda: [0, 0])
    for r in rows(V2 / "R8c_stress/capture/T3_recovery_all_methods.csv.gz"):
        k = f"{r['domain']}|{r['role']}|{r['method']}"
        rec[k][1] += 1
        rec[k][0] += int(r["right_censored"] in ("False", "false", "0"))
    N["step_recovery"] = {k: dict(recovered=v[0], streams=v[1]) for k, v in rec.items()}

    # ---------------- two-tier combination ----------------
    N["combination"] = {f"{r['role']}|{r['threshold']}|{r['reference']}": dict(
        AP=float(r["AP"]), baseline=float(r["baseline_AP"]), delta=float(r["delta"]), CI=json.loads(r["paired_CI95"]))
        for r in rows(V2 / "R8e_two_tier/results.csv")}
    N["combination_coefficients"] = jload(V2 / "R8e_two_tier/coefficients.json")["coefficients"]
    N["combination_choice"] = jload(V2 / "R8e_two_tier/calibration_choice.json")["cal_AP70"]
    N["verification_extensions"] = jload(RES / "verification/verify_research_v2.json")["_all_equal"]

    (HERE / "numbers.json").write_text(json.dumps(N, indent=1))
    print("wrote", HERE / "numbers.json")


if __name__ == "__main__":
    main()
