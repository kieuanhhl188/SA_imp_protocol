"""
analyze_symbol_signal.py — phan tich Phase 3 (V1) SymbolSignal, giao thuc thong ke Y HET cac phan tich C2 truoc:
ghep cap theo MAU (gop 3 lop), bootstrap B=20000 seed=0, KTC percentile 95%, "co y nghia" <=> KTC loai 0.

Buoc 1: KIEM TOAN VEN. Neu bat ky kiem tra nao FAIL -> ghi integrity.json, in FAIL, DUNG (khong ra so).
Buoc 2: phu song ky hieu, bang chinh, chan doan recall-vs-mass, phan ra bac K*, muc mau, clean-191, tieu chi C2.
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

W = "/workspace"
OUT = f"{W}/phase5_repobench_symbolsignal_n200"
SC = "/tmp/claude-0/-workspace-SA-imp-protocol/d3a5a08b-8399-49d7-ab98-49df3d5f9753/scratchpad"
REPO = "/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
SA = "/workspace/p2-longchat-repobench/sa/repobench-p"
TRUNC = [34, 46, 59, 70, 75, 90, 154, 161, 170]
D6 = [21, 102]
SP = [70, 80, 90]
LAMS = ["0.0", "0.5", "1.0", "2.0"]
NL, B, SEED = 3, 20000, 0

R = json.load(open(f"{OUT}/results.json"))
cov = json.load(open(f"{OUT}/symbol_coverage_raw.json"))
ps = R["per_sample"]


def lay(v): return np.asarray(v, float).reshape(-1, NL)
def pool(v): return lay(v).mean(1)


# ============================================================ 1. INTEGRITY
integ = {}
def check(name, ok, detail=""):
    integ[name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


print("== INTEGRITY CHECKS")
old = json.load(open(f"{W}/phase5_repobench_class_n200.json"))
oldm = json.load(open(f"{W}/phase5_repobench_class_tw_n200.json"))
diag = json.load(open(f"{W}/diag_hb_vs_sa/rank_tiers_n200.json"))
exact = True; det = []
for sp in SP:
    a = ps["0.0"][str(sp)]["recall"]; b = old["per_sample"]["sa"][str(sp)]
    am = ps["0.0"][str(sp)]["mass"]; bm = oldm["per_sample_mass"]["sa"][str(sp)]
    e1, e2 = (a == b), (am == bm)
    exact &= e1 and e2
    det.append(f"sp{sp}: recall {'==' if e1 else '!='} mass {'==' if e2 else '!='} (n={len(a)})")
check("1. lambda=0 bit-exact vs existing SA C2 per-sample recall & mass", exact, "; ".join(det))
agg = {sp: (float(np.mean(ps['0.0'][str(sp)]['recall'])) * 100, float(np.mean(ps['0.0'][str(sp)]['mass'])) * 100) for sp in SP}
exp = {70: (78.24, 98.02), 80: (74.46, 97.28), 90: (69.00, 95.96)}
check("1b. SA aggregate == expected", all(abs(agg[sp][0] - exp[sp][0]) < 0.006 and abs(agg[sp][1] - exp[sp][1]) < 0.006 for sp in SP),
      ", ".join(f"sp{sp} {agg[sp][0]:.2f}/{agg[sp][1]:.2f}" for sp in SP))
before = open(f"{SC}/sym_sa_manifest_before.txt").read()
after = subprocess.run(f"find {SA} -type f -printf '%s %T@ %p\\n' | sort -k3", shell=True, capture_output=True, text=True).stdout
check("2a. SA files unchanged (600 files, size+mtime manifest)", before == after and len(after.splitlines()) == 600)
shab = open(f"{SC}/sym_sa_sha_before.txt").read()
shaa = subprocess.run(f"sha256sum {SA}/centroids_tensor_dict_0_819.pt {SA}/centroids_labels_dict_65_1030.pt {SA}/centroids_tensor_dict_120_861.pt {SA}/global_threshold_199_*.pt",
                      shell=True, capture_output=True, text=True).stdout
check("2b. SA sha256 (4 files) unchanged", shab == shaa)
inrun = json.load(open(f"{OUT}/integrity_inrun.json"))["in_run_checks"]
check("3. #retrieved keys == N (in-run assert every row/lambda/sp/layer/sample)", "PASS" in inrun["n_selected_keys_equals_N"], inrun["n_selected_keys_equals_N"])
check("3b. lambda=0 == original recall_one_sample every sample x layer x sp (in-run)", "PASS" in inrun["lambda0_bit_exact_vs_recall_one_sample"])
check("3c. determinism re-check (lambda>0 run twice, identical)", inrun["determinism_recheck_samples"] >= 4, f"{inrun['determinism_recheck_samples']} samples")
ss_ok = True
for sp in SP:
    for t in "ABC":
        base = ps["0.0"][str(sp)][f"star_share_{t}"]
        ss_ok &= all(ps[l][str(sp)][f"star_share_{t}"] == base for l in LAMS)
        ss_ok &= bool(np.abs(lay(base) - lay(diag["branches"]["sa"][str(sp)][f"star_share_{t}"])).max() < 1e-6)
check("4. ideal K* identical across lambdas and == earlier diagnostic (star mass shares)", ss_ok)
check("5. same samples/layers/queries", R["idx"] == list(range(200)) and R["layers"] == "first,mid,last" and len(ps["0.0"]["70"]["recall"]) == 600 and R["sparsity"] == SP)
src = open(f"{REPO}/phase5_symbol_signal.py").read() + open(f"{REPO}/symbol_signal.py").read()
bad = [w for w in ("from struct_clustering", "import struct_clustering", "hard_boundary_kmeans", "struct_hierarchy_l1", "build_l1_groups", "hierarchical") if w in src]
check("6/7. no HardBoundary / StructHierarchy code path", not bad and os.path.normpath(R["sa_dir"]).split(os.sep)[-2] == "sa", f"forbidden tokens found: {bad}" if bad else "sa_dir=SA only")
check("8. no generation", ".generate(" not in src)
now = subprocess.run(f"ls {W}/phase5_*.json | xargs stat -c '%s %Y %n'", shell=True, capture_output=True, text=True).stdout
check("9. existing phase5_*.json result files unmodified", now == open(f"{SC}/sym_existing_results_before.txt").read() and len(now.splitlines()) == 18)
check("10. new results in separate directory", OUT.endswith("phase5_repobench_symbolsignal_n200"))
json.dump(integ, open(f"{OUT}/integrity.json", "w"), indent=1)
if not all(v["pass"] for v in integ.values()):
    print("\n*** INTEGRITY FAILURE — stopping, no results produced ***"); sys.exit(1)
print("== all integrity checks PASSED\n")

# ============================================================ data
rng = np.random.default_rng(SEED)
idx = np.arange(200)
SUBS = {"ALL 200": np.ones(200, bool), "CLEAN 191": ~np.isin(idx, TRUNC),
        "D6 198 (excl. 21,102)": ~np.isin(idx, D6)}
REC = {l: {sp: lay(ps[l][str(sp)]["recall"]) for sp in SP} for l in LAMS}
MAS = {l: {sp: lay(ps[l][str(sp)]["mass"]) for sp in SP} for l in LAMS}


def ci(x):
    ii = rng.integers(0, len(x), size=(B, len(x)))
    return np.percentile(x[ii].mean(1), [2.5, 97.5])


def f_ci(x):
    lo, hi = ci(x)
    return f"{x.mean()*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}]{'*' if (lo > 0 or hi < 0) else ' '}", lo, hi


AN = {"main": {}, "sample": {}, "sensitivity": {}, "c2": {}}
DG = {"quadrants": {}, "tiers": {}}

# ============================================================ 2. symbol coverage
pc = cov["per_sample"]; pl = cov["per_sample_layer"]
def arr(k): return np.array([p[k] for p in pc], float)
SCV = {"n_samples": len(pc)}
for k in ("n_index_ids", "n_query_ids", "n_matched_ids", "n_matched_positions", "frac_clusters_with_hit"):
    a = arr(k); SCV[k] = dict(mean=float(a.mean()), median=float(np.median(a)), sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()))
SCV["pct_samples_with_match"] = float(np.mean(arr("has_match")) * 100)
SCV["pct_queries_with_match"] = SCV["pct_samples_with_match"]  # cua so quan sat: 100 query cua mau chia se cung tap dinh danh
SCV["by_language"] = {}
for lg in ("python", "java"):
    m = np.array([p["language"] == lg for p in pc])
    SCV["by_language"][lg] = dict(n=int(m.sum()), n_query_ids=float(arr("n_query_ids")[m].mean()), n_matched_ids=float(arr("n_matched_ids")[m].mean()),
                                  n_matched_positions=float(arr("n_matched_positions")[m].mean()), pct_match=float(arr("has_match")[m].mean() * 100))
sstd = np.array([p["S_SA_std_across_clusters"] for p in pl]); mh = np.array([p["mean_hit"] for p in pl]); mxh = np.array([p["max_hit"] for p in pl])
SCV["S_SA_std_across_clusters"] = dict(mean=float(sstd.mean()), median=float(np.median(sstd)))
SCV["symbol_hit"] = dict(mean=float(mh.mean()), max_mean=float(mxh.mean()), frac_clusters_hit_mean=float(np.mean([p["frac_clusters_with_hit"] for p in pl])))
SCV["lambda_x_maxhit_over_SA_std"] = {l: float(float(l) * mxh.mean() / sstd.mean()) for l in LAMS}
SCV["examples"] = [{k: p[k] for k in ("idx", "language", "n_query_ids", "n_matched_ids", "query_ids", "matched_ids")} for p in pc[:5]]
json.dump(SCV, open(f"{OUT}/symbol_coverage.json", "w"), indent=1)
print("== SYMBOL COVERAGE (200 samples, whole observation window = the 'query')")
print(f"  distinct query identifiers / sample: mean {SCV['n_query_ids']['mean']:.1f} (median {SCV['n_query_ids']['median']:.0f}, min {SCV['n_query_ids']['min']:.0f}, max {SCV['n_query_ids']['max']:.0f})")
print(f"  matched in fixed context / sample  : mean {SCV['n_matched_ids']['mean']:.1f} (median {SCV['n_matched_ids']['median']:.0f}, min {SCV['n_matched_ids']['min']:.0f});  samples with >=1 match: {SCV['pct_samples_with_match']:.1f}%  (== % of queries)")
print(f"  matched fixed-context token positions / sample: mean {SCV['n_matched_positions']['mean']:.0f} (median {SCV['n_matched_positions']['median']:.0f}) of n_ctx mean {arr('n_ctx').mean():.0f}")
for lg, v in SCV["by_language"].items():
    print(f"   {lg:6s} n={v['n']}: query ids {v['n_query_ids']:.1f}, matched {v['n_matched_ids']:.1f}, positions {v['n_matched_positions']:.0f}")
print(f"  symbol_hit: mean value {SCV['symbol_hit']['mean']:.4f}, mean max {SCV['symbol_hit']['max_mean']:.3f}, clusters with hit>0: {SCV['symbol_hit']['frac_clusters_hit_mean']*100:.1f}%")
print(f"  scale: S_SA std across clusters (per row) = {SCV['S_SA_std_across_clusters']['mean']:.2f};  lambda*max_hit / that std: " + ", ".join(f"lam{l}: {v:.3f}" for l, v in SCV['lambda_x_maxhit_over_SA_std'].items()))

# ============================================================ 3. main tables + sample-level + diagnostic
def part(name, mk):
    n = int(mk.sum())
    print(f"\n{'#'*110}\n# {name}   n={n}\n{'#'*110}")
    print("Method / lambda : Recall% / Attention-mass%   (precision == recall by construction: |K_m|=|K*|=N)")
    print(f"{'':22s}" + "".join(f"{'sp'+str(sp):>20s}" for sp in SP))
    for l in LAMS:
        lab = "SA (lambda=0)" if l == "0.0" else f"SA + SymbolSignal l={l}"
        print(f"{lab:22s}" + "".join(f"{REC[l][sp].mean(1)[mk].mean()*100:8.2f} / {MAS[l][sp].mean(1)[mk].mean()*100:7.2f} " for sp in SP))
    print("\nDelta vs SA (pp), paired 95% bootstrap CI  (* = CI excludes 0)")
    print(f"{'':10s}" + "".join(f"{'sp'+str(sp)+' dRecall':>31s}" for sp in SP) + "   |" + "".join(f"{'sp'+str(sp)+' dMass':>31s}" for sp in SP))
    for l in LAMS[1:]:
        cellsR, cellsM = [], []
        for sp in SP:
            dR = (REC[l][sp] - REC["0.0"][sp]).mean(1)[mk]; dM = (MAS[l][sp] - MAS["0.0"][sp]).mean(1)[mk]
            tR, loR, hiR = f_ci(dR); tM, loM, hiM = f_ci(dM)
            AN["main"][f"{name}|{l}|{sp}"] = dict(dRecall=float(dR.mean() * 100), ciR=[float(loR * 100), float(hiR * 100)], sigR=int(loR > 0) - int(hiR < 0),
                                                 dMass=float(dM.mean() * 100), ciM=[float(loM * 100), float(hiM * 100)], sigM=int(loM > 0) - int(hiM < 0))
            cellsR.append(f"{tR:>31s}"); cellsM.append(f"{tM:>31s}")
        print(f"lambda={l:4s}" + "".join(cellsR) + "   |" + "".join(cellsM))


for name, mk in SUBS.items():
    part(name, mk)

print(f"\n{'#'*110}\n# SAMPLE-LEVEL (ALL 200): paired dRecall per sample (3 layers pooled)\n{'#'*110}")
mk = SUBS["ALL 200"]
for l in LAMS[1:]:
    for sp in SP:
        d = (REC[l][sp] - REC["0.0"][sp]).mean(1)[mk]
        dm = (MAS[l][sp] - MAS["0.0"][sp]).mean(1)[mk]
        srt = np.sort(np.abs(d))[::-1]
        drop10 = d[np.argsort(-np.abs(d))[10:]].mean()
        try:
            wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        except ValueError:
            wp = float("nan")
        s = dict(mean=float(d.mean() * 100), median=float(np.median(d) * 100), sd=float(d.std(ddof=1) * 100), min=float(d.min() * 100), max=float(d.max() * 100),
                 pct_pos=float(np.mean(d > 0) * 100), pct_neg=float(np.mean(d < 0) * 100), pct_zero=float(np.mean(d == 0) * 100),
                 top10_abs_share=float(srt[:10].sum() / max(srt.sum(), 1e-12) * 100), mean_drop_top10_abs=float(drop10 * 100), wilcoxon_p=float(wp),
                 mass_mean=float(dm.mean() * 100), mass_median=float(np.median(dm) * 100), mass_sd=float(dm.std(ddof=1) * 100), mass_min=float(dm.min() * 100), mass_max=float(dm.max() * 100),
                 mass_pct_pos=float(np.mean(dm > 0) * 100), mass_pct_neg=float(np.mean(dm < 0) * 100))
        AN["sample"][f"{l}|{sp}"] = s
        print(f" lam={l} sp{sp}: dRecall mean {s['mean']:+.3f} med {s['median']:+.3f} sd {s['sd']:.3f} min/max {s['min']:+.2f}/{s['max']:+.2f} | >0 {s['pct_pos']:.1f}% <0 {s['pct_neg']:.1f}% =0 {s['pct_zero']:.1f}% | "
              f"top10|d| share {s['top10_abs_share']:.0f}%  mean w/o top10 {s['mean_drop_top10_abs']:+.3f}  Wilcoxon p={wp:.2e}")
        print(f"                 dMass  mean {s['mass_mean']:+.3f} med {s['mass_median']:+.3f} sd {s['mass_sd']:.3f} min/max {s['mass_min']:+.2f}/{s['mass_max']:+.2f} | >0 {s['mass_pct_pos']:.1f}% <0 {s['mass_pct_neg']:.1f}%")
# by language
print("\n by language (dRecall mean, ALL 200):")
for lg in ("python", "java"):
    m = np.array([p["language"] == lg for p in pc])
    print("  " + lg + ": " + "  ".join(f"l{l}/sp{sp} {((REC[l][sp]-REC['0.0'][sp]).mean(1)[m].mean()*100):+.3f}" for l in ("1.0", "2.0") for sp in SP))

print(f"\n{'#'*110}\n# DIAGNOSTIC recall-vs-mass and rank tiers (ALL 200 / CLEAN 191)\n{'#'*110}")
for name in ("ALL 200", "CLEAN 191"):
    mk = SUBS[name]
    print(f"\n--- {name} ---")
    for l in LAMS[1:]:
        for sp in SP:
            dR = (REC[l][sp] - REC["0.0"][sp]).mean(1)[mk]; dM = (MAS[l][sp] - MAS["0.0"][sp]).mean(1)[mk]
            n = len(dR)
            q = {"recall+ mass-": (dR > 0) & (dM < 0), "recall+ mass+": (dR > 0) & (dM > 0), "recall- mass+": (dR < 0) & (dM > 0), "recall- mass-": (dR < 0) & (dM < 0),
                 "recall+ mass=": (dR > 0) & (dM == 0), "recall= (any mass)": (dR == 0), "recall- mass=": (dR < 0) & (dM == 0)}
            DG["quadrants"][f"{name}|{l}|{sp}"] = {k: float(v.mean() * 100) for k, v in q.items()}
            print(f" lam={l} sp{sp}: " + "  ".join(f"{k}: {v.mean()*100:.1f}%" for k, v in q.items() if v.any()))
    print("  tiers (HB-style rank tiers of K*: A=top10% attn, B=10-50%, C=bottom50%); values are lambda - SA, pp, [95% CI]")
    for l in LAMS[1:]:
        for sp in SP:
            def dd(key):
                a = pool(ps[l][str(sp)][key])[mk]; b = pool(ps["0.0"][str(sp)][key])[mk]
                return a - b
            row = {}
            txt = []
            for key in ("recall", "recall_A", "recall_B", "recall_C", "mass_A", "mass_B", "mass_C", "extra", "mass"):
                x = dd(key); lo, hi = ci(x)
                row[key] = [float(x.mean() * 100), float(lo * 100), float(hi * 100)]
                txt.append(f"{key} {x.mean()*100:+.3f}{'*' if (lo>0 or hi<0) else ''}")
            fr = {"A": 0.1, "B": 0.4, "C": 0.5}
            row["recall_gain_by_tier"] = {t: float(fr[t] * row[f"recall_{t}"][0]) for t in "ABC"}
            DG["tiers"][f"{name}|{l}|{sp}"] = row
            print(f"  lam={l} sp{sp}: " + " | ".join(txt) + f"   [recall gain by tier A/B/C: {row['recall_gain_by_tier']['A']:+.3f}/{row['recall_gain_by_tier']['B']:+.3f}/{row['recall_gain_by_tier']['C']:+.3f}]")

# ============================================================ 4. sensitivity + C2 criterion
print(f"\n{'#'*110}\n# SENSITIVITY: ALL 200 vs CLEAN 191 (and D6-198)\n{'#'*110}")
for l in LAMS[1:]:
    for sp in SP:
        a = AN["main"][f"ALL 200|{l}|{sp}"]; c = AN["main"][f"CLEAN 191|{l}|{sp}"]; d6 = AN["main"][f"D6 198 (excl. 21,102)|{l}|{sp}"]
        chg = []
        for m, sg in (("R", "sigR"), ("M", "sigM")):
            k = "dRecall" if m == "R" else "dMass"
            if a[sg] != c[sg] or np.sign(a[k]) != np.sign(c[k]): chg.append(f"{k}: sign/sig CHANGED (all {a[k]:+.3f}/{a[sg]:+d} -> clean {c[k]:+.3f}/{c[sg]:+d})")
        AN["sensitivity"][f"{l}|{sp}"] = dict(all=a, clean=c, d6=d6, changed=chg,
                                             shift_recall=c["dRecall"] - a["dRecall"], shift_mass=c["dMass"] - a["dMass"])
        print(f" lam={l} sp{sp}: dRecall all {a['dRecall']:+.3f} clean {c['dRecall']:+.3f} D6 {d6['dRecall']:+.3f} | dMass all {a['dMass']:+.3f} clean {c['dMass']:+.3f} D6 {d6['dMass']:+.3f} | {'; '.join(chg) if chg else 'no sign/significance change'}")

print(f"\n{'#'*110}\n# C2 CRITERION: significant recall improvement over SA at >=2 of 3 sparsities (paired protocol)\n{'#'*110}")
for name in ("ALL 200", "CLEAN 191"):
    for l in LAMS[1:]:
        sigpos = [sp for sp in SP if AN["main"][f"{name}|{l}|{sp}"]["sigR"] == 1]
        signeg = [sp for sp in SP if AN["main"][f"{name}|{l}|{sp}"]["sigR"] == -1]
        massneg = [sp for sp in SP if AN["main"][f"{name}|{l}|{sp}"]["sigM"] == -1]
        massneg_any = [sp for sp in SP if AN["main"][f"{name}|{l}|{sp}"]["sigM"] == 1]
        verdict = "PASS" if len(sigpos) >= 2 else "FAIL"
        AN["c2"][f"{name}|{l}"] = dict(sig_recall_improvement_at=sigpos, sig_recall_decrease_at=signeg, sig_mass_decrease_at=massneg, sig_mass_increase_at=massneg_any, verdict=verdict)
        print(f" {name:9s} lam={l}: significant recall improvement at {sigpos or 'none'} -> C2 {verdict};  significant recall DEcrease at {signeg or 'none'};  significant mass decrease at {massneg or 'none'}, mass increase at {massneg_any or 'none'}")
print("\nDescriptive ranking of lambda (ALL 200; mean over sp; no tuning):")
for l in LAMS[1:]:
    mr = np.mean([AN["main"][f"ALL 200|{l}|{sp}"]["dRecall"] for sp in SP]); mm = np.mean([AN["main"][f"ALL 200|{l}|{sp}"]["dMass"] for sp in SP])
    print(f"  lambda={l}: mean dRecall {mr:+.3f} pp, mean dMass {mm:+.3f} pp")
json.dump(AN, open(f"{OUT}/analysis.json", "w"), indent=1)
json.dump(DG, open(f"{OUT}/diagnostic.json", "w"), indent=1)
print("\n>>> wrote analysis.json, diagnostic.json, symbol_coverage.json, integrity.json in", OUT)
