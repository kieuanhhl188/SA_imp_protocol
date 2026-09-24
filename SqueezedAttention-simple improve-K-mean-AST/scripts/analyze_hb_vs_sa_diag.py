"""
analyze_hb_vs_sa_diag.py — chan doan Proposal 1 (SA + HardBoundary) tren RepoBench-P n=200.

CHI DOC ket qua da co. Khong sua clustering/danh gia, khong tinh lai C2, khong sinh van ban.

Nguon:
  recall theo (mau, lop):  phase5_repobench_class_n200.json         (sa, hard_boundary_class)
  mass   theo (mau, lop):  phase5_repobench_class_tw_n200.json       (per_sample_mass; sa/hbc dong nhat bit)
  phan ra theo bac K*:     diag_hb_vs_sa/rank_tiers_n200.json        (scripts/diag_hb_vs_sa_rank_tiers.py)
  cac level khac:          phase5_repobench{,_block,_statement}.json (chi recall theo mau; mass chi co TRUNG BINH)
"""
import json
import sys
import numpy as np
from scipy import stats

W = "/workspace"
OUT = f"{W}/diag_hb_vs_sa"
TRUNC = [34, 46, 59, 70, 75, 90, 154, 161, 170]
SP = [70, 80, 90]
NL, B, SEED = 3, 20000, 0
rng = np.random.default_rng(SEED)

cls = json.load(open(f"{W}/phase5_repobench_class_n200.json"))
tw = json.load(open(f"{W}/phase5_repobench_class_tw_n200.json"))
dg = json.load(open(f"{OUT}/rank_tiers_n200.json"))
sl = json.load(open(f"{W}/phase5_repobench_sa_struct_l1_n200.json"))
assert cls["n_samples"] == 200 and dg["idx"] == list(range(200))
HB = "hard_boundary_class"


def lay(v): return np.asarray(v, float).reshape(-1, NL)       # [n, 3]
def pool(v): return lay(v).mean(1)                            # [n]


REC = {b: {sp: lay(cls["per_sample"][b][str(sp)]) for sp in SP} for b in ("sa", HB)}
MAS = {b: {sp: lay(tw["per_sample_mass"][b][str(sp)]) for sp in SP} for b in ("sa", HB)}
# doi chieu: recall diag == recall C2, mass diag == mass C2 (bit/1e-6)
for b in ("sa", HB):
    for sp in SP:
        assert np.abs(lay(dg["branches"][b][str(sp)]["recall"]) - REC[b][sp]).max() < 1e-6
        assert np.abs(lay(dg["branches"][b][str(sp)]["mass"]) - MAS[b][sp]).max() < 1e-5
print("[ok] diag recall/mass == C2 recall/mass (per sample x layer, 200x3)")

idx = np.arange(200)
SUBS = {"ALL 200": np.ones(200, bool), "CLEAN 191": ~np.isin(idx, TRUNC)}


def ci(x):
    ii = rng.integers(0, len(x), size=(B, len(x)))
    return np.percentile(x[ii].mean(1), [2.5, 97.5])


def fmt_ci(x):
    lo, hi = ci(x)
    return f"{x.mean()*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}]{'*' if (lo > 0 or hi < 0) else ' '}"


R = {}   # results for json


# ====================================================================== 1. recall vs mass
def part1(name, mk):
    print(f"\n{'#'*100}\n# PART 1 — recall vs attention mass, class level   [{name}, n={int(mk.sum())}]\n{'#'*100}")
    for sp in SP:
        dR = (REC[HB][sp] - REC["sa"][sp])          # [n,3]
        dM = (MAS[HB][sp] - MAS["sa"][sp])
        for unit, a, bb in (("sample (pooled 3 layers)", dR.mean(1)[mk], dM.mean(1)[mk]),
                            ("sample x layer (n*3)", dR[mk].ravel(), dM[mk].ravel())):
            n = len(a)
            q = {"R+ M-": (a > 0) & (bb < 0), "R+ M+": (a > 0) & (bb >= 0),
                 "R- M-": (a <= 0) & (bb < 0), "R- M+": (a <= 0) & (bb >= 0)}
            print(f"\n  sp{sp}  unit={unit}  n={n}   mean dRecall={a.mean()*100:+.3f}pp  mean dMass={bb.mean()*100:+.3f}pp")
            for k, m in q.items():
                if m.sum():
                    print(f"     {k}: {m.sum():4d} ({m.mean()*100:5.1f}%)   mean dRecall={a[m].mean()*100:+.3f}  "
                          f"median={np.median(a[m])*100:+.3f} | mean dMass={bb[m].mean()*100:+.3f}  median={np.median(bb[m])*100:+.3f}")
                else:
                    print(f"     {k}:    0")
            R[(name, "quad", sp, unit)] = {k: [int(m.sum()), float(a[m].mean() * 100) if m.sum() else None,
                                              float(bb[m].mean() * 100) if m.sum() else None] for k, m in q.items()}
        # tier decomposition (sample-level pooled)
        print(f"\n  sp{sp}  RANK-TIER DECOMPOSITION of K* (A=top10% attention, B=10-50%, C=bottom 50% of K*)   [pooled over samples/layers]")
        g = lambda b, key: pool(dg["branches"][b][str(sp)][key])[mk]
        print(f"     {'':26s}{'SA':>9s}{'HB':>9s}   {'HB-SA (pp) [95% CI]':>34s}")
        rows = [("recall (overall)", "recall")] + [(f"recall tier {t}", f"recall_{t}") for t in "ABC"] + \
               [(f"mass from tier {t} (of mass*)", f"mass_{t}") for t in "ABC"] + \
               [("mass outside K* (extras)", "extra"), ("mass ratio (total)", "mass")]
        for lab, key in rows:
            sa, hb = g("sa", key), g(HB, key)
            print(f"     {lab:26s}{sa.mean()*100:9.2f}{hb.mean()*100:9.2f}   {fmt_ci(hb - sa):>34s}")
            R[(name, "tier", sp, key)] = [float(sa.mean() * 100), float(hb.mean() * 100), float((hb - sa).mean() * 100)] + [float(x * 100) for x in ci(hb - sa)]
        share = {t: g("sa", f"star_share_{t}").mean() for t in "ABC"}
        print(f"     share of K* attention mass by tier: A={share['A']*100:.1f}%  B={share['B']*100:.1f}%  C={share['C']*100:.1f}%")
        # contribution of each tier to the recall gain (tier length fractions ~0.1/0.4/0.5)
        fr = {"A": 0.1, "B": 0.4, "C": 0.5}
        contrib = {t: fr[t] * (g(HB, f"recall_{t}") - g("sa", f"recall_{t}")).mean() * 100 for t in "ABC"}
        tot = (g(HB, "recall") - g("sa", "recall")).mean() * 100
        print(f"     recall gain decomposed by tier length-weight: A={contrib['A']:+.3f}  B={contrib['B']:+.3f}  C={contrib['C']:+.3f}  (sum={sum(contrib.values()):+.3f}; measured {tot:+.3f})")
        # extras
        for b in ("sa", HB):
            eo = dg["branches"][b][str(sp)]["extra_over_thr"]
            ef = dg["branches"][b][str(sp)]["n_extra_frac"]
            print(f"     {b:20s} #extras (not in K*) = {pool(ef)[mk].mean()*100:.1f}% of N;  extra key attention / N-th-largest attention = {np.nanmean(pool(eo)[mk]):.3f}")


# ====================================================================== 3. sample-level
def part3(name, mk):
    print(f"\n{'#'*100}\n# PART 3 — sample-level distribution   [{name}, n={int(mk.sum())}]\n{'#'*100}")
    k1 = np.array(sl["struct_l1"]["k1"], float); nctx = np.array(sl["struct_l1"]["n_ctx"], float)
    for sp in SP:
        print(f"\n  sp{sp}")
        for metric, D in (("recall", REC), ("mass", MAS)):
            sa = D["sa"][sp].mean(1)[mk]; hb = D[HB][sp].mean(1)[mk]; d = hb - sa
            n = len(d)
            lo, hi = ci(d)
            w = stats.wilcoxon(d) if np.any(d != 0) else None
            tt = stats.ttest_1samp(d, 0.0)
            srt = np.sort(d)[::-1]
            top10_share = srt[:10][srt[:10] > 0].sum() / d[d > 0].sum() if (d > 0).any() else float("nan")
            drop5 = np.sort(d)[:-5].mean(); drop10 = np.sort(d)[:-10].mean()
            trim = stats.trim_mean(d, 0.1)
            print(f"   {metric:6s} SA mean/med/sd = {sa.mean()*100:.2f}/{np.median(sa)*100:.2f}/{sa.std(ddof=1)*100:.2f} | HB = {hb.mean()*100:.2f}/{np.median(hb)*100:.2f}/{hb.std(ddof=1)*100:.2f}")
            print(f"          paired d: mean {d.mean()*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}]  median {np.median(d)*100:+.3f}  sd {d.std(ddof=1)*100:.3f}  dz={d.mean()/d.std(ddof=1):+.2f}"
                  f"  10%-trimmed {trim*100:+.3f}")
            print(f"          %samples d>0: {np.mean(d>0)*100:.1f}%  d<0: {np.mean(d<0)*100:.1f}%   pct5/25/75/95 = "
                  f"{np.percentile(d,5)*100:+.2f}/{np.percentile(d,25)*100:+.2f}/{np.percentile(d,75)*100:+.2f}/{np.percentile(d,95)*100:+.2f}  min/max {d.min()*100:+.2f}/{d.max()*100:+.2f}  skew {stats.skew(d):+.2f}")
            print(f"          Wilcoxon p={w.pvalue if w else float('nan'):.2e}  t-test p={tt.pvalue:.2e}   mean after dropping top5/top10 samples: {drop5*100:+.3f}/{drop10*100:+.3f}"
                  f"   top-10 samples' share of total positive d: {top10_share*100:.0f}%")
            r_n = stats.spearmanr(d, nctx[mk])[0]; r_k = stats.spearmanr(d, k1[mk])[0]
            print(f"          Spearman(d, n_ctx)={r_n:+.2f}  Spearman(d, #class-units)={r_k:+.2f}")
            R[(name, "sample", sp, metric)] = dict(mean=float(d.mean() * 100), median=float(np.median(d) * 100), sd=float(d.std(ddof=1) * 100),
                                                   ci=[float(lo * 100), float(hi * 100)], frac_pos=float(np.mean(d > 0)), wilcoxon_p=float(w.pvalue) if w else None,
                                                   drop_top5=float(drop5 * 100), drop_top10=float(drop10 * 100), trimmed=float(trim * 100))
        # per layer
        for li, ln in enumerate(("layer 0 (first)", "layer 16 (mid)", "layer 31 (last)")):
            dr = (REC[HB][sp][:, li] - REC["sa"][sp][:, li])[mk]; dm = (MAS[HB][sp][:, li] - MAS["sa"][sp][:, li])[mk]
            print(f"   {ln:16s} dRecall {fmt_ci(dr)}   dMass {fmt_ci(dm)}   (%samples dRecall>0: {np.mean(dr>0)*100:.0f}%, dMass<0: {np.mean(dm<0)*100:.0f}%)")


# ====================================================================== 2. by level
def load_level(fn, branch, hier):
    j = json.load(open(f"{W}/{fn}"))
    ps = j["per_sample"]
    g = (lambda b, sp: ps[b]["1.0"][str(sp)]) if hier else (lambda b, sp: ps[b][str(sp)])
    sm = (lambda b, sp: j["summary"][b]["1.0"][str(sp)]) if hier else (lambda b, sp: j["summary"][b][str(sp)])
    ids = [i for i in range(200) if i not in set(j["skipped_idx"])]
    return j, g, sm, ids


def part2():
    print(f"\n{'#'*100}\n# PART 2 — by structural level (RepoBench-P, LongChat, 5% budget)\n{'#'*100}")
    fe = json.load(open("/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST/phase2_evidence/repobench_11-9/feasibility_repobench-p_hard_boundary_function_pc5.json"))
    infeas_fn = set(fe["infeasible"] if not isinstance(fe["infeasible"][0], dict) else [x["dataidx"] for x in fe["infeasible"]]) if fe["infeasible"] else set()
    levels = [("class", "phase5_repobench_class_n200.json", HB, False),
              ("function", "phase5_repobench.json", "hard_boundary", True),
              ("block", "phase5_repobench_block.json", "hard_boundary_block", False),
              ("statement", "phase5_repobench_statement.json", "hard_boundary_statement", False)]
    for sub, mkall in (("ALL (each level's own feasible set)", None), ("CLEAN (drop the 9 truncated ids where present)", TRUNC)):
        print(f"\n  === {sub} ===")
        print(f"  {'level':10s}{'n':>5s} | {'sp':>3s} {'SA rec':>7s} {'HB rec':>7s} {'dRecall (pp) [95% CI]':>28s} {'%d>0':>5s} | {'SA mass':>8s}{'HB mass':>8s}{'dMass (mean)':>13s} | {'class-level on SAME ids: dRecall / dMass':>44s}")
        for lv, fn, hb, hier in levels:
            j, g, sm, ids = load_level(fn, hb, hier)
            if lv == "function":
                ids = [i for i in ids if i not in infeas_fn]
            assert len(ids) * NL == len(g("sa", 70)), (lv, len(ids), len(g("sa", 70)))
            keep = np.array([True if mkall is None else (i not in mkall) for i in ids])
            ids_a = np.array(ids)[keep]
            for sp in SP:
                sa = pool(g("sa", sp))[keep]; h = pool(g(hb, sp))[keep]; d = h - sa
                # mass: chi co TRUNG BINH (summary) trong ban goc -> chi bao cho ALL
                if mkall is None:
                    ms, mh = sm("sa", sp)["mass"], sm(hb, sp)["mass"]; mtxt = f"{ms*100:8.2f}{mh*100:8.2f}{(mh-ms)*100:+13.2f}"
                else:
                    mtxt = f"{'n/a':>8s}{'n/a':>8s}{'n/a':>13s}"
                # class-level tren cung tap id (recall + mass theo mau)
                cd_r = (REC[HB][sp].mean(1) - REC["sa"][sp].mean(1))[ids_a]
                cd_m = (MAS[HB][sp].mean(1) - MAS["sa"][sp].mean(1))[ids_a]
                print(f"  {lv if sp==70 else '':10s}{len(ids_a) if sp==70 else '':>5} | {sp:>3d} {sa.mean()*100:7.2f} {h.mean()*100:7.2f} {fmt_ci(d):>28s} {np.mean(d>0)*100:5.0f} | {mtxt} | {cd_r.mean()*100:+8.2f} / {cd_m.mean()*100:+7.2f}")
                R[(sub[:5], "level", lv, sp)] = dict(n=int(len(ids_a)), d_recall=float(d.mean() * 100), frac_pos=float(np.mean(d > 0)))


for name, mk in SUBS.items():
    part1(name, mk)
part2()
for name, mk in SUBS.items():
    part3(name, mk)
json.dump({" | ".join(map(str, k)): v for k, v in R.items()}, open(f"{OUT}/analysis_numbers.json", "w"), indent=1, default=float)
print("\n>>> wrote", f"{OUT}/analysis_numbers.json")
