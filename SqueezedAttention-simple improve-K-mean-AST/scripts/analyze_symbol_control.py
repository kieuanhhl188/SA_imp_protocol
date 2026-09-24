"""Chan doan phu cho SymbolSignal: (a) kiem chung xao lien ket bonus<->cluster, (b) do giau vi tri dinh danh khop. KHONG thuoc C2."""
import json
import numpy as np
O = "/workspace/phase5_repobench_symbolsignal_n200"
real = json.load(open(f"{O}/results.json"))["per_sample"]; ctl = json.load(open(f"{O}/control_shuffle/results.json"))["per_sample"]
en = json.load(open(f"{O}/control_shuffle/enrichment_raw.json"))
SP = [70, 80, 90]; NL, B = 3, 20000
rng = np.random.default_rng(0)
pool = lambda v: np.asarray(v, float).reshape(-1, NL).mean(1)
# lambda=0 cua control phai == lambda=0 cua chay chinh (bit-exact)
assert all(ctl["0.0"][str(sp)]["recall"] == real["0.0"][str(sp)]["recall"] and ctl["0.0"][str(sp)]["mass"] == real["0.0"][str(sp)]["mass"] for sp in SP)
print("[ok] control lambda=0 == main-run lambda=0 (bit-exact)")
def ci(x):
    ii = rng.integers(0, len(x), size=(B, len(x))); return np.percentile(x[ii].mean(1), [2.5, 97.5])
out = {}
print("\n(a) mean dRecall / dMass vs SA (pp):  REAL symbol bonus  vs  SHUFFLED bonus (same values, random clusters)   and  real - shuffled [95% CI]")
for l in ("0.5", "1.0", "2.0"):
    for sp in SP:
        b0r, b0m = pool(real["0.0"][str(sp)]["recall"]), pool(real["0.0"][str(sp)]["mass"])
        rr = pool(real[l][str(sp)]["recall"]) - b0r; cr = pool(ctl[l][str(sp)]["recall"]) - b0r
        rm = pool(real[l][str(sp)]["mass"]) - b0m; cm = pool(ctl[l][str(sp)]["mass"]) - b0m
        d = rr - cr; dm = rm - cm; lo, hi = ci(d); lom, him = ci(dm)
        out[f"{l}|{sp}"] = dict(real_dR=rr.mean()*100, ctl_dR=cr.mean()*100, real_minus_ctl_dR=[d.mean()*100, lo*100, hi*100], real_dM=rm.mean()*100, ctl_dM=cm.mean()*100, real_minus_ctl_dM=[dm.mean()*100, lom*100, him*100])
        print(f"  l={l} sp{sp}: recall real {rr.mean()*100:+.3f} | shuffled {cr.mean()*100:+.3f} | real-shuf {d.mean()*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}]{'*' if (lo>0 or hi<0) else ''}"
              f"     mass real {rm.mean()*100:+.3f} | shuffled {cm.mean()*100:+.3f} | real-shuf {dm.mean()*100:+.3f} [{lom*100:+.3f},{him*100:+.3f}]{'*' if (lom>0 or him<0) else ''}")
print("\n(b) key-level enrichment of matched-identifier token positions (ratio to base rate; 1.0 = no enrichment)")
base = np.array([e["base_rate"] for e in en]); ae = np.array([e["attn_enrichment"] for e in en])
print(f"  base rate (fraction of fixed-context keys that are matched-identifier tokens): mean {base.mean()*100:.2f}%")
print(f"  attention-mass enrichment: mean {ae.mean():.2f}x  median {np.median(ae):.2f}x  (share of samples*layers with enrichment>1: {np.mean(ae>1)*100:.0f}%)")
out["enrichment"] = dict(base_rate=float(base.mean()), attn_enrichment_mean=float(ae.mean()), attn_enrichment_median=float(np.median(ae)))
for sp in SP:
    ke = np.array([e[f"kstar_enrichment_sp{sp}"] for e in en]); kf = np.array([e[f"kstar_frac_matched_sp{sp}"] for e in en])
    print(f"  K* (sp{sp}) enrichment: mean {ke.mean():.2f}x  median {np.median(ke):.2f}x  (matched positions are {kf.mean()*100:.2f}% of K* vs {base.mean()*100:.2f}% of all keys); enrichment>1 in {np.mean(ke>1)*100:.0f}% of sample*layers")
    out["enrichment"][f"kstar_sp{sp}"] = dict(mean=float(ke.mean()), median=float(np.median(ke)), frac_gt1=float(np.mean(ke > 1)))
for li, ln in enumerate((0, 16, 31)):
    sub = [e for e in en if e["layer"] == ln]
    print(f"   layer {ln:2d}: attn enrichment {np.mean([e['attn_enrichment'] for e in sub]):.2f}x, K*(sp70) enrichment {np.mean([e['kstar_enrichment_sp70'] for e in sub]):.2f}x, K*(sp90) {np.mean([e['kstar_enrichment_sp90'] for e in sub]):.2f}x")
json.dump(out, open(f"{O}/control_shuffle/control_analysis.json", "w"), indent=1, default=float)
