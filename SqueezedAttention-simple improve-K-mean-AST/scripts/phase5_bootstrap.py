#!/usr/bin/env python3
"""phase5_bootstrap.py — KTC ghep cap (paired bootstrap) cho output phase5_recall.py.

Doc file --out cua phase5_recall.py, tinh hieu so recall giua moi nhanh va mot
nhanh goc (mac dinh `sa`), lay khoang tin cay 95% bang bootstrap ghep cap o cap
MAU (gop cac lop theo mau truoc — "gop lop theo mau", giong Phase 0).

  python scripts/phase5_bootstrap.py /workspace/phase5_lcc_func_hier_sp5.json
  python scripts/phase5_bootstrap.py out.json --baseline sa --n_layers 3 --B 20000

Ho tro ca file phang (per_sample[branch][sp]) lan file --hierarchical
(per_sample[branch][ratio][sp]).
"""
import argparse, json
import numpy as np


def _pool(vec, n_layers):
    a = np.asarray(vec, dtype=float)
    n = a.size // n_layers
    return a.reshape(n, n_layers).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--baseline", default="sa")
    ap.add_argument("--n_layers", type=int, default=3,
                    help="so lop trong --layers cua phase5_recall.py (thu tu stride)")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.json_path, encoding="utf-8"))
    ps = d["per_sample"]
    sps = d["sparsity"]
    hier = d.get("hierarchical", False)
    rng = np.random.default_rng(args.seed)

    def series(branch, ratio, sp):
        node = ps[branch]
        node = node[str(ratio)] if hier else node
        raw = node[str(sp)]
        if raw is None or len(raw) == 0:
            return np.empty(0)
        return _pool(raw, args.n_layers)

    ratios = (d.get("l1_ratios") or [1.0]) if hier else [1.0]
    base_ratio = 1.0 if hier else None

    n = series(args.baseline, base_ratio, sps[0]).size
    print(f"file       : {args.json_path}")
    print(f"baseline   : {args.baseline}   n_samples={n}   layers_pooled={args.n_layers}   B={args.B}")
    print(f"hierarchical: {hier}   ratios={ratios}\n")
    print(f"Hieu so (x100) = branch - {args.baseline},  KTC 95% ghep cap (percentile)")
    hdr = f"{'config':30s} " + " ".join(f"sp{sp}".ljust(24) for sp in sps)
    print(hdr)

    for branch in ps:
        for r in ratios:
            if branch == args.baseline and (r == base_ratio):
                continue
            if series(branch, r, sps[0]).size == 0:
                continue          # nhanh khong co du lieu o ty le nay (sa, hard_boundary chi co r=1.0)
            cells = []
            allneg = True
            for sp in sps:
                base = series(args.baseline, base_ratio, sp)
                cur = series(branch, r, sp)
                if base.size != cur.size:
                    cells.append("size-mismatch")
                    continue
                diff = (cur - base) * 100.0
                idx = rng.integers(0, diff.size, size=(args.B, diff.size))
                boot = diff[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cells.append(f"{diff.mean():+.2f} [{lo:+.2f};{hi:+.2f}]")
                if not (hi < 0):
                    allneg = False
            tag = "" if branch == args.baseline else ("  <all CI<0" if allneg else "")
            label = branch if not hier else f"{branch} r={r}"
            print(f"{label:30s} " + " ".join(c.ljust(24) for c in cells) + tag)


if __name__ == "__main__":
    main()
