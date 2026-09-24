#!/usr/bin/env python
"""
phase6_bootstrap.py — bang Phase 6 (C1) cho RepoBench-P: EM, ES, bootstrap ghep cap, W/L/T,
thoi gian va VRAM, doc truc tiep tu output cua pred.py/eval.py.

VI SAO KHONG DUNG compare_runs.py
---------------------------------
compare_runs.py chi nhan 2 lan chay, KTC la +-1.96*SE (xap xi chuan) va khong co EM.
Phase 6 can 3 cau hinh, cung mot bo resample cho moi cap, va EM.

EM: dong code dau tien khong chua ` # // (dung cach code_sim_score trong LongBench/metrics.py
trich dong), strip hai dau, so khop nguyen van voi answer. ES lay tu per_sample trong
result_detail.json — dung so eval.py da cham, khong tinh lai (pod khong co fuzzywuzzy).

USAGE (tu thu muc goc repo con)
-----
    python scripts/phase6_bootstrap.py
    python scripts/phase6_bootstrap.py --task repobench-p --B 20000 --seed 0
"""
import argparse
import json
import sys

import numpy as np

P = "LongBench/pred/longchat-v1.5-7b-32k_"
RUNS = {"All-KV": P + "baseline_lim200_runfull200",
        "SA-70": P + "PC5_PERC0.7_lim200_runfull200sa",
        "Class-70": P + "PC5_PERC0.7_lim200_runfull200class"}
PAIRS = [("All-KV", "SA-70"), ("All-KV", "Class-70"), ("SA-70", "Class-70")]


def first_code_line(pred):
    # cung cach trich dong voi code_sim_score
    for line in pred.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def load(d, task):
    rows = {}
    with open(f"{d}/{task}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["dataidx"]] = r
    idx = sorted(rows)
    em = np.array([float(any(first_code_line(rows[i]["pred"]).strip() == a.strip()
                             for a in rows[i]["answers"])) for i in idx])
    with open(f"{d}/result_detail.json", encoding="utf-8") as f:
        blk = json.load(f)[task]
    es = np.array([float(blk["per_sample"][str(i)]) for i in idx])
    if abs(es.mean() * 100 - blk["score"]) >= 0.01:
        raise SystemExit(f"[ERROR] {d}: TB per_sample {es.mean()*100:.3f} != score {blk['score']}")
    stats = [json.loads(l) for l in open(f"{d}/_logs/{task}.stats.jsonl", encoding="utf-8")]
    return idx, es * 100, em * 100, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="repobench-p")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    es, em, st, ids = {}, {}, {}, {}
    for k, d in RUNS.items():
        ids[k], es[k], em[k], st[k] = load(d, args.task)
    ref = ids["All-KV"]
    for k in RUNS:
        if ids[k] != ref:
            raise SystemExit(f"[ERROR] {k} khong cung tap dataidx voi All-KV")
    n = len(ref)

    print(f"task={args.task}  n={n}  B={args.B}  seed={args.seed}  (percentile 95% CI)")
    print(f"\n{'Config':<10} {'EM %':>7} {'ES':>7} {'t/mau (s)':>16} {'VRAM max':>9} {'VRAM tb':>8}")
    for k in RUNS:
        t = np.array([r["gen_time_s"] for r in st[k]])
        v = np.array([r["peak_alloc_gib"] for r in st[k]])
        print(f"{k:<10} {em[k].mean():>7.2f} {es[k].mean():>7.2f} "
              f"{t.mean():>7.2f} +- {t.std():>5.2f} {v.max():>9.2f} {v.mean():>8.2f}")

    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, n, size=(args.B, n))  # cung bo resample cho moi cap
    print(f"\n{'Cap (B - A)':<20} {'dES':>7} {'KTC95':>18} {'p':>7} {'dEM':>7} {'KTC95':>18} {'W/L/T (ES)':>12}")
    for a, b in PAIRS:
        out = []
        for x in (es, em):
            d = x[b] - x[a]
            bm = d[boot].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            p = min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))
            out.append((d.mean(), lo, hi, p))
        d = es[b] - es[a]
        wlt = f"{(d > 1e-9).sum()}/{(d < -1e-9).sum()}/{(abs(d) <= 1e-9).sum()}"
        (d1, l1, h1, p1), (d2, l2, h2, _) = out
        print(f"{b + ' - ' + a:<20} {d1:>+7.2f} [{l1:+6.2f}; {h1:+6.2f}] {p1:>7.4f} "
              f"{d2:>+7.2f} [{l2:+6.2f}; {h2:+6.2f}] {wlt:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
