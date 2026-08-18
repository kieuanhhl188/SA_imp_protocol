#!/usr/bin/env python
"""
compare_runs.py — so hai lan chay THEO CAP tren cung tap mau.

VI SAO CAN FILE NAY
-------------------
Hieu cua hai diem trung binh khong noi len gi khi n nho. Ngay 18/8, Sq-70% 62.55 vs
All-KV 65.35 tren 20 mau: sai so chuan cua rieng mot trung binh da vuot 5 diem, nen
-2.80 doc mot minh la vo nghia.

Nhung hai lan chay dung CUNG 20 mau, CUNG model, chi khac co use_centroids. Do la thiet
ke ghep cap: dai luong dung la phan bo HIEU SO TUNG MAU. Phan lon mau thuong cho ra
prediction y het nhau -> hieu so = 0 -> phuong sai nho hon han, va vai mau khac biet
that thi lo ra ngay.

Protocol muc 5.5 cung yeu cau "paired test qua cac mau" cho Phase 5.

USAGE
-----
    python scripts/compare_runs.py \
        LongBench/pred/<A>/result_detail.json \
        LongBench/pred/<B>/result_detail.json --task lcc
"""
import argparse
import json
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(path, task):
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] khong thay {path}")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if task not in d:
        raise SystemExit(f"[ERROR] {path} khong co task '{task}' (co: {list(d)})")
    blk = d[task]
    per = blk.get("per_sample") or {}
    if not per:
        raise SystemExit(f"[ERROR] {path} thieu 'per_sample' — chay lai eval.py ban moi")
    return blk, {int(k): float(v) for k, v in per.items()}


def sign_test_p(n_pos, n_neg):
    """Sign test hai phia, chinh xac. Khong can scipy."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="result_detail.json cua lan chay A (thuong la All-KV)")
    ap.add_argument("b", help="result_detail.json cua lan chay B (thuong la Sq-70%)")
    ap.add_argument("--task", default="lcc")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--show", type=int, default=8, help="in N mau lech nhieu nhat")
    args = ap.parse_args()

    blk_a, pa = load(args.a, args.task)
    blk_b, pb = load(args.b, args.task)

    common = sorted(set(pa) & set(pb))
    if not common:
        raise SystemExit("[ERROR] hai lan chay khong co dataidx nao chung")
    only_a, only_b = sorted(set(pa) - set(pb)), sorted(set(pb) - set(pa))

    diffs = [pb[i] - pa[i] for i in common]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else 0.0

    same = sum(1 for d in diffs if abs(d) < 1e-9)
    better = sum(1 for d in diffs if d > 1e-9)
    worse = sum(1 for d in diffs if d < -1e-9)

    print("=" * 70)
    print(f"  SO THEO CAP — task '{args.task}', {n} mau chung")
    print(f"    {args.label_a}: {blk_a.get('score')}   <- {args.a}")
    print(f"    {args.label_b}: {blk_b.get('score')}   <- {args.b}")
    print("=" * 70)
    if only_a or only_b:
        print(f"  [!] chi co o A: {only_a[:10]} | chi co o B: {only_b[:10]}")
    print()
    print(f"  Mau GIONG HET      : {same}/{n}  ({100*same/n:.0f}%)")
    print(f"  {args.label_b} tot hon : {better}/{n}")
    print(f"  {args.label_b} kem hon : {worse}/{n}")
    print()
    print(f"  Hieu so trung binh : {100*mean:+.2f} diem")
    print(f"  Do lech chuan      : {100*sd:.2f}")
    print(f"  Sai so chuan (SE)  : {100*se:.2f}")
    print(f"  Khoang tin cay 95% : [{100*(mean-1.96*se):+.2f}, {100*(mean+1.96*se):+.2f}]")
    print()

    p = sign_test_p(better, worse)
    print(f"  Sign test (2 phia) : p = {p:.4f}  tren {better+worse} mau co thay doi")
    try:
        from scipy.stats import wilcoxon
        nz = [d for d in diffs if abs(d) > 1e-9]
        if len(nz) >= 5:
            print(f"  Wilcoxon signed-rank: p = {wilcoxon(nz).pvalue:.4f}")
    except Exception:
        pass
    print()

    ci_lo = 100 * (mean - 1.96 * se)
    ci_hi = 100 * (mean + 1.96 * se)
    if ci_lo <= 0 <= ci_hi:
        print("  => Khoang tin cay CHUA 0: chenh lech KHONG co y nghia thong ke o n nay.")
        print("     Muon ket luan thi phai tang so mau, khong phai doi nguong.")
    elif ci_hi < 0:
        print(f"  => {args.label_b} THUC SU kem hon, khong phai nhieu.")
    else:
        print(f"  => {args.label_b} THUC SU tot hon, khong phai nhieu.")

    if same:
        print()
        print(f"  Ghi chu: {same}/{n} mau cho prediction y het nhau. Do la ly do phai so")
        print("  theo cap — phuong sai cua hieu so nho hon han phuong sai cua tung diem.")

    ranked = sorted(common, key=lambda i: pb[i] - pa[i])
    show = [i for i in ranked if abs(pb[i] - pa[i]) > 1e-9][:args.show]
    if show:
        print()
        print(f"  {len(show)} mau lech nhieu nhat ({args.label_b} kem nhat truoc):")
        print(f"    {'idx':>5} {'A':>8} {'B':>8} {'hieu':>8}")
        for i in show:
            print(f"    {i:>5} {100*pa[i]:>8.1f} {100*pb[i]:>8.1f} {100*(pb[i]-pa[i]):>+8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
