#!/usr/bin/env python
"""
compare_partitions.py — hai nhanh Phase 2 cho ra phan hoach khac nhau BAO NHIEU?

Chay CPU, khong can GPU, khong can model. Chi doc file .pt da luu.

VI SAO CAN FILE NAY
-------------------
`check_phase2_invariants.py` tra loi "co dung ranh gioi cung khong" (co/khong).
File nay tra loi cau hoi ke tiep, dinh luong: **can thiep manh tay den dau**.

Neu `hard_boundary` cho ra phan hoach gan trung `sa` (ARI ~0.9) thi du bat bien A dep,
accuracy hai ben cung se gan nhu nhau — va con so "khong khac gi SA" o Phase 6 se KHONG
phai bang chung chong lai y tuong, ma chi la he qua cua viec can thiep qua nhe. Biet
truoc dieu do thay doi cach doc ket qua.

Nguoc lai neu ARI thap (~0.3) thi hai phuong phap thuc su phan hoach khac han nhau, va
moi chenh lech accuracy sau nay deu quy duoc cho cau truc.

BON DAI LUONG
-------------
  1. ARI (adjusted Rand index) giua tung cap nhanh, tinh tren nhan cluster cua tung head.
     1.0 = trung khop tuyet doi · 0.0 = giong nhu gan ngau nhien.
  2. So cluster THUC SU duoc dung (nhan xuat hien it nhat 1 lan). Ngan sach danh nghia
     bang nhau khong co nghia la so cluster hieu dung bang nhau — cluster rong la ngan
     sach bi lang phi, phai bao cao.
  3. Phan bo kich thuoc cluster: trung vi va max. Ranh gioi cung ep cluster nho lai o
     unit nho, nen phan bo lech han la dieu CAN kiem chung, khong phai gia dinh.
  4. Nguong toan cuc (global_threshold) hai ben. Protocol 2.6 yeu cau giu nguyen cach
     tinh nguong; neu nguong lech nhieu thi mot phan chenh lech accuracy den tu nguong
     chu khong tu clustering.

USAGE
-----
    python scripts/compare_partitions.py \\
        --cluster_dir /workspace/p2-longchat/sa/lcc=sa \\
        --cluster_dir /workspace/p2-longchat/hard_boundary/lcc=hard_boundary \\
        --cluster_dir /workspace/p2-longchat/struct_hierarchy/lcc=struct_hierarchy \\
        --max_samples 50
"""
import argparse
import os
import re
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

LABEL_RE = re.compile(r"^centroids_labels_dict_(\d+)_(\d+)\.pt$")


def index_dir(d):
    """dataidx -> (duong dan file label, K danh nghia)."""
    out = {}
    for fn in os.listdir(d):
        m = LABEL_RE.match(fn)
        if m:
            out[int(m.group(1))] = (os.path.join(d, fn), int(m.group(2)))
    return out


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", action="append", required=True,
                    help="dang TEN=/duong/dan; lap lai cho tung nhanh")
    ap.add_argument("--max_samples", type=int, default=50,
                    help="so mau lay de tinh ARI (ARI tren toan bo 500 mau x 28 lop rat cham)")
    ap.add_argument("--layers", default="first,mid,last",
                    help="lop nao de tinh ARI: 'first,mid,last' hoac danh sach so '0,13,27'")
    args = ap.parse_args()

    import torch
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    branches = {}
    for spec in args.cluster_dir:
        if "=" not in spec:
            raise SystemExit(f"[ERROR] --cluster_dir phai dang TEN=/duong/dan, nhan '{spec}'")
        name, d = spec.split("=", 1)
        if not os.path.isdir(d):
            raise SystemExit(f"[ERROR] khong phai thu muc: {d}")
        idx = index_dir(d)
        if not idx:
            raise SystemExit(f"[ERROR] khong thay file label nao trong {d}")
        branches[name] = (d, idx)
        print(f">>> {name:18s} {len(idx):4d} mau  {d}")

    common = sorted(set.intersection(*[set(v[1]) for v in branches.values()]))
    if not common:
        raise SystemExit("[ERROR] khong co dataidx nao chung giua cac nhanh")
    take = common[:args.max_samples]
    print(f">>> {len(common)} dataidx chung · tinh tren {len(take)} mau dau\n")

    names = list(branches)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    ari = defaultdict(list)
    used = defaultdict(list)          # so cluster thuc su dung / K danh nghia
    csize_med, csize_max = defaultdict(list), defaultdict(list)
    thr_val = defaultdict(list)
    budget_mismatch = []

    for di in take:
        Ks = {n: branches[n][1][di][1] for n in names}
        if len(set(Ks.values())) > 1:
            budget_mismatch.append((di, Ks))

        labs = {}
        for n in names:
            path, K = branches[n][1][di]
            L = torch.load(path, map_location="cpu")
            # {layer: [1,H,S]} -> gom lai
            layers = sorted(L)
            if args.layers in ("first,mid,last", ""):
                sel = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
            else:
                want = {int(x) for x in args.layers.split(",")}
                sel = [l for l in layers if l in want]
            labs[n] = (L, sel, K)

            # thong ke tren toan bo lop da chon
            for l in sel:
                arr = L[l].reshape(-1, L[l].shape[-1]).numpy()   # [H, S]
                for h in range(arr.shape[0]):
                    v, c = np.unique(arr[h], return_counts=True)
                    used[n].append(len(v) / max(K, 1))
                    csize_med[n].append(float(np.median(c)))
                    csize_max[n].append(int(c.max()))

        for a, b in pairs:
            La, sel, _ = labs[a]
            Lb, _, _ = labs[b]
            for l in sel:
                x = La[l].reshape(-1, La[l].shape[-1]).numpy()
                y = Lb[l].reshape(-1, Lb[l].shape[-1]).numpy()
                if x.shape != y.shape:
                    continue
                for h in range(x.shape[0]):
                    ari[(a, b)].append(adjusted_rand_score(x[h], y[h]))

    def stat(v):
        a = np.asarray(v, dtype=float)
        return a.mean(), a.std(), np.median(a)

    print("=" * 74)
    print("1. ARI — hai nhanh cho ra phan hoach giong nhau bao nhieu")
    print("=" * 74)
    print(f"   {'cap nhanh':40s} {'trung binh':>10s} {'trung vi':>10s} {'do lech':>9s}")
    for (a, b), v in ari.items():
        m, s, md = stat(v)
        print(f"   {a + ' vs ' + b:40s} {m:10.3f} {md:10.3f} {s:9.3f}")
    print()
    print("   Doc the nao: 1.0 = hai ben phan hoach y het (can thiep KHONG lam gi).")
    print("                0.0 = khac nhau nhu gan ngau nhien.")
    print("   ARI cao giua sa va hard_boundary => moi chenh lech accuracy sau nay se rat")
    print("   nho, va 'khong khac gi SA' KHONG phai bang chung chong lai y tuong.")

    print()
    print("=" * 74)
    print("2. Ngan sach centroid co duoc dung het khong")
    print("=" * 74)
    print(f"   {'nhanh':20s} {'cluster dung/K':>16s} {'size trung vi':>15s} {'size max':>10s}")
    for n in names:
        um, _, _ = stat(used[n])
        _, _, cm = stat(csize_med[n])
        _, _, cx = stat(csize_max[n])
        print(f"   {n:20s} {um * 100:15.1f}% {cm:15.0f} {cx:10.0f}")
    print()
    print("   Cluster rong = ngan sach bi lang phi. Hai nhanh 'cung 5%' ma so cluster")
    print("   HIEU DUNG lech nhau thi so sanh 'cung budget' da khong con dung nghia.")

    if budget_mismatch:
        print()
        print(f"   [!] {len(budget_mismatch)} mau co K danh nghia LECH giua cac nhanh:")
        for di, Ks in budget_mismatch[:5]:
            print(f"       dataidx {di}: {Ks}")

    # ---- nguong toan cuc ----
    print()
    print("=" * 74)
    print("3. Nguong toan cuc (protocol 2.6: phai giu nguyen cach tinh)")
    print("=" * 74)
    for n in names:
        d, idx = branches[n]
        vals = []
        for di in take:
            _, K = idx[di]
            f = os.path.join(d, f"global_threshold_{di}_{K}.pt")
            if not os.path.exists(f):
                continue
            t = torch.load(f, map_location="cpu")
            if isinstance(t, dict):
                t = list(t.values())
            if isinstance(t, (list, tuple)):
                t = [x for x in t if hasattr(x, "float")]
                if not t:
                    continue
                t = torch.stack([x.float().flatten().mean() for x in t])
            vals.append(float(torch.as_tensor(t).float().mean()))
        if vals:
            m, s, md = stat(vals)
            thr_val[n] = vals
            print(f"   {n:20s} trung binh {m:10.4f}  trung vi {md:10.4f}  do lech {s:8.4f}")
        else:
            print(f"   {n:20s} (khong doc duoc)")
    if len(thr_val) >= 2:
        base = names[0]
        for n in names[1:]:
            if thr_val.get(base) and thr_val.get(n):
                a, b = np.asarray(thr_val[base]), np.asarray(thr_val[n])
                k = min(len(a), len(b))
                rel = np.abs(a[:k] - b[:k]) / (np.abs(a[:k]) + 1e-12)
                print(f"   lech tuong doi {n} so voi {base}: trung vi {np.median(rel):.3%}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
