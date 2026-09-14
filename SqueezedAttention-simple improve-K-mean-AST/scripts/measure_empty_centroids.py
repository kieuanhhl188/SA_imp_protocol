#!/usr/bin/env python
"""
measure_empty_centroids.py -- do lai "ty le centroid rong" tren du lieu LongChat/RepoBench-P
va LongChat/LCC hien co, thay cho so Qwen 22/8 cu trong docs/PHASE2_RESULTS.md Bang 3.

VI SAO CAN FILE NAY
--------------------
Bang 3 (docs/PHASE2_RESULTS.md) do "hang centroid toan 0" (cuML cap phat du K hang nhung
khong dung het) tren luot Qwen2.5-Coder 22/8. Tu 28/8 project doi han sang LongChat, nhung
khong ai do lai chi so nay tren model/dataset hien hanh -- bao cao v3 (13-14/9) phai ghi
"CHUA do" thay vi muon tam so Qwen.

Khong can chay lai GPU: centroid da luu san duoi dang .pt (torch.save cua
offline_clustering_struct.py / offline_clustering.py). Script nay chi doc lai file, dem hang
co norm == 0 -- dung logic da co san trong scripts/check_phase2_invariants.py (check [C]),
chi tach ra thanh script rieng de tong hop qua nhieu mau + nhieu nhanh mot luc, khong in
tung dong nhu check_phase2_invariants.

Chay CPU, khong can GPU, khong can model.

CACH DUNG
---------
    # RepoBench-P: sa (baseline) da co san, khong ton GPU
    python scripts/measure_empty_centroids.py \\
        --cluster_dir sa=/workspace/p2-longchat-repobench/sa \\
        --cluster_dir hard_boundary_class=/workspace/p2-longchat-repobench/hard_boundary_class

    # LCC (neu con centroid tren pod, xem README kiem tra truoc bang du -sh)
    python scripts/measure_empty_centroids.py \\
        --cluster_dir sa=/workspace/p2-longchat/sa/lcc \\
        --cluster_dir hard_boundary=/workspace/p2-longchat/hard_boundary/lcc

Neu thu muc khong con (da bi don dep) thi script bao ro "khong tim thay file nao", khong
lang le tra ve 0% -- 0% "khong co du lieu" khac han 0% "da do va dung la khong rong".
"""
import argparse
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_NAME = re.compile(r"^centroids_tensor_dict_(\d+)_(\d+)\.pt$")


def discover(cluster_dir):
    """Tra ve list (dataidx, K, duong_dan) tim duoc trong thu muc."""
    out = []
    for p in glob.glob(os.path.join(cluster_dir, "centroids_tensor_dict_*.pt")):
        m = _NAME.match(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def measure_one_branch(name, cluster_dir, limit):
    import torch

    files = discover(cluster_dir)
    if not files:
        print(f"  [!] {name}: KHONG TIM THAY file centroids_tensor_dict_*.pt nao trong "
              f"{cluster_dir} -- co the da bi don dep, khong the do lai tu day, "
              f"can chay lai clustering (GPU) neu muon co so that.")
        return None

    if limit > 0:
        files = files[:limit]

    total_rows = 0
    total_zero = 0
    per_sample = []
    for dataidx, K, path in files:
        cen = torch.load(path, map_location="cpu")
        nz = 0
        tot = 0
        for lyr in cen:
            v = cen[lyr].float()
            n = v.reshape(-1, v.shape[-1]).norm(dim=-1)
            nz += int((n == 0).sum())
            tot += n.numel()
        pct = 100.0 * nz / max(tot, 1)
        per_sample.append((dataidx, K, pct))
        total_rows += tot
        total_zero += nz

    overall = 100.0 * total_zero / max(total_rows, 1)
    print(f"  {name:20s}  {len(files):4d} mau  ·  tong {total_rows:>12,d} hang centroid "
          f"(moi layer x head x K)  ·  RONG (norm=0): {overall:.3f}%")
    worst = sorted(per_sample, key=lambda x: -x[2])[:3]
    for dataidx, K, pct in worst:
        if pct > 0:
            print(f"      vd mau dataidx={dataidx} K={K}: {pct:.2f}% rong")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", action="append", required=True,
                    help="ten=duong_dan, lap lai de do nhieu nhanh (giong check_phase2_invariants.py)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="chi doc N mau dau de test nhanh, -1 = tat ca file tim duoc")
    args = ap.parse_args()

    dirs = {}
    for spec in args.cluster_dir:
        if "=" not in spec:
            raise SystemExit(f"[ERROR] --cluster_dir phai dang ten=duong_dan, nhan '{spec}'")
        name, path = spec.split("=", 1)
        dirs[name] = path

    print("=" * 78)
    print("  DO LAI TY LE CENTROID RONG (thay so Qwen 22/8 cu trong PHASE2_RESULTS.md Bang 3)")
    print("=" * 78)
    results = {}
    for name, path in dirs.items():
        r = measure_one_branch(name, path, args.limit)
        if r is not None:
            results[name] = r
    print("=" * 78)
    if results:
        print("  Tom tat:")
        for name, pct in results.items():
            print(f"    {name:20s} {pct:.3f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
