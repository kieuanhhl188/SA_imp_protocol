# -*- coding: utf-8 -*-
"""
inspect_centroids.py — kiem tra tinh hop le cua artifact offline clustering.

Chay tren CPU, KHONG can GPU, KHONG can load model:

    python scripts/inspect_centroids.py /workspace/fixed-prompt-clusters/qwen2.5-coder-7b/lcc
    python scripts/inspect_centroids.py <dir> --dataidx 0 --verbose

Muc dich: khi accuracy tut ma khong crash, cau hoi dau tien la "centroid co dung
khong", truoc khi di soi kernel. Script nay tra loi bang chinh file .pt da luu.

BAY KIEM TRA, moi cai la mot loi tung xay ra hoac co the xay ra am tham:

  1. SHAPE      centroids [1,H,K,D] va labels [1,H,S_ctx].
                Thieu batch dim -> online doc shape[2] se lay nham D lam K.
  2. DO DAI     len(labels) phai == shared_prefix_length - observation_window.
                Lech -> online cat them obs_window roi gather se lech TOAN BO nhan.
  3. K          so centroid trong file phai == K suy ra tu ten file va tu labels.max()+1.
  4. RONG       bao nhieu cluster khong co key nao. run_clustering dat centroid cua
                cluster rong = VECTOR 0 -> q.0 = 0 -> exp(0)=1, diem gia o muc trung
                binh. Nhieu cluster rong = centroid vo nghia.
  5. ZERO       dem centroid toan 0 (dau hieu cluster rong, doi chieu voi (4)).
  6. NAN/INF    centroid co gia tri khong hop le.
  7. THRESHOLD  cac quantile phai tang dan va huu han; sp_len/obs_window khop.

Ma thoat: 0 neu moi kiem tra qua, 1 neu co van de.
"""
import argparse
import glob
import os
import re
import sys

import torch


def find_tag(d, dataidx):
    """Tim (K) tu ten file global_threshold_{dataidx}_{K}.pt."""
    pat = os.path.join(d, "global_threshold_%d_*.pt" % dataidx)
    hits = glob.glob(pat)
    if not hits:
        raise SystemExit(
            "[ERROR] khong thay %s\n"
            "        Kiem lai duong dan, hoac dataidx khac." % pat)
    m = re.search(r"global_threshold_%d_(\d+)\.pt$" % dataidx,
                  hits[0].replace("\\", "/"))
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_dir", help="vd /workspace/fixed-prompt-clusters/qwen2.5-coder-7b/lcc")
    ap.add_argument("--dataidx", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="in chi tiet tung layer")
    ap.add_argument("--max_layers", type=int, default=-1, help="chi kiem N layer dau (nhanh hon)")
    args = ap.parse_args()

    d = args.cluster_dir.rstrip("/\\")
    K_name = find_tag(d, args.dataidx)
    tag = "%d_%d" % (args.dataidx, K_name)

    cpath = os.path.join(d, "centroids_tensor_dict_%s.pt" % tag)
    lpath = os.path.join(d, "centroids_labels_dict_%s.pt" % tag)
    tpath = os.path.join(d, "global_threshold_%s.pt" % tag)
    for p in (cpath, lpath, tpath):
        if not os.path.exists(p):
            raise SystemExit("[ERROR] thieu %s" % p)

    cent = torch.load(cpath, map_location="cpu")
    lab = torch.load(lpath, map_location="cpu")
    thr = torch.load(tpath, map_location="cpu")

    print("=" * 70)
    print("  dir      : %s" % d)
    print("  dataidx  : %d   K tu ten file: %d" % (args.dataidx, K_name))
    print("  so layer : centroids=%d  labels=%d" % (len(cent), len(lab)))
    print("=" * 70)

    # ---------- threshold ----------
    sp_len = thr.get("shared_prefix_length")
    obs = thr.get("observation_window")
    qs = sorted([k for k in thr if isinstance(k, float)])
    print("\n[THRESHOLD]")
    print("  shared_prefix_length = %s   observation_window = %s" % (sp_len, obs))
    for q in qs:
        print("    q=%.2f  tau = %.6e" % (q, thr[q]))

    problems = []

    if sp_len is None or obs is None:
        problems.append("threshold dict thieu shared_prefix_length / observation_window")
    S_ctx_expect = (sp_len - obs) if (sp_len is not None and obs is not None) else None
    K_expect = max(1, int(0.05 * S_ctx_expect)) if S_ctx_expect else None
    if K_expect is not None and K_expect != K_name:
        print("  [i] K tu ten file (%d) vs 5%%*S_ctx (%d) — khac nhau la binh thuong "
              "neu percent_clusters khac 5" % (K_name, K_expect))

    vals = [thr[q] for q in qs]
    if any(not torch.isfinite(torch.tensor(float(v))) for v in vals):
        problems.append("threshold co gia tri khong huu han")
    if vals != sorted(vals):
        problems.append("threshold KHONG tang dan theo quantile: %s" % vals)

    # ---------- per layer ----------
    n_lyr = len(cent) if args.max_layers <= 0 else min(args.max_layers, len(cent))
    agg = {"empty": [], "zero": [], "used": [], "sizes_max": [], "sizes_min": []}
    shape_c = shape_l = None

    print("\n[PER-LAYER]" + ("" if args.verbose else "  (dung --verbose de in tung layer)"))
    for li in range(n_lyr):
        c, l = cent[li], lab[li]

        if shape_c is None:
            shape_c, shape_l = tuple(c.shape), tuple(l.shape)
            print("  centroids shape = %s   labels shape = %s   dtype = %s / %s"
                  % (shape_c, shape_l, c.dtype, l.dtype))
            # (1) SHAPE
            if c.dim() != 4:
                problems.append("centroids phai 4 chieu [1,H,K,D], nhan %s -> online doc "
                                "shape[2] se lay nham D lam K" % (shape_c,))
            if l.dim() != 3:
                problems.append("labels phai 3 chieu [1,H,S], nhan %s" % (shape_l,))
            if c.dim() == 4 and c.shape[0] != 1:
                problems.append("centroids batch dim = %d, phai la 1" % c.shape[0])
            # (2) DO DAI
            if l.dim() == 3 and S_ctx_expect is not None:
                if l.shape[-1] != S_ctx_expect:
                    problems.append(
                        "labels dai %d nhung sp_len-obs = %d. LECH -> online cat them "
                        "obs_window roi gather se lech TOAN BO nhan."
                        % (l.shape[-1], S_ctx_expect))
                else:
                    print("  do dai labels = %d == sp_len(%d) - obs(%d)  OK"
                          % (l.shape[-1], sp_len, obs))
            # (3) K
            if c.dim() == 4 and c.shape[2] != K_name:
                problems.append("centroids co K=%d nhung ten file noi K=%d"
                                % (c.shape[2], K_name))
            if c.dim() == 4 and l.dim() == 3 and c.shape[1] != l.shape[1]:
                problems.append("so head lech: centroids=%d labels=%d"
                                % (c.shape[1], l.shape[1]))

        if c.dim() != 4 or l.dim() != 3:
            break

        H, K = c.shape[1], c.shape[2]
        cc, ll = c[0], l[0]                       # [H,K,D], [H,S]

        # (4) cluster rong, tinh RIENG TUNG HEAD
        used_per_head, empty_per_head, smax, smin = [], [], [], []
        for h in range(H):
            cnt = torch.bincount(ll[h], minlength=K)
            used_per_head.append(int((cnt > 0).sum()))
            empty_per_head.append(int((cnt == 0).sum()))
            smax.append(int(cnt.max()))
            smin.append(int(cnt[cnt > 0].min()) if (cnt > 0).any() else 0)

        # (5) centroid toan 0
        n_zero = int((cc.abs().sum(dim=-1) == 0).sum())
        # (6) NaN/Inf
        n_bad = int((~torch.isfinite(cc)).sum())

        agg["empty"].append(sum(empty_per_head) / H)
        agg["zero"].append(n_zero / H)
        agg["used"].append(sum(used_per_head) / H)
        agg["sizes_max"].append(max(smax))
        agg["sizes_min"].append(min(smin))

        if int(ll.max()) >= K or int(ll.min()) < 0:
            problems.append("layer %d: label ngoai dai [0,%d): min=%d max=%d"
                            % (li, K, int(ll.min()), int(ll.max())))
        if n_bad:
            problems.append("layer %d: %d gia tri NaN/Inf trong centroid" % (li, n_bad))

        if args.verbose:
            print("  L%-3d H=%d K=%d | cluster dung tb %.1f/%d | rong tb %.1f | "
                  "centroid toan 0: %d | size max %d min %d"
                  % (li, H, K, sum(used_per_head) / H, K, sum(empty_per_head) / H,
                     n_zero, max(smax), min(smin)))

    # ---------- tong hop ----------
    if agg["used"]:
        K = cent[0].shape[2]
        used = sum(agg["used"]) / len(agg["used"])
        empty = sum(agg["empty"]) / len(agg["empty"])
        zero = sum(agg["zero"]) / len(agg["zero"])
        print("\n[TONG HOP qua %d layer, trung binh moi head]" % n_lyr)
        print("  cluster CO key      : %.1f / %d   (%.1f%%)" % (used, K, 100.0 * used / K))
        print("  cluster RONG        : %.1f / %d   (%.1f%%)" % (empty, K, 100.0 * empty / K))
        print("  centroid TOAN 0     : %.1f / %d   (%.1f%%)" % (zero, K, 100.0 * zero / K))
        print("  cluster lon nhat    : %d key" % max(agg["sizes_max"]))
        print("  cluster nho nhat    : %d key" % min(agg["sizes_min"]))

        # (4)/(5) nguong canh bao
        if 100.0 * empty / K > 20.0:
            problems.append(
                "%.1f%% cluster RONG. run_clustering dat centroid cua cluster rong = "
                "vector 0 -> q.0 = 0 -> exp(0)=1, sinh diem gia o muc trung binh. "
                "Nhieu cluster rong lam threshold toan cuc mat y nghia." % (100.0 * empty / K))
        if 100.0 * zero / K > 20.0:
            problems.append("%.1f%% centroid TOAN 0" % (100.0 * zero / K))
        if max(agg["sizes_max"]) > 0.5 * (sp_len - obs if S_ctx_expect else 1):
            problems.append(
                "mot cluster chua %d key (>50%% toan bo context) -> K-means suy bien, "
                "centroid khong con phan biet duoc gi" % max(agg["sizes_max"]))

    print("\n" + "=" * 70)
    if problems:
        print("  CO VAN DE (%d):" % len(problems))
        for p in problems:
            print("   - " + p)
    else:
        print("  Moi kiem tra QUA. Artifact offline hop le -> bug nam o duong ONLINE.")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
