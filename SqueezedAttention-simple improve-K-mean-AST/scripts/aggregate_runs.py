#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_runs.py — gop N luot chay lap cua cung mot cau hinh thanh mean +- std.

VI SAO CO FILE NAY
------------------
Phase 0 nay khong con so sanh voi Table 2 nua (xem docs/PHASE0.md). Cai thay the la
tai lap chinh minh: chay lai cung mot cau hinh nhieu lan tren LCC + LongChat-7B roi
lay trung binh, de moi cai tien ve sau duoc do bang do lech chuan CUA CHINH duong ong
chu khong phai bang mot con so trong bai bao.

CANH BAO QUAN TRONG VE PHUONG SAI
---------------------------------
Duong ong nay gan nhu tat dinh:
  - pred.py giai ma tham lam (do_sample=False, num_beams=1) -> seed cua torch/numpy
    khong anh huong den output.
  - K-means truoc day hardcode random_state=0 -> centroid khong doi giua cac lan chay.
Nen neu lap lai ma KHONG doi seed K-means, std se ra 0.00 va khong noi len dieu gi
ngoai nhieu phan cung. Muon so std co nghia thi phai sinh centroid voi cac seed khac
nhau (offline_clustering.py --seeds 0 1 2) roi chay pred.py tren tung bo centroid do.

Script nay bao cao ca hai chieu: std giua cac lan chay VA so mau thuc su doi diem,
de khong nham "std = 0" voi "phuong sai da duoc do".

USAGE
-----
    python scripts/aggregate_runs.py --model longchat-v1.5-7b-32k --task lcc \
        --config baseline --config PC5_PERC0.7 --run_tags s0 s1 s2 \
        --out phase0_results/repro_lcc_summary
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MD_MARKER = "<!-- check_gate.py phụ lục bên dưới. Không xoá dòng này. -->"


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    """Do lech chuan MAU (ddof=1). n=1 -> None chu khong phai 0: mot lan chay khong
    do duoc phuong sai, tra ve 0 se bi doc nham thanh 'khong co phuong sai'."""
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(v, nd=2):
    return "-" if v is None else ("%.*f" % (nd, v))


def config_dir(model, config, run_tag):
    d = "%s_%s" % (model, config)
    if run_tag:
        d = "%s_run%s" % (d, run_tag)
    return d


def load_run(pred_dir, model, config, run_tag, task):
    """Tra ve (score, per_sample_dict|None, n_samples|None, duong_dan)."""
    d = os.path.join(pred_dir, config_dir(model, config, run_tag))
    res = os.path.join(d, "result.json")
    if not os.path.exists(res):
        return None, None, None, d
    with open(res, encoding="utf-8") as f:
        scores = json.load(f)
    if task not in scores:
        return None, None, None, d

    per = None
    n_samples = None
    detail_path = os.path.join(d, "result_detail.json")
    if os.path.exists(detail_path):
        with open(detail_path, encoding="utf-8") as f:
            detail = json.load(f)
        n_samples = (detail.get(task) or {}).get("n_samples")
        blk = (detail.get(task) or {}).get("per_sample")
        # eval.py luu per_sample o thang GOC cua metric, tuc 0..1 (code_sim_score la
        # fuzz.ratio/100), roi moi nhan 100 khi tinh trung binh cho result.json. Khong
        # nhan 100 o day thi cot mean/std (doc tu result.json, da x100) va dong ghep cap
        # (doc tu per_sample) lech nhau 100 lan -- lan chay 27/8 in ra "+0.02" trong khi
        # hieu so that la +2.0 diem.
        if isinstance(blk, dict):
            per = dict((int(k), 100.0 * float(v)) for k, v in blk.items())
        elif isinstance(blk, list):
            per = dict((i, 100.0 * float(v)) for i, v in enumerate(blk))
    return float(scores[task]), per, n_samples, d


def sample_disagreement(pers):
    """So mau co diem KHONG giong nhau qua cac lan chay, tinh tren giao cac dataidx.

    Day la phep kiem 'co that su co ngau nhien khong'. Bang 0 nghia la cac lan chay la
    ban sao cua nhau va mean +- std khong them thong tin gi.
    """
    pers = [p for p in pers if p]
    if len(pers) < 2:
        return None, None
    common = set(pers[0])
    for p in pers[1:]:
        common &= set(p)
    if not common:
        return None, None
    n_diff = sum(1 for i in common if len(set(round(p[i], 9) for p in pers)) > 1)
    return n_diff, len(common)


def paired_delta(per_a, per_b):
    """Hieu so theo tung mau (b - a): trung binh, KTC95, sign test hai phia."""
    common = sorted(set(per_a) & set(per_b))
    if not common:
        return None
    diffs = [per_b[i] - per_a[i] for i in common]
    n = len(diffs)
    m = mean(diffs)
    sd = stdev(diffs)
    se = (sd / math.sqrt(n)) if sd else None
    ci = (m - 1.96 * se, m + 1.96 * se) if se else None
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    k = min(pos, neg)
    nn = pos + neg
    p = 1.0 if nn == 0 else min(
        1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / (2 ** nn))
    return {"n": n, "mean_diff": m, "sd": sd, "ci95": ci,
            "n_better": pos, "n_worse": neg, "n_tie": n - pos - neg, "p_sign": p}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=os.path.join(repo, "LongBench", "pred"))
    ap.add_argument("--model", default="longchat-v1.5-7b-32k")
    ap.add_argument("--task", default="lcc")
    ap.add_argument("--config", action="append", default=None,
                    help="hau to thu muc sau ten model, vd 'baseline' hoac 'PC5_PERC0.7'. "
                         "Lap lai co nay cho nhieu cau hinh. Cau hinh DAU TIEN la moc "
                         "de tinh delta.")
    ap.add_argument("--run_tags", nargs="+", required=True,
                    help="danh sach --run_tag da dung khi chay pred.py")
    ap.add_argument("--out", default=None,
                    help="tien to duong dan ghi ket qua; sinh ra <out>.json va <out>.md")
    ap.add_argument("--log_md", default=None,
                    help="phu luc mot muc vao file nhat ky markdown (vd EXPERIMENT_LOG.md)")
    ap.add_argument("--note", default="", help="ghi chu tu do cho lan tong hop nay")
    args = ap.parse_args()

    configs = args.config or ["baseline", "PC5_PERC0.7"]

    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "model": args.model, "task": args.task, "run_tags": args.run_tags,
           "note": args.note, "configs": {}}
    missing = []

    all_n = {}          # (cfg, tag) -> so mau, de bat truong hop so lech nhau
    for cfg in configs:
        runs, pers = {}, []
        for tag in args.run_tags:
            score, per, n_s, d = load_run(args.pred_dir, args.model, cfg, tag, args.task)
            if score is None:
                missing.append(d)
                continue
            runs[tag] = score
            pers.append(per)
            all_n[(cfg, tag)] = n_s
        vals = list(runs.values())
        n_diff, n_common = sample_disagreement(pers)
        out["configs"][cfg] = {
            "runs": runs,
            "n_runs": len(vals),
            "mean": mean(vals) if vals else None,
            "std": stdev(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n_samples_differing_across_runs": n_diff,
            "n_samples_compared": n_common,
            "n_samples_per_run": dict((t, all_n.get((cfg, t))) for t in runs),
            "_per": pers,
        }

    # ---- delta so voi cau hinh dau tien ----
    base = configs[0]
    for cfg in configs[1:]:
        a, b = out["configs"][base], out["configs"][cfg]
        if a["mean"] is None or b["mean"] is None:
            continue
        entry = {"vs": base, "delta_of_means": b["mean"] - a["mean"]}
        pa = [p for p in a["_per"] if p]
        pb = [p for p in b["_per"] if p]
        if pa and pb:
            # trung binh per-sample qua cac lan chay roi moi ghep cap
            ka = set.intersection(*[set(p) for p in pa])
            kb = set.intersection(*[set(p) for p in pb])
            avg_a = dict((i, mean([p[i] for p in pa])) for i in ka)
            avg_b = dict((i, mean([p[i] for p in pb])) for i in kb)
            entry["paired"] = paired_delta(avg_a, avg_b)
        b["delta"] = entry

    for cfg in out["configs"]:
        out["configs"][cfg].pop("_per", None)

    # ---------- bao cao ----------
    L = []
    L.append("# Tai lap Phase 0 - %s / %s" % (args.model, args.task))
    L.append("")
    L.append("- Thoi diem: %s" % out["generated"])
    L.append("- So lan chay: %d  (run_tag: %s)" % (len(args.run_tags), ", ".join(args.run_tags)))
    if args.note:
        L.append("- Ghi chu: %s" % args.note)
    L.append("")
    L.append("| Cau hinh | so luot | so mau | mean | std | min | max | mau doi diem giua cac lan |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg in configs:
        c = out["configs"][cfg]
        diff = ("-" if c["n_samples_differing_across_runs"] is None
                else "%d/%d" % (c["n_samples_differing_across_runs"], c["n_samples_compared"]))
        ns = sorted(set(v for v in c["n_samples_per_run"].values() if v is not None))
        ns_txt = "-" if not ns else (str(ns[0]) if len(ns) == 1 else ",".join(map(str, ns)))
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            cfg, c["n_runs"], ns_txt, fmt(c["mean"]), fmt(c["std"]),
            fmt(c["min"]), fmt(c["max"]), diff))
    L.append("")

    # Diem cua 226 mau va diem cua 500 mau KHONG so duoc voi nhau. Ngay 27/8 hai con so
    # do da nam canh nhau trong bang nay va sinh ra mot "delta +2.45" vo nghia.
    seen_n = set(v for v in all_n.values() if v is not None)
    if len(seen_n) > 1:
        L.append("> [SAI] So mau KHONG giong nhau giua cac cau hinh/luot: %s. Cot mean va "
                 "delta ben duoi VO NGHIA — diem cua tap con khong so duoc voi diem cua ca "
                 "tap. Nguyen nhan thuong gap: mot luot pred.py chet giua chung. Chay lai "
                 "luot thieu voi --overwrite, va truyen --expect cho eval.py de no tu chan."
                 % ", ".join("%s/%s=%s" % (c, t, n) for (c, t), n in sorted(all_n.items())))
        L.append("")
    L.append("Diem tung lan chay:")
    L.append("")
    L.append("| Cau hinh | " + " | ".join(args.run_tags) + " |")
    L.append("|---" * (len(args.run_tags) + 1) + "|")
    for cfg in configs:
        c = out["configs"][cfg]
        L.append("| %s | " % cfg + " | ".join(fmt(c["runs"].get(t)) for t in args.run_tags) + " |")
    L.append("")

    for cfg in configs[1:]:
        d = out["configs"][cfg].get("delta")
        if not d:
            continue
        L.append("**%s vs %s**: delta cua hai trung binh = %+.2f" % (cfg, d["vs"], d["delta_of_means"]))
        pr = d.get("paired")
        if pr:
            ci = pr["ci95"]
            msg = ("- Ghep cap tren %d mau (diem da trung binh qua cac lan chay): %+.2f"
                   % (pr["n"], pr["mean_diff"]))
            if ci:
                msg += ", KTC95 [%+.2f, %+.2f]" % ci
            msg += (", tot hon %d / xau hon %d / hoa %d, sign test p=%.3f"
                    % (pr["n_better"], pr["n_worse"], pr["n_tie"], pr["p_sign"]))
            L.append(msg)
        L.append("")

    zero_var = [c for c in configs
                if out["configs"][c]["n_runs"] >= 2
                and out["configs"][c]["n_samples_differing_across_runs"] == 0]
    if zero_var:
        L.append("> [CANH BAO] " + ", ".join(zero_var) + ": KHONG mau nao doi diem giua cac "
                 "lan chay. Cac lan chay la ban sao cua nhau; std = 0 chi phan anh dieu do chu "
                 "khong phai da do duoc phuong sai. Muon so co nghia: sinh centroid voi seed "
                 "K-means khac nhau (offline_clustering.py --seeds 0 1 2) roi chay pred.py tren "
                 "tung bo centroid.")
        L.append("")
    if missing:
        L.append("> [CANH BAO] Thieu ket qua: " + ", ".join(missing))
        L.append("")

    text = "\n".join(L)
    print("")
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("  -> %s.json" % args.out)
        print("  -> %s.md" % args.out)

    if args.log_md:
        block = "\n### Tai lap " + out["generated"] + "\n\n" + text + "\n"
        if os.path.exists(args.log_md):
            with open(args.log_md, encoding="utf-8") as f:
                cur = f.read()
        else:
            cur = MD_MARKER + "\n"
        if MD_MARKER in cur:
            cur = cur.replace(MD_MARKER, MD_MARKER + "\n" + block, 1)
        else:
            cur = cur + "\n" + block
        with open(args.log_md, "w", encoding="utf-8") as f:
            f.write(cur)
        print("  -> phu luc vao %s" % args.log_md)

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
