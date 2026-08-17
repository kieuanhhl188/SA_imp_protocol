#!/usr/bin/env python
"""
inspect_preds.py — nhin vao prediction tho, va vao DONG MA METRIC THAT SU CHAM.

VI SAO CAN FILE NAY
-------------------
Ngay 17/8, gate Phase 1 bao PASS (Sq-70% 20.85 >= All-KV 17.60) trong khi ca hai con
so deu hong: model instruct sinh ra gan nhu KHONG GI CA. Diem so tong hop khong the
cho thay dieu do — chi doc prediction tho moi thay.

`code_sim_score` cua LongBench khong cham ca prediction. No lay DONG DAU TIEN khong
chua '`', '#', '//' roi fuzzy-match voi dap an:

    for line in prediction.lstrip('\\n').split('\\n'):
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line; break

Nghia la mot model sinh ra ' ```\\nfoo()\\n``` ' bi cham o dong RONG dau tien, khong
phai o `foo()`. Cot 'CHAM' ben duoi hien dung dong ma metric lay, nen khoang cach giua
'model sinh gi' va 'metric thay gi' khong bao gio con an nua.

USAGE
-----
    python scripts/inspect_preds.py LongBench/pred/<config>/lcc.jsonl
    python scripts/inspect_preds.py <duong_dan> -n 10 --raw 400
    python scripts/inspect_preds.py <duong_dan> --worst 5   # 5 mau diem thap nhat
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def scored_line(prediction):
    """Ban sao chinh xac cua vong lap trong LongBench/metrics.py::code_sim_score."""
    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            return line
    return ""


def try_fuzz():
    """fuzz co tren pod (metrics.py dung), nhung khong bat buoc de xem prediction."""
    try:
        from fuzzywuzzy import fuzz
        return fuzz
    except Exception:
        pass
    try:
        from thefuzz import fuzz
        return fuzz
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="duong dan toi <dataset>.jsonl trong thu muc pred/")
    ap.add_argument("-n", type=int, default=5, help="so mau in ra")
    ap.add_argument("--raw", type=int, default=220, help="cat prediction tho o N ky tu")
    ap.add_argument("--worst", type=int, default=0,
                    help="thay vi N mau dau, in N mau co diem THAP nhat (can fuzz)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[ERROR] khong thay file: {args.path}")
        return 1

    fuzz = try_fuzz()
    rows = []
    for i, line in enumerate(open(args.path, encoding="utf-8")):
        if not line.strip():
            continue
        d = json.loads(line)
        pred = d.get("pred", "")
        gold = (d.get("answers") or [""])[0]
        chosen = scored_line(pred)
        score = fuzz.ratio(chosen, gold) / 100 if fuzz else None
        rows.append((i, pred, chosen, gold, score))

    if not rows:
        print(f"[ERROR] file rong: {args.path}")
        return 1

    if args.worst > 0:
        if fuzz is None:
            print("[!] --worst can fuzzywuzzy/thefuzz; khong co nen in N mau dau")
        else:
            rows.sort(key=lambda r: r[4])
            args.n = args.worst

    n_empty = sum(1 for r in rows if not r[2].strip())
    n_fence = sum(1 for r in rows if "```" in r[1])
    avg = (sum(r[4] for r in rows) / len(rows)) if fuzz else None

    print(f">>> {args.path}")
    print(f">>> {len(rows)} mau"
          + (f" | diem trung binh {100 * avg:.2f}" if avg is not None else ""))
    print(f">>> dong duoc cham bi RONG: {n_empty}/{len(rows)}"
          f" | prediction co markdown fence: {n_fence}/{len(rows)}")
    if n_empty > len(rows) // 4:
        print()
        print("    [!!] Qua nhieu dong cham rong. Diem so KHONG phan anh chat luong model,")
        print("         no phan anh viec model sinh sai DINH DANG. Doi model/prompt truoc,")
        print("         dung doc gi vao con so hien tai.")
    print()

    for i, pred, chosen, gold, score in rows[:args.n]:
        head = f"--- sample {i} ---"
        if score is not None:
            head += f"  (diem {100 * score:.1f})"
        print(head)
        print("  RAW   :", repr(pred[:args.raw]))
        print("  CHAM  :", repr(chosen))
        print("  DAP AN:", repr(gold))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
