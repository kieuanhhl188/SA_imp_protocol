#!/usr/bin/env python
"""
build_repobench_groups.py — gom RepoBench v1.1 thanh "MOT fixed context -> NHIEU query".

VI SAO CAN FILE NAY
-------------------
Premise cua Squeezed Attention: mot fixed context duoc DUNG LAI cho nhieu query, nho vay
chi phi clustering offline duoc khau hao. LongBench LCC va RepoBench-P deu la
**1 context <-> 1 query**, nen khong bo nao kiem duoc premise nay — claim C3 (chi phi khau
hao) hien khong co du lieu de do.

RepoBench v1.1 thi co: nhieu dong cung mot repo dung CHUNG mot bo snippet cross-file.
Do 19/8 tren toan bo du lieu: **1.646 fixed context >=16k, 6.848 query** (Python + Java),
trung vi 3 query/context, max 50.

KHOA NHOM — cho sai la sai het
------------------------------
Ngay 15/8 tung nhom theo `repo_name` roi ket luan "RepoBench khong dung duoc". Sai: mot repo
co the co NHIEU bo context khac nhau. Khoa dung la `(repo_name, noi_dung_context)`.

Khoa mac dinh o day la **`strict`**: bam chuoi context GHEP NGUYEN THU TU. Ly do: fixed
context chi dung lai duoc that su khi chuoi giong nhau TUNG BYTE — do moi la dieu kien de
KV cache dung chung. Che do `sorted` gom them duoc vai nhom nua bang cach sap lai thu tu
snippet, nhung khi do ban dang TU DUNG mot context khac voi ban benchmark phat hanh, phai
ghi ro trong bai.

VA MOT CAI BAY DA GAP: khoa nhom PHAI tu choi context rong. Bien the
`line_completion.jsonl` cua CrossCodeEval khong co cross-file context; ham trich tra chuoi
rong, `hash("")` gom het query cung repo vao mot nhom, va ti le "dung chung" nhay len
96-99,6%. Con so dep bat thuong la dau hieu khoa nhom dang do nham thu.

OUTPUT
------
    <out>/contexts.jsonl   group_id · repo_name · language · fixed_context · n_query · ...
    <out>/queries.jsonl    query_id · group_id · user_input · answer · ...

`group_id` dong vai `dataidx` cua duong ong hien co, nen `prepare_code_data.py` va
`offline_clustering_struct.py` chay duoc gan nhu khong phai sua: clustering MOT lan cho
moi group, roi pred nhieu lan tra ve cung bo centroid.

USAGE
-----
    python scripts/build_repobench_groups.py --out repobench_v11_groups
    python scripts/build_repobench_groups.py --languages python --min_level 16k --min_query 2
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Thu tu tang dan, de --min_level loc duoc
LEVELS = ["2k", "4k", "8k", "12k", "16k", "24k", "32k", "64k", "128k"]

HF_CACHE = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
if not HF_CACHE.rstrip("/").endswith("huggingface"):
    HF_CACHE = os.path.join(HF_CACHE, "")


def find_parquet(language, split):
    """Tim parquet trong cache HF. Khong goi Hub -> chay offline duoc."""
    pats = [
        os.path.join(HF_CACHE, "hub",
                     f"datasets--tianyang--repobench_{language}_v1.1",
                     "snapshots", "*", "data", f"{split}-*.parquet"),
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                     f"datasets--tianyang--repobench_{language}_v1.1",
                     "snapshots", "*", "data", f"{split}-*.parquet"),
    ]
    for p in pats:
        fs = sorted(glob.glob(p))
        if fs:
            return fs
    return []


def context_text(row, mode="strict"):
    """
    Chuoi cross-file context cua MOT dong, dung dinh dang ma RepoBench dung khi dung prompt:
    moi snippet mo dau bang comment duong dan file.

    mode='strict'  giu nguyen thu tu -> fixed context giong nhau tung byte
    mode='sorted'  sap theo path      -> gom duoc nhieu nhom hon, nhung la context TU DUNG
    """
    items = list(row["context"])
    if mode == "sorted":
        items = sorted(items, key=lambda c: str(c.get("path", "")))
    parts = []
    for c in items:
        path = str(c.get("path", ""))
        snip = str(c.get("snippet", ""))
        if not snip.strip():
            continue
        parts.append(f"# Path: {path}\n{snip}")
    return "\n\n".join(parts)


def group_key(repo_name, ctx_text):
    """
    Khoa nhom = (repo, noi dung context). TU CHOI context rong — xem ghi chu dau file.
    """
    if not ctx_text.strip():
        return None
    h = hashlib.sha1(ctx_text.encode("utf-8")).hexdigest()[:16]
    return f"{repo_name}::{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "repobench_v11_groups"))
    ap.add_argument("--languages", nargs="+", default=["python", "java"],
                    choices=["python", "java"])
    ap.add_argument("--splits", nargs="+", default=["cross_file_first"],
                    choices=["cross_file_first", "cross_file_random", "in_file"])
    ap.add_argument("--key_mode", choices=["strict", "sorted"], default="strict",
                    help="strict (MAC DINH): bam chuoi context nguyen thu tu — fixed context "
                         "giong nhau tung byte. sorted: sap lai theo path, gom duoc nhieu "
                         "nhom hon nhung la context TU DUNG, phai ghi ro trong bai")
    ap.add_argument("--min_level", default="16k", choices=LEVELS + ["none"],
                    help="chi giu dong co level >= muc nay (mac dinh 16k: fixed context dai)")
    ap.add_argument("--min_query", type=int, default=2,
                    help="chi giu nhom co it nhat bay nhieu query (mac dinh 2 — duoi do thi "
                         "khong con la 'nhieu query tren mot context')")
    args = ap.parse_args()

    import pandas as pd

    min_rank = -1 if args.min_level == "none" else LEVELS.index(args.min_level)

    rows = []
    for lang in args.languages:
        for split in args.splits:
            fs = find_parquet(lang, split)
            if not fs:
                print(f"[WARN] khong thay parquet cho {lang}/{split} trong cache HF")
                continue
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            df["__lang"] = lang
            df["__split"] = split
            rows.append(df)
            print(f">>> {lang:7s} {split:18s} {len(df):6d} dong  ({len(fs)} shard)")
    if not rows:
        raise SystemExit("[ERROR] khong doc duoc du lieu nao. Tai truoc bang:\n"
                         "        python -c \"from datasets import load_dataset; "
                         "load_dataset('tianyang/repobench_python_v1.1')\"")
    import pandas as pd
    data = pd.concat(rows, ignore_index=True)

    # ---- gom nhom ----
    groups = defaultdict(list)
    n_empty_ctx = n_short = 0
    for i in range(len(data)):
        r = data.iloc[i]
        if min_rank >= 0 and LEVELS.index(str(r["level"])) < min_rank:
            n_short += 1
            continue
        ctx = context_text(r, args.key_mode)
        k = group_key(str(r["repo_name"]), ctx)
        if k is None:
            n_empty_ctx += 1
            continue
        groups[k].append((i, ctx))

    kept = {k: v for k, v in groups.items() if len(v) >= args.min_query}

    os.makedirs(args.out, exist_ok=True)
    fc = open(os.path.join(args.out, "contexts.jsonl"), "w", encoding="utf-8")
    fq = open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8")

    nq_all, tok_all = [], []
    for gid, (k, members) in enumerate(sorted(kept.items())):
        idx0, ctx = members[0]
        r0 = data.iloc[idx0]
        fc.write(json.dumps({
            "group_id": gid,
            "repo_name": str(r0["repo_name"]),
            "language": str(r0["__lang"]),
            "split": str(r0["__split"]),
            "level": str(r0["level"]),
            "token_num": int(r0["token_num"]),
            "n_query": len(members),
            "key_mode": args.key_mode,
            "fixed_context": ctx,
        }, ensure_ascii=False) + "\n")
        nq_all.append(len(members))
        tok_all.append(int(r0["token_num"]))
        for j, (idx, _) in enumerate(members):
            r = data.iloc[idx]
            fq.write(json.dumps({
                "query_id": f"{gid}_{j}",
                "group_id": gid,
                "language": str(r["__lang"]),
                # phan file hien tai truoc con tro — protocol goi day la mot phan cua
                # fixed_context, nhung no KHAC NHAU giua cac query cung nhom, nen phai
                # de o day. Xem ghi chu "user_input" duoi.
                "in_file_prefix": str(r["cropped_code"]),
                "import_statement": str(r["import_statement"]),
                "user_input": str(r["next_line"]),
                "answer": str(r["next_line"]),
                "gold_snippet_index": int(r["gold_snippet_index"]),
            }, ensure_ascii=False) + "\n")
    fc.close()
    fq.close()

    import statistics as S
    print()
    print(f">>> Da doc {len(data)} dong")
    print(f"    bo vi level < {args.min_level}: {n_short}")
    print(f"    bo vi context rong           : {n_empty_ctx}")
    print(f">>> Nhom tim duoc            : {len(groups)}")
    print(f">>> Nhom co >= {args.min_query} query        : {len(kept)}")
    if nq_all:
        print(f"    query/nhom : trung vi {S.median(nq_all):.0f} · tb {S.mean(nq_all):.1f} "
              f"· max {max(nq_all)}")
        print(f"    token_num  : trung vi {S.median(tok_all):.0f} · max {max(tok_all)}")
        print(f"    TONG query : {sum(nq_all)}")
    print()
    print(f">>> Da ghi {args.out}/contexts.jsonl  ({len(kept)} dong)")
    print(f">>> Da ghi {args.out}/queries.jsonl   ({sum(nq_all)} dong)")
    print()
    print("    LUU Y cho bai: trung vi query/context o day thap hon nhieu so voi ~24 cua")
    print("    PreFixQA trong bai goc. Muc khau hao chi phi clustering vi vay khiem ton hon,")
    print("    khong duoc noi qua.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
