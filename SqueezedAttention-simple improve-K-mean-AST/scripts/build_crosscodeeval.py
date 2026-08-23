#!/usr/bin/env python
"""
build_crosscodeeval.py — dua CrossCodeEval ve dinh dang chung cua du an.

VI SAO CAN FILE NAY
-------------------
Protocol nhac CrossCodeEval **ba lan**, khong chi o Phase 1:

    Phase 1:  "CrossCodeEval va RepoEval / RepoBench — completion cap repository"
    Phase 6:  "full grid: {models} x {datasets: RB, LCC, CrossCodeEval, RepoPreFixQA} x ..."
    Bang chinh: "Accuracy@matched-budget ... tren RB+LCC+CrossCodeEval+RepoPreFixQA"

Bo no la **hong mot cot cua bang ket qua chinh** — bang dung de chung minh C1.

DIEU PHAI GHI RO KHI BAO CAO
----------------------------
Protocol dua bo nay vao voi ly do *"dung fixed context dai"*. Do lai 9/9 bien the (3 loai
x 3 retriever, 4 ngon ngu): **tran ~10,2K token**, p95 chi ~3.700 ky tu. Ly do do KHONG
dung. Bo nay van cho mot cot accuracy@budget hop le, nhung **khong dung de chung minh
long-context**. Xem docs/PHASE1_DATASETS.md.

Va no gan nhu **1 query / 1 context**: nhom theo (repo, noi dung context) chi cho 7,1%
(Python) den 37,8% (TypeScript) so nhom co >=2 query. Nen no khong kiem duoc premise cua SA.

HAI BIEN THE RETRIEVAL
----------------------
    retrieval      = rg1     — retrieve khong tham chieu dap an (mac dinh, cong bang)
    retrievalwref  = oracle  — retrieve CO tham chieu dap an, tran tren cua retrieval

Chon `retrieval` lam mac dinh: `retrievalwref` dung dap an de chon chunk nen la thuong
can tren, dung no lam ket qua chinh se thoi phong diem.

DINH DANG RA — giong build_repobench_groups.py de dung chung duong ong
---------------------------------------------------------------------
    <out>/contexts.jsonl   group_id · language · fixed_context · n_query · ...
    <out>/queries.jsonl    query_id · group_id · user_input · answer · ...

`fixed_context` = **cross-file context + phan file hien tai truoc con tro**, dung dinh
nghia cua protocol (che do `full`). Vi `prompt` khac nhau giua cac mau nen moi mau la mot
context rieng — do la ban chat cua bo nay, khong phai loi cau hinh. Muon nhom chung thi
dung --fixed_context crossfile.

USAGE
-----
    python scripts/build_crosscodeeval.py --out crosscodeeval_data
    python scripts/build_crosscodeeval.py --languages python java --variant retrievalwref
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

LANGS = ["python", "java", "csharp", "typescript"]
VARIANTS = {"retrieval": "crossfile_context_retrieval",
            "retrievalwref": "crossfile_context_retrievalwref"}


def find_parquet(language):
    pats = [
        os.path.expanduser(f"~/.cache/huggingface/hub/datasets--ZHENGRAN--"
                           f"cross_code_eval_{language}/snapshots/*/data/*.parquet"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub",
                     f"datasets--ZHENGRAN--cross_code_eval_{language}",
                     "snapshots", "*", "data", "*.parquet"),
    ]
    for p in pats:
        fs = sorted(glob.glob(p))
        if fs:
            return fs
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "crosscodeeval_data"))
    ap.add_argument("--languages", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="retrieval",
                    help="retrieval (MAC DINH, cong bang) hoac retrievalwref (oracle, dung "
                         "dap an de retrieve -> thuong can tren, khong dung lam ket qua chinh)")
    ap.add_argument("--context_format", choices=["raw", "commented"], default="raw",
                    help="raw (MAC DINH): ghep tu truong `retrieved_chunk`, la CODE THAT. commented: dung truong `text` cua bo goc, trong do MOI DONG bi them dau # -> tree-sitter chi thay comment, khong con don vi cau truc nao. Do that: che do commented lam 65% mau chi con <=2 unit, tuc Idea 1 khong co gi de rang buoc")
    ap.add_argument("--fixed_context", choices=["full", "crossfile"], default="full",
                    help="full (MAC DINH): cross-file + phan file truoc con tro, dung dinh "
                         "nghia protocol. crossfile: chi cross-file, de nhieu query dung "
                         "chung mot context")
    args = ap.parse_args()

    import pandas as pd

    col = VARIANTS[args.variant]
    groups = defaultdict(list)      # key -> [(fixed_context, row_dict)]
    n_rows = n_empty = 0
    per_lang = {}

    for lang in args.languages:
        fs = find_parquet(lang)
        if not fs:
            print(f"[WARN] khong thay parquet cho {lang} trong cache HF")
            continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        per_lang[lang] = len(df)
        print(f">>> {lang:11s} {len(df):5d} dong")
        for i in range(len(df)):
            r = df.iloc[i]
            n_rows += 1
            cf = r[col]
            if args.context_format == "commented":
                cf_text = str(cf.get("text", "")) if hasattr(cf, "get") else ""
            else:
                parts = []
                for c in cf.get("list", []):
                    chunk = str(c.get("retrieved_chunk", ""))
                    if not chunk.strip():
                        continue
                    parts.append("# Path: " + str(c.get("filename", "")) + chr(10) + chunk)
                cf_text = (chr(10) + chr(10)).join(parts)
            if not cf_text.strip():
                # Bay da gap: bien the khong co cross-file context tra chuoi rong, roi
                # hash("") gom het query cung repo vao mot nhom -> ti le "dung chung" nhay
                # len 96-99,6%. Tu choi tu day.
                n_empty += 1
                continue
            infile = str(r["prompt"])
            fixed = cf_text + "\n" + infile if args.fixed_context == "full" else cf_text
            meta = r["metadata"]
            repo = str(meta.get("repository", ""))
            h = hashlib.sha1(fixed.encode("utf-8")).hexdigest()[:16]
            groups[f"{lang}::{repo}::{h}"].append((fixed, {
                "language": lang,
                "repository": repo,
                "task_id": str(meta.get("task_id", "")),
                "file": str(meta.get("file", "")),
                "user_input": str(r["groundtruth"]),
                "answer": str(r["groundtruth"]),
            }))

    if not groups:
        raise SystemExit("[ERROR] khong doc duoc du lieu nao. Tai truoc bang:\n"
                         "        python -c \"from datasets import load_dataset; "
                         "load_dataset('ZHENGRAN/cross_code_eval_python')\"")

    os.makedirs(args.out, exist_ok=True)
    fc = open(os.path.join(args.out, "contexts.jsonl"), "w", encoding="utf-8")
    fq = open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8")

    nq, lens = [], []
    for gid, (k, members) in enumerate(sorted(groups.items())):
        fixed = members[0][0]
        fc.write(json.dumps({
            "group_id": gid,
            "language": members[0][1]["language"],
            "repository": members[0][1]["repository"],
            "variant": args.variant,
            "context_format": args.context_format,
            "fixed_context_mode": args.fixed_context,
            "n_query": len(members),
            "n_chars": len(fixed),
            "fixed_context": fixed,
        }, ensure_ascii=False) + "\n")
        nq.append(len(members))
        lens.append(len(fixed))
        for j, (_, q) in enumerate(members):
            fq.write(json.dumps({"query_id": f"{gid}_{j}", "group_id": gid, **q},
                                ensure_ascii=False) + "\n")
    fc.close()
    fq.close()

    import statistics as S
    shared = sum(1 for x in nq if x >= 2)
    print()
    print(f">>> Doc {n_rows} dong · bo {n_empty} dong khong co cross-file context")
    print(f">>> Context : {len(nq)}   |   query: {sum(nq)}")
    print(f"    query/context : trung vi {S.median(nq):.0f} · max {max(nq)} "
          f"· nhom >=2 query: {shared} ({100*shared/len(nq):.1f}%)")
    print(f"    do dai (ky tu): trung vi {S.median(lens):.0f} · p95 "
          f"{sorted(lens)[int(.95*len(lens))]} · max {max(lens)}")
    print(f"    uoc token (~3,5 ky tu/token): trung vi {S.median(lens)/3.5:.0f} "
          f"· max {max(lens)/3.5:.0f}")
    print()
    print(f">>> Da ghi {args.out}/contexts.jsonl · {args.out}/queries.jsonl")
    print()
    print("    NHAC LAI de khong doc nham: bo nay co context NGAN (tran ~10,2K token) va")
    print("    gan nhu 1 query/context. Dung cho cot accuracy@budget cua bang chinh,")
    print("    KHONG dung de chung minh long-context hay premise khau hao chi phi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
