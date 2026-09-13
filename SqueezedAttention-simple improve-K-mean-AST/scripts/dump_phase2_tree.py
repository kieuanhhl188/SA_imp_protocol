#!/usr/bin/env python
"""
dump_phase2_tree.py — in ra dang NGUOI DOC DUOC cau truc cluster cua mot mau Phase 2.

Bat bien [A] cua check_phase2_invariants.py chi cho ra con so "0.0% vat bien". File nay
cho ra CAI CAY dang sau con so do:

    function `def build_l1_groups(...)`   (unit 3, 412 token)
      ├── cluster 17   (128 token)
      ├── cluster 44   ( 91 token)
      └── cluster 59   (193 token)
    function `def merge_units_to_budget(...)`   (unit 4, 88 token)
      └── cluster 12   ( 88 token)

Voi `--method sa` se thay cluster nam vat qua nhieu function (in do), voi
`--method hard_boundary` / `struct_hierarchy` thi khong.

Chay CPU, khong can GPU. Dung chung helper voi check_phase2_invariants.py nen chuoi
prompt/unit_id ra y het bat bien [A].

USAGE
-----
    python scripts/dump_phase2_tree.py \\
        --cluster_dir /workspace/p2-longchat/hard_boundary/lcc \\
        --model longchat-v1.5-7b-32k --dataset lcc \\
        --phase1_dir /workspace/phase1_data/longchat-v1.5-7b-32k \\
        --method hard_boundary --dataidx 0 --layer 0 --head 0

    # doi chieu ba nhanh cung mot mau:
    for M in sa hard_boundary struct_hierarchy; do
      python scripts/dump_phase2_tree.py --cluster_dir /workspace/p2-longchat/$M/lcc \\
        --model longchat-v1.5-7b-32k --dataset lcc \\
        --phase1_dir /workspace/phase1_data/longchat-v1.5-7b-32k \\
        --method $M --dataidx 0 --max_units 8
    done
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402
from check_phase2_invariants import (  # noqa: E402
    discover, load_meta, rebuild_prompts,
)
from struct_clustering import parse_units, assign_token_units  # noqa: E402

RED = "\033[31m"
DIM = "\033[2m"
RST = "\033[0m"


def first_line(code, s, e):
    seg = code[s:e].strip().splitlines()
    return (seg[0][:70] if seg else "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="lcc")
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--method", default="hard_boundary")
    ap.add_argument("--dataidx", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--level", default="function")
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--max_units", type=int, default=12, help="in toi da bao nhieu unit")
    ap.add_argument("--no_color", action="store_true")
    args = ap.parse_args()

    if args.no_color or not sys.stdout.isatty():
        globals()["RED"] = globals()["DIM"] = globals()["RST"] = ""

    found = discover(args.cluster_dir)
    if args.dataidx not in found:
        raise SystemExit(f"khong co dataidx {args.dataidx} trong {args.cluster_dir}")
    K = found[args.dataidx]

    meta = load_meta(args.phase1_dir, args.dataset)
    rec = meta[args.dataidx]
    npz_path = os.path.join(args.phase1_dir, f"{args.dataset}_offsets.npz")
    import numpy as np
    npz = np.load(npz_path)
    offs = npz[f"offsets_{args.dataidx}"]

    n_ctx = rec["shared_prefix_length"] - args.observation_window
    prompts = rebuild_prompts(args.model, args.dataset, {args.dataidx}, meta)
    prompt, sp = prompts[args.dataidx]
    assert sp == rec["shared_prefix_length"], (sp, rec["shared_prefix_length"])

    # --- span cac unit (function), theo ky tu, khop offset Phase 1.4 ---
    code = prompt[rec["code_char_start"]:rec["code_char_end"]]
    raw_spans, _ = parse_units(code, rec["language"], args.level)
    spans = [(s + rec["code_char_start"], e + rec["code_char_start"]) for s, e in raw_spans]
    spans.append((0, len(prompt) + 1))                       # span phu toan bo (unit "ngoai ham")
    starts = torch.from_numpy(offs[:n_ctx, 0].astype("int64"))
    raw_uid = assign_token_units(starts, spans)              # [n_ctx] -> chi so trong `spans`

    used = sorted(set(int(u) for u in raw_uid.tolist()))
    span_tok = {u: int((raw_uid == u).sum()) for u in used}

    # --- nhan cluster tung token o (layer, head) ---
    lab = torch.load(os.path.join(args.cluster_dir,
                                 f"centroids_labels_dict_{args.dataidx}_{K}.pt"),
                     map_location="cpu")
    if args.layer not in lab:
        raise SystemExit(f"layer {args.layer} khong co; co: {sorted(lab)[:5]}...")
    t = lab[args.layer]
    t = t if t.dim() == 3 else t.unsqueeze(0)
    labels = t[0, args.head].to(torch.int64)                 # [n_ctx]
    assert labels.numel() == n_ctx, (labels.numel(), n_ctx)

    # cluster -> tap unit no cham, va so token trong tung (unit, cluster)
    from collections import defaultdict
    cu_tok = defaultdict(int)                                # (cluster, unit) -> n token
    cl_units = defaultdict(set)
    for c, u in zip(labels.tolist(), raw_uid.tolist()):
        cu_tok[(int(c), int(u))] += 1
        cl_units[int(c)].add(int(u))

    n_cross = sum(1 for c, us in cl_units.items() if len(us) > 1)
    n_cl = len(cl_units)

    print(f"# mau {args.dataidx} · {args.dataset} · {args.method} · layer {args.layer} head {args.head}")
    print(f"# ngon ngu {rec['language']} · n_ctx {n_ctx} token · K {K} cluster · "
          f"{len(used)} unit ({args.level})")
    print(f"# cluster vat qua >1 unit: {n_cross}/{n_cl} "
          f"({100.0*n_cross/max(n_cl,1):.1f}%)"
          + (f"  {RED}<-- pha bien{RST}" if n_cross else "  <- ranh gioi cung giu"))
    print()

    file_span_idx = len(spans) - 1
    shown = 0
    for u in sorted(used, key=lambda u: -span_tok[u]):
        if u == file_span_idx:
            label = "(token ngoai moi function: import, code top-level)"
        else:
            label = "`" + first_line(prompt, *spans[u]) + "`"
        if shown >= args.max_units:
            print(f"... con {len(used) - shown} unit nua (dung --max_units de xem het)")
            break
        shown += 1
        # cluster nao co token trong unit nay, sap theo so token giam dan
        cs = sorted((c for c in cl_units if (c, u) in cu_tok),
                    key=lambda c: -cu_tok[(c, u)])
        print(f"unit {u}  ·  {span_tok[u]} token  ·  {label}")
        for j, c in enumerate(cs):
            branch = "└──" if j == len(cs) - 1 else "├──"
            other = cl_units[c] - {u}
            tag = ""
            if other:
                tag = f"  {RED}(cluster nay CON cham unit {sorted(other)}){RST}"
            print(f"  {branch} cluster {c:4d}  ({cu_tok[(c, u)]:4d} token){tag}")
        print()


if __name__ == "__main__":
    main()
