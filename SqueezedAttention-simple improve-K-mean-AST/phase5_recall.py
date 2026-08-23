"""
phase5_recall.py — Phase 5 (C2): recall@budget cua tap key duoc chon.

BANG CHUNG TRUC TIEP VA RE NHAT CHO H0
--------------------------------------
Protocol xep Phase 5 chay TRUOC Phase 6: *"Neu C2 fail thi H0 sai -> dung, khong chay
C1/C3"*. Ly do no re: khong sinh text. Chi mot luot forward de lay q/k, roi toan bo phan
con lai la nhan ma tran. Buoc [5] cua Phase 6 (sinh text voi Sq-70%) do duoc la 19,3
giay/mau; o day khong co buoc do.

DO CAI GI
---------
Voi moi (lop, head, truy van):

    a[j]  = trong so attention THAT cua key j   (softmax cua q.k)
    K*    = N key co a lon nhat                  <- tap ly tuong
    s_m[j]= diem ma phuong phap m gan cho key j  = q . centroid[label[j]]
    K_m   = N key co s_m lon nhat                <- tap phuong phap m chon

    Recall@budget = |K_m giao K*| / N
    Mass@budget   = tong a tren K_m / tong a tren K*

N = (1 - sparsity) * n_ctx, giong nhau cho MOI phuong phap -> so cung ngan sach.

VI SAO XEP HANG THAY VI DUNG NGUONG
-----------------------------------
Cach cai dat that (`centroid_lookup`) chon cluster bang mot NGUONG toan cuc, va nguong do
duoc hieu chinh rieng cho tung lan chay. Neu do theo nguong thi ket qua lan lon hai thu:
chat luong XEP HANG cua centroid, va do chinh xac cua HIEU CHINH NGUONG. C2 hoi ve thu
nhat. Cat top-N cho ca ba nhanh o cung N loai bo hoan toan yeu to thu hai.

He qua phai ghi ro khi bao cao: **day la can tren cua recall thuc te**. Cai dat that con
mat them mot phan do nguong khong hoan hao. Nhung sai so do la NHU NHAU cho moi nhanh, nen
so sanh giua cac nhanh van hop le.

MOT DIEU CAN THIET NUA
----------------------
`hard_boundary` dung HET ngan sach centroid con `sa` bo trong 0,71% o (do o Phase 2). O
cung K danh nghia, hai ben co so cluster HIEU DUNG khac nhau. Script in ca hai con so de
bao cao trung thuc, khong gop lai lam mot.

USAGE
-----
    python phase5_recall.py qwen2.5-coder-7b-instruct --force_chat --dataset lcc \\
        --cluster_dir sa=/workspace/p2/sa/lcc \\
        --cluster_dir hard_boundary=/workspace/p2/hard_boundary/lcc \\
        --sparsity 70 80 90 --limit 100 --out phase5_lcc.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from squeezedattention.utils import truncate_fn, apply_rope_scaling  # noqa: E402

JSONL_HEAD = "Please complete the code given below. " + chr(10)
JSONL_TAIL = "Next line of code:" + chr(10)


def load_jsonl_contexts(data_dir):
    path = os.path.join(data_dir, "contexts.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] khong thay {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append({"context": d["fixed_context"], "input": "",
                        "language": d.get("language", "python")})
    return out


def recall_one_sample(q, k, cents, labels, sparsity_list):
    """
    q:      [H, Q, D]  truy van cua observation window
    k:      [H, S, D]  key cua fixed context
    cents:  [H, K, D]  centroid
    labels: [H, S]     key j thuoc cluster nao

    Tra ve {sparsity: (recall, mass)} — trung binh tren moi head va truy van.
    """
    H, S, D = k.shape
    sm = 1.0 / (D ** 0.5)

    # attention THAT
    attn = torch.softmax((q @ k.transpose(-1, -2)) * sm, dim=-1)      # [H, Q, S]

    # diem cua phuong phap: key j nhan diem cua centroid chua no
    cs = (q @ cents.transpose(-1, -2)) * sm                            # [H, Q, K]
    idx = labels.unsqueeze(1).expand(H, q.shape[1], S)                 # [H, Q, S]
    smeth = torch.gather(cs, 2, idx)                                   # [H, Q, S]

    out = {}
    for sp in sparsity_list:
        N = max(1, int(round((1.0 - sp / 100.0) * S)))
        star = attn.topk(N, dim=-1).indices                            # [H, Q, N]
        mine = smeth.topk(N, dim=-1).indices

        # |giao| bang cach danh dau tren mask boolean — nhanh hon so tung cap
        m_star = torch.zeros_like(attn, dtype=torch.bool).scatter_(2, star, True)
        m_mine = torch.zeros_like(attn, dtype=torch.bool).scatter_(2, mine, True)
        inter = (m_star & m_mine).sum(-1).float()                      # [H, Q]
        recall = (inter / N).mean().item()

        mass_star = (attn * m_star).sum(-1)
        mass_mine = (attn * m_mine).sum(-1)
        mass = (mass_mine / mass_star.clamp_min(1e-9)).mean().item()
        out[sp] = (recall, mass)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cluster_dir", action="append", required=True,
                    help="dang TEN=/duong/dan, lap lai cho tung nhanh")
    ap.add_argument("--dataset", default="lcc")
    ap.add_argument("--data_source", choices=["longbench", "jsonl"], default="longbench")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--fixed_context", choices=["full", "crossfile"], default="full")
    ap.add_argument("--force_chat", action="store_true")
    ap.add_argument("--rope_scaling", default=None)
    ap.add_argument("--sparsity", type=int, nargs="+", default=[70, 80, 90])
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--layers", default="first,mid,last",
                    help="'first,mid,last' hoac danh sach so '0,13,27'. Tinh tren MOI lop "
                         "rat cham ma khong doi ket luan")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="phase5_recall.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset

    branches = {}
    for spec in args.cluster_dir:
        name, d = spec.split("=", 1)
        if not os.path.isdir(d):
            raise SystemExit(f"[ERROR] khong phai thu muc: {d}")
        branches[name] = d
    print(">>> nhanh:", ", ".join(f"{k} -> {v}" for k, v in branches.items()))

    DEV = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    m2p = json.load(open("LongBench/config/model2path.json", encoding="utf-8"))
    m2l = json.load(open("LongBench/config/model2maxlen.json", encoding="utf-8"))
    d2p = json.load(open("LongBench/config/dataset2prompt.json", encoding="utf-8"))
    model_path, max_length = m2p[args.model], m2l[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    config = apply_rope_scaling(config, args.rope_scaling)
    config.return_qkv_states = True
    config._attn_implementation = "flash_attention_2"
    if getattr(config, "use_sliding_window", False):
        config.use_sliding_window = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16).eval().to(DEV)

    if args.data_source == "jsonl":
        data = load_jsonl_contexts(args.data_dir)
        prompt_format = JSONL_HEAD + "{context}" + JSONL_TAIL
        prompt_only_format = JSONL_HEAD + "{context}"
        ds_trunc = "lcc"
    else:
        prompt_format = d2p[args.dataset]
        key_only = (args.dataset + "_prompt_full" if args.fixed_context == "full"
                    else args.dataset + "_prompt")
        prompt_only_format = d2p[key_only]
        data = load_dataset("THUDM/LongBench", args.dataset, split="test")
        ds_trunc = args.dataset

    meta = {}
    for cand in (os.path.join(args.phase1_dir, args.model), args.phase1_dir):
        mp = os.path.join(cand, f"{args.dataset}_meta.jsonl")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    meta[d["dataidx"]] = d
            break
    if not meta:
        raise SystemExit("[ERROR] thieu meta Phase 1.4")

    all_q, all_k = [], []
    state = {"sp": 0}

    def hook(module, inp, out):
        _, qkv, _ = out
        q, k, _v = qkv
        all_q.append(q[:, :, :state["sp"]])
        all_k.append(k[:, :, :state["sp"]])

    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(hook)

    n = min(args.limit, len(data)) if args.limit > 0 else len(data)
    res = {b: {sp: {"recall": [], "mass": []} for sp in args.sparsity} for b in branches}
    n_used = 0

    for i in tqdm(range(n)):
        rec = meta.get(i)
        if rec is None:
            continue
        d = data[i]
        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, sp_len = truncate_fn(prompt, prompt_only, tokenizer, max_length, ds_trunc,
                                     DEV, model_name=args.model, force_chat=args.force_chat)
        if sp_len != rec["shared_prefix_length"]:
            raise SystemExit(f"[ERROR] dataidx {i}: sp_len lech voi Phase 1.4 "
                             f"({sp_len} vs {rec['shared_prefix_length']})")
        state["sp"] = sp_len
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear(); all_k.clear()
        with torch.no_grad():
            model.generate(ids, do_sample=False, max_new_tokens=1, use_cache=False,
                           output_attentions=True)

        n_ctx = sp_len - args.observation_window
        if n_ctx <= 0:
            continue
        L = len(all_k)
        layers = sorted({0, L // 2, L - 1}) if args.layers == "first,mid,last" \
            else [int(x) for x in args.layers.split(",")]

        ok = True
        for b, path in branches.items():
            import glob
            f = glob.glob(os.path.join(path, f"centroids_tensor_dict_{i}_*.pt"))
            if not f:
                ok = False
                break
        if not ok:
            continue

        for b, path in branches.items():
            import glob, re
            f = glob.glob(os.path.join(path, f"centroids_tensor_dict_{i}_*.pt"))[0]
            K = int(re.search(r"_(\d+)\.pt$", f).group(1))
            cent = torch.load(f, map_location=DEV)
            lab = torch.load(os.path.join(path, f"centroids_labels_dict_{i}_{K}.pt"),
                             map_location=DEV)
            for l in layers:
                # GQA: hook lay q/k TRUOC repeat_kv nen q co H_q head (28) con k,
                # centroid, label deu theo H_kv (4). Appendix G cua bai quy dinh moi
                # query head TU CHON key rieng, nen nhan ban ca ba thu len H_q —
                # dung `repeat_interleave` giong `expand_kv_heads_to_query_heads`
                # cua fork, khong tu che cach khac.
                q = all_q[l].squeeze(0).float()[:, n_ctx:sp_len, :]     # [H_q, Q, D]
                k = all_k[l].squeeze(0).float()[:, :n_ctx, :]           # [H_kv, S, D]
                c = cent[l].squeeze(0).float()                          # [H_kv, K, D]
                lb = lab[l].squeeze(0)[:, :n_ctx].long()                # [H_kv, S]
                if q.shape[0] != k.shape[0]:
                    rep = q.shape[0] // k.shape[0]
                    assert q.shape[0] == k.shape[0] * rep, (q.shape, k.shape)
                    k = k.repeat_interleave(rep, dim=0)
                    c = c.repeat_interleave(rep, dim=0)
                    lb = lb.repeat_interleave(rep, dim=0)
                assert c.shape[0] == k.shape[0] == lb.shape[0] == q.shape[0], (
                    q.shape, k.shape, c.shape, lb.shape)
                out = recall_one_sample(q, k, c, lb, args.sparsity)
                for sp, (r, m) in out.items():
                    res[b][sp]["recall"].append(r)
                    res[b][sp]["mass"].append(m)
        n_used += 1

    print(f"\n>>> Do tren {n_used} mau · lop {args.layers} · sparsity {args.sparsity}")
    print(f"{'nhanh':20s} " + " ".join(f"{'sp'+str(s):>18s}" for s in args.sparsity))
    summary = {}
    for b in branches:
        row = []
        summary[b] = {}
        for sp in args.sparsity:
            r = float(np.mean(res[b][sp]["recall"]))
            m = float(np.mean(res[b][sp]["mass"]))
            summary[b][sp] = {"recall": r, "mass": m,
                              "n": len(res[b][sp]["recall"])}
            row.append(f"{r*100:7.2f}% / {m*100:6.2f}%")
        print(f"{b:20s} " + " ".join(f"{x:>18s}" for x in row))
    print("   (recall / attention-mass, cang cao cang tot)")

    json.dump({"model": args.model, "dataset": args.dataset, "n_samples": n_used,
               "layers": args.layers, "sparsity": args.sparsity,
               "summary": summary,
               "per_sample": {b: {str(s): res[b][s]["recall"] for s in args.sparsity}
                              for b in branches}},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f">>> Da ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
