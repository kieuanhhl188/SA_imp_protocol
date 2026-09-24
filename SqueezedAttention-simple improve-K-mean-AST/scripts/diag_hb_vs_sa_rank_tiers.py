"""
diag_hb_vs_sa_rank_tiers.py — chan doan CHI DOC: recall cua HardBoundary hon SA la nho lay them
key attention CAO hay key attention THAP?

KHONG sua clustering, KHONG sua phase5_recall.py. Dung dung: cung forward q/k, cung centroid
(doc thang tu thu muc, chi doc), cung diem `q . centroid[label[j]]`, cung N = round((1-sp)*S),
cung trung binh tren head va query. Chi them phan RA: phan ra recall/mass theo BAC attention
that cua K*.

Voi moi (mau, lop, sp, nhanh) va moi hang (head, query):
    K*   = N key co attention that lon nhat, sap giam dan        (rank 0 = manh nhat)
    K_m  = N key co diem cua phuong phap lon nhat
    hit[p] = 1 neu key hang p cua K* nam trong K_m
Bac cua K*:  A = top 10% (rank < 0.1N),  B = 10%..50%,  C = 50%..100% (bien, attention thap nhat trong K*)
    recall_T = trung binh hit tren bac T
    mass_T   = sum_{p in T} a[p]*hit[p] / mass(K*)          (phan mass cua giao K_m & K* thuoc bac T)
    extra    = mass(K_m \ K*) / mass(K*)                     (mass cua key ngoai K* ma phuong phap chon)
  => mass_ratio = mass_A + mass_B + mass_C + extra   (dung bang mass cua phase5_recall.py)
  => recall     = 0.1*recall_A + 0.4*recall_B + 0.5*recall_C  (xap xi theo do dai bac)
Kiem chung: recall & mass tong hop phai TRUNG phase5_recall.py tung mau/tung lop (assert).

USAGE
    python scripts/diag_hb_vs_sa_rank_tiers.py longchat-v1.5-7b-32k --dataset repobench-p \\
        --cluster_dir sa=... --cluster_dir hard_boundary_class=... --limit 200 --out diag.json
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from squeezedattention.utils import truncate_fn, apply_rope_scaling  # noqa: E402

TIERS = (("A", 0.0, 0.1), ("B", 0.1, 0.5), ("C", 0.5, 1.0))


@torch.no_grad()
def decompose(q, k, cents, labels, sparsity_list):
    H, S, D = k.shape
    sm = 1.0 / (D ** 0.5)
    attn = torch.softmax((q @ k.transpose(-1, -2)) * sm, dim=-1)              # [H,Q,S]
    cs = (q @ cents.transpose(-1, -2)) * sm
    smeth = torch.gather(cs, 2, labels.unsqueeze(1).expand(H, q.shape[1], S))  # [H,Q,S]
    out = {}
    for sp in sparsity_list:
        N = max(1, int(round((1.0 - sp / 100.0) * S)))
        star_v, star_i = attn.topk(N, dim=-1)                                  # sorted desc
        mine_i = smeth.topk(N, dim=-1).indices
        m_mine = torch.zeros_like(attn, dtype=torch.bool).scatter_(2, mine_i, True)
        hit = m_mine.gather(2, star_i)                                         # [H,Q,N] theo bac
        mass_star = star_v.sum(-1).clamp_min(1e-9)                             # [H,Q]
        mass_mine = (attn * m_mine).sum(-1)
        inter_mass = (star_v * hit).sum(-1)
        r = {"recall": hit.float().mean(-1).mean().item(),
             "mass": (mass_mine / mass_star).mean().item(),
             "extra": ((mass_mine - inter_mass) / mass_star).mean().item()}
        bA, bB = max(1, int(np.ceil(0.1 * N))), max(2, int(np.ceil(0.5 * N)))
        bounds = {"A": (0, bA), "B": (bA, bB), "C": (bB, N)}
        for name in ("A", "B", "C"):
            a, b = bounds[name]
            r[f"recall_{name}"] = hit[..., a:b].float().mean(-1).mean().item()
            r[f"mass_{name}"] = ((star_v[..., a:b] * hit[..., a:b]).sum(-1) / mass_star).mean().item()
            r[f"star_share_{name}"] = (star_v[..., a:b].sum(-1) / mass_star).mean().item()  # mass K* o bac nay
        # attention trung binh cua 1 key "thua" (thuoc K_m nhung ngoai K*) so voi key hang N cua K*
        n_extra = (N - hit.sum(-1)).clamp_min(0).float()                       # [H,Q]
        extra_mass_abs = (mass_mine - inter_mass)
        thr = star_v[..., -1].clamp_min(1e-12)
        ok = n_extra > 0
        r["extra_over_thr"] = ((extra_mass_abs / n_extra.clamp_min(1)) / thr)[ok].mean().item() if ok.any() else float("nan")
        r["n_extra_frac"] = (n_extra / N).mean().item()
        out[sp] = r
        del star_v, star_i, mine_i, m_mine, hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cluster_dir", action="append", required=True)
    ap.add_argument("--dataset", default="repobench-p")
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--sparsity", type=int, nargs="+", default=[70, 80, 90])
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset
    branches = dict(s.split("=", 1) for s in args.cluster_dir)
    DEV = torch.device(f"cuda:{args.device}")
    m2p = json.load(open("LongBench/config/model2path.json", encoding="utf-8"))
    m2l = json.load(open("LongBench/config/model2maxlen.json", encoding="utf-8"))
    d2p = json.load(open("LongBench/config/dataset2prompt.json", encoding="utf-8"))
    model_path, max_length = m2p[args.model], m2l[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    config = apply_rope_scaling(config, None)
    config.return_qkv_states = True
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                 torch_dtype=torch.bfloat16).eval().to(DEV)
    prompt_format = d2p[args.dataset]
    prompt_only_format = d2p[args.dataset + "_prompt_full"]
    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    meta = {}
    for cand in (os.path.join(args.phase1_dir, args.model), args.phase1_dir):
        mp = os.path.join(cand, f"{args.dataset}_meta.jsonl")
        if os.path.exists(mp):
            for line in open(mp, encoding="utf-8"):
                d = json.loads(line); meta[d["dataidx"]] = d
            break
    all_q, all_k, state = [], [], {"sp": 0}

    def hook(module, inp, out):
        _, qkv, _ = out
        q, k, _v = qkv
        all_q.append(q[:, :, :state["sp"]]); all_k.append(k[:, :, :state["sp"]])
    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(hook)

    res = {b: {} for b in branches}      # b -> sp -> key -> list [n*3]
    idxs = []
    for i in tqdm(range(min(args.limit, len(data)))):
        rec = meta[i]
        prompt = prompt_format.format(**data[i]); prompt_only = prompt_only_format.format(**data[i])
        prompt, sp_len = truncate_fn(prompt, prompt_only, tokenizer, max_length, args.dataset, DEV,
                                     model_name=args.model, force_chat=False)
        assert sp_len == rec["shared_prefix_length"]
        state["sp"] = sp_len
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear(); all_k.clear()
        with torch.no_grad():
            model.model(input_ids=ids, use_cache=False)
        n_ctx = sp_len - args.observation_window
        L = len(all_k); layers = sorted({0, L // 2, L - 1})
        for b, path in branches.items():
            f = glob.glob(os.path.join(path, f"centroids_tensor_dict_{i}_*.pt"))[0]
            K = int(re.search(r"_(\d+)\.pt$", f).group(1))
            cent = torch.load(f, map_location=DEV)
            lab = torch.load(os.path.join(path, f"centroids_labels_dict_{i}_{K}.pt"), map_location=DEV)
            for l in layers:
                q = all_q[l].squeeze(0).float()[:, n_ctx:sp_len, :]
                k = all_k[l].squeeze(0).float()[:, :n_ctx, :]
                c = cent[l].squeeze(0).float(); lb = lab[l].squeeze(0)[:, :n_ctx].long()
                assert q.shape[0] == k.shape[0] == c.shape[0] == lb.shape[0]   # LongChat = MHA
                out = decompose(q, k, c, lb, args.sparsity)
                for sp, r in out.items():
                    cell = res[b].setdefault(str(sp), {})
                    for key, v in r.items():
                        cell.setdefault(key, []).append(v)
        idxs.append(i)
        all_q.clear(); all_k.clear()
        if len(idxs) % 20 == 0:
            torch.cuda.empty_cache()
    json.dump({"model": args.model, "dataset": args.dataset, "idx": idxs, "layers": "first,mid,last",
               "sparsity": args.sparsity, "tiers": {n: [lo, hi] for n, lo, hi in TIERS},
               "branches": res}, open(args.out, "w"))
    print(">>> Da ghi", args.out)


if __name__ == "__main__":
    main()
