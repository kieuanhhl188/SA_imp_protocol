"""
phase5_symbol_signal.py — Phase 3 (V1) danh gia C2 cho SA + SymbolSignal.

    S_final(cluster, query) = S_SA(cluster, query) + lambda * symbol_hit(cluster, query_identifiers)

CHI THEM. Khong sua SA, centroid/label SA, phase5_recall.py, ranh gioi AST, hierarchy. Chi doc thu muc SA.
Giao thuc C2 y het phase5_recall.py (cung forward q/k, cung 200 mau, lop first/mid/last, sp 70/80/90,
cung K*, cung N = round((1-sp)*S), cung cong thuc recall & mass). SymbolSignal chi doi XEP HANG cac
cluster SA san co; so key duoc chon van la N (assert).

lambda = 0 di qua CHINH `recall_one_sample` cua phase5_recall.py (doi chieu bit-exact tung mau/lop) va
so voi file ket qua C2 da co (kiem o buoc phan tich).

Them phan ra theo bac attention cua K* (A = top10%, B = 10-50%, C = 50-100%) nhu diag_hb_vs_sa_rank_tiers.py.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squeezedattention.utils import truncate_fn, apply_rope_scaling  # noqa: E402
from phase5_recall import recall_one_sample                            # noqa: E402  (ham C2 goc)
from symbol_signal import build_symbol_index, symbol_hit_from_labels   # noqa: E402


@torch.no_grad()
def enrichment(q, k, mpos_mask, sps):
    """Vi tri dinh danh khop co duoc attention that / K* uu ai hon nen tang khong? (ty so voi ty le co so)"""
    H, S, D = k.shape
    attn = torch.softmax((q @ k.transpose(-1, -2)) * (1.0 / (D ** 0.5)), dim=-1)
    base = float(mpos_mask.float().mean())
    r = {"base_rate": base, "attn_mass_on_matched": float(attn[:, :, mpos_mask].sum(-1).mean())}
    r["attn_enrichment"] = r["attn_mass_on_matched"] / max(base, 1e-12)
    for sp in sps:
        N = max(1, int(round((1.0 - sp / 100.0) * S)))
        star = attn.topk(N, dim=-1).indices
        frac = float(mpos_mask[star].float().mean())
        r[f"kstar_frac_matched_sp{sp}"] = frac
        r[f"kstar_enrichment_sp{sp}"] = frac / max(base, 1e-12)
    return r


@torch.no_grad()
def eval_lambdas(q, k, cents, labels, bonus, lams, sparsity_list):
    """
    Cung phep tinh voi phase5_recall.recall_one_sample, nhung xep hang cluster bang
    S_SA + lam * bonus. attn / K* tinh MOT lan. Tra ve {lam: {sp: {metric: value}}}.
    """
    H, S, D = k.shape
    Q = q.shape[1]
    sm = 1.0 / (D ** 0.5)
    attn = torch.softmax((q @ k.transpose(-1, -2)) * sm, dim=-1)              # [H,Q,S]
    cs0 = (q @ cents.transpose(-1, -2)) * sm                                   # [H,Q,K]  = S_SA
    idx = labels.unsqueeze(1).expand(H, Q, S)
    stats_scale = float(cs0.std(dim=-1).mean())                               # do lon S_SA giua cac cluster
    out = {lam: {} for lam in lams}
    for sp in sparsity_list:
        N = max(1, int(round((1.0 - sp / 100.0) * S)))
        star_v, star_i = attn.topk(N, dim=-1)                                  # K* (giong goc)
        m_star = torch.zeros_like(attn, dtype=torch.bool).scatter_(2, star_i, True)
        mass_star = (attn * m_star).sum(-1)
        bA, bB = max(1, int(np.ceil(0.1 * N))), max(2, int(np.ceil(0.5 * N)))
        bounds = {"A": (0, bA), "B": (bA, bB), "C": (bB, N)}
        for lam in lams:
            cs = cs0 if lam == 0 else cs0 + lam * bonus.unsqueeze(1)          # [H,Q,K]
            smeth = torch.gather(cs, 2, idx)
            mine = smeth.topk(N, dim=-1).indices
            m_mine = torch.zeros_like(attn, dtype=torch.bool).scatter_(2, mine, True)
            assert bool((m_mine.sum(-1) == N).all()), "so key duoc chon != N"
            inter = (m_star & m_mine).sum(-1).float()
            recall = (inter / N).mean().item()
            mass_mine = (attn * m_mine).sum(-1)
            mass = (mass_mine / mass_star.clamp_min(1e-9)).mean().item()
            r = {"recall": recall, "mass": mass}
            # phan ra theo bac (chan doan)
            hit = m_mine.gather(2, star_i)                                     # [H,Q,N] theo bac attention
            ms = star_v.sum(-1).clamp_min(1e-9)
            inter_mass = (star_v * hit).sum(-1)
            r["extra"] = ((mass_mine - inter_mass) / ms).mean().item()
            for name in ("A", "B", "C"):
                a, b = bounds[name]
                r[f"recall_{name}"] = hit[..., a:b].float().mean(-1).mean().item()
                r[f"mass_{name}"] = ((star_v[..., a:b] * hit[..., a:b]).sum(-1) / ms).mean().item()
                r[f"star_share_{name}"] = (star_v[..., a:b].sum(-1) / ms).mean().item()
            n_extra = (N - hit.sum(-1)).clamp_min(0).float()
            r["n_extra_frac"] = (n_extra / N).mean().item()
            out[lam][sp] = r
            del smeth, mine, m_mine, hit
        del star_v, star_i, m_star
    return out, stats_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sa_dir", required=True)
    ap.add_argument("--dataset", default="repobench-p")
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--sparsity", type=int, nargs="+", default=[70, 80, 90])
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--control", choices=["shuffle"], default=None,
                    help="CHAN DOAN (khong thuoc tieu chi C2): 'shuffle' = xao ngau nhien (co dinh seed) lien ket "
                         "bonus<->cluster trong tung head (giu nguyen phan phoi gia tri), va do do giau (enrichment) "
                         "cua vi tri dinh danh khop trong K* / attention. Mac dinh tat -> hanh vi chinh KHONG doi.")
    args = ap.parse_args()
    assert args.lambdas[0] == 0.0, "lambda=0 phai dung dau (doi chieu SA)"
    # KHONG cho phep nhanh khac SA
    assert os.path.normpath(args.sa_dir).split(os.sep)[-2] == "sa", f"--sa_dir phai la thu muc SA: {args.sa_dir}"
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset
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
    p1 = None
    for cand in (os.path.join(args.phase1_dir, args.model), args.phase1_dir):
        mp = os.path.join(cand, f"{args.dataset}_meta.jsonl")
        if os.path.exists(mp):
            p1 = cand
            for line in open(mp, encoding="utf-8"):
                d = json.loads(line); meta[d["dataidx"]] = d
            break
    offsets_npz = np.load(os.path.join(p1, f"{args.dataset}_offsets.npz"))

    all_q, all_k, state = [], [], {"sp": 0}

    def hook(module, inp, out):
        _, qkv, _ = out
        q, k, _v = qkv
        all_q.append(q[:, :, :state["sp"]]); all_k.append(k[:, :, :state["sp"]])
    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(hook)

    lams, sps = args.lambdas, args.sparsity
    res = {lam: {sp: {} for sp in sps} for lam in lams}     # lam -> sp -> metric -> list[n*3]
    cov = []                                                 # thong ke phu song ky hieu, theo mau
    layer_stats = []                                         # theo (mau, lop)
    enrich = []                                              # chi khi --control
    n_det_checked = 0
    idxs = []
    t0 = time.time()
    for i in tqdm(range(min(args.limit, len(data)))):
        rec = meta[i]
        prompt = prompt_format.format(**data[i]); prompt_only = prompt_only_format.format(**data[i])
        prompt, sp_len = truncate_fn(prompt, prompt_only, tokenizer, max_length, args.dataset, DEV,
                                     model_name=args.model, force_chat=False)
        assert sp_len == rec["shared_prefix_length"]
        n_ctx = sp_len - args.observation_window
        S_info = build_symbol_index(prompt, rec, offsets_npz[f"offsets_{i}"], n_ctx, sp_len)
        matched_pos = [S_info["index"][n] for n in S_info["matched"]]

        state["sp"] = sp_len
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear(); all_k.clear()
        with torch.no_grad():
            model.model(input_ids=ids, use_cache=False)
        L = len(all_k); layers = sorted({0, L // 2, L - 1})

        f = glob.glob(os.path.join(args.sa_dir, f"centroids_tensor_dict_{i}_*.pt"))[0]
        K = int(re.search(r"_(\d+)\.pt$", f).group(1))
        cent = torch.load(f, map_location=DEV)
        lab = torch.load(os.path.join(args.sa_dir, f"centroids_labels_dict_{i}_{K}.pt"), map_location=DEV)

        frac_hit_acc = []
        for l in layers:
            q = all_q[l].squeeze(0).float()[:, n_ctx:sp_len, :]
            k = all_k[l].squeeze(0).float()[:, :n_ctx, :]
            c = cent[l].squeeze(0).float(); lb = lab[l].squeeze(0)[:, :n_ctx].long()
            assert q.shape[0] == k.shape[0] == c.shape[0] == lb.shape[0]   # LongChat = MHA
            bonus = symbol_hit_from_labels(matched_pos, lb, K)             # [H,K] in [0,1]
            if args.control == "shuffle":
                g = torch.Generator().manual_seed(1000003 * i + l)
                perm = torch.stack([torch.randperm(K, generator=g) for _ in range(bonus.shape[0])]).to(bonus.device)
                bonus = torch.gather(bonus, 1, perm)                       # cung phan phoi, sai lien ket cluster
                mpos = torch.zeros(k.shape[1], dtype=torch.bool, device=k.device)
                mpos[torch.from_numpy(np.concatenate(matched_pos)).to(k.device)] = True
                er = enrichment(q, k, mpos, sps); er.update({"idx": i, "layer": l}); enrich.append(er)
            assert float(bonus.min()) >= 0.0 and float(bonus.max()) <= 1.0
            out, scale = eval_lambdas(q, k, c, lb, bonus, lams, sps)
            # ---- doi chieu lambda=0 voi ham C2 GOC (bit-exact) ----
            ref = recall_one_sample(q, k, c, lb, sps)
            for sp in sps:
                assert out[0.0][sp]["recall"] == ref[sp][0] and out[0.0][sp]["mass"] == ref[sp][1], \
                    f"lambda=0 KHAC recall_one_sample (mau {i}, lop {l}, sp{sp}): " \
                    f"{out[0.0][sp]['recall']} {out[0.0][sp]['mass']} vs {ref[sp]}"
            # ---- tinh xac dinh: chay lai lambda>0 tren vai mau dau ----
            if i < 4 and l == layers[-1]:
                out2, _ = eval_lambdas(q, k, c, lb, bonus, [0.0, 1.0], sps)
                for sp in sps:
                    for key in out[1.0][sp]:
                        assert out2[1.0][sp][key] == out[1.0][sp][key], "khong xac dinh giua 2 lan chay"
                n_det_checked += 1
            for lam in lams:
                for sp in sps:
                    for key, v in out[lam][sp].items():
                        res[lam][sp].setdefault(key, []).append(v)
            fh = float((bonus > 0).float().mean())
            frac_hit_acc.append(fh)
            layer_stats.append({"idx": i, "layer": l, "K": K, "S_SA_std_across_clusters": scale,
                                "frac_clusters_with_hit": fh, "mean_hit": float(bonus.mean()),
                                "max_hit": float(bonus.max())})
        cov.append({"idx": i, "language": rec["language"], "n_ctx": n_ctx, "K": K,
                    "n_index_ids": S_info["n_index_ids"],
                    "n_query_ids": len(S_info["query_ids"]), "n_matched_ids": len(S_info["matched"]),
                    "n_matched_positions": S_info["n_matched_positions"],
                    "has_match": len(S_info["matched"]) > 0,
                    "frac_clusters_with_hit": float(np.mean(frac_hit_acc)),
                    "query_ids": S_info["query_ids"], "matched_ids": S_info["matched"]})
        idxs.append(i)
        all_q.clear(); all_k.clear()
        if len(idxs) % 20 == 0:
            torch.cuda.empty_cache()

    json.dump({"model": args.model, "dataset": args.dataset, "idx": idxs, "layers": "first,mid,last",
               "lambdas": lams, "sparsity": sps, "sa_dir": args.sa_dir,
               "per_sample": {str(lam): {str(sp): res[lam][sp] for sp in sps} for lam in lams}},
              open(os.path.join(args.out_dir, "results.json"), "w"))
    if args.control:
        json.dump(enrich, open(os.path.join(args.out_dir, "enrichment_raw.json"), "w"))
    json.dump({"per_sample": cov, "per_sample_layer": layer_stats},
              open(os.path.join(args.out_dir, "symbol_coverage_raw.json"), "w"))
    json.dump({"in_run_checks": {
        "lambda0_bit_exact_vs_recall_one_sample": "PASS (assert per sample x layer x sp)",
        "n_selected_keys_equals_N": "PASS (assert per row, every lambda/sp/layer/sample)",
        "determinism_recheck_samples": n_det_checked,
        "n_samples": len(idxs), "elapsed_s": round(time.time() - t0, 1)}},
        open(os.path.join(args.out_dir, "integrity_inrun.json"), "w"), indent=1)
    print(">>> Da ghi", args.out_dir)


if __name__ == "__main__":
    main()
