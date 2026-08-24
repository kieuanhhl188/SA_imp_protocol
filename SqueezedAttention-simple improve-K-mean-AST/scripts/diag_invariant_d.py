#!/usr/bin/env python
"""
diag_invariant_d.py — tim NGUYEN NHAN THAT cua do lech o bat bien D (Phase 2).

BOI CANH
--------
`check_phase2_invariants.py` bat bien D doi chieu nhanh `sa` cua
`offline_clustering_struct.py` voi ban goc `offline_clustering.py`. Ket qua chay 22/8:
**55/500 mau (11%) lech > 5%**, dai khoang cach tap hop 5,7e-03 … 3,4e-01.

`docs/PHASE2_RESULTS.md` (ban dau) giai thich do lech nay bang *"cuML k-means khoi tao
ngau nhien, khong ghim random_state"* va de xuat *"ghim seed roi sinh lai, ~45 phut GPU"*.

**Giai thich do SAI, va cach sua do la NO-OP.** `squeezedattention/clustering.py:69` da
dat `random_state=0` — kiem `git show b03a63d` thi thay dong do co tu first commit, tuc
tu code goc cua Squeezed Attention. CA HAI ben cua phep doi chieu deu di qua dung ham
`run_clustering` do, nen ca hai deu dang chay voi seed ghim san. Ghim them khong doi gi.

Do lech con lai chi co the den tu MOT trong ba cho, va ba cho nay doi hoi ba cach xu ly
hoan toan khac nhau — nen phai tach ra truoc khi ghi bat cu gi vao bai:

  (1) cuML k-means khong tat dinh DU DA GHIM SEED. Lloyd iteration reduce bang atomic
      tren GPU, thu tu cong khac nhau moi lan chay -> ket qua khac o chu so cuoi, roi
      `tol=1e-4` chan som o vong khac nhau. Neu la cho nay: KHONG sua duoc bang seed.
      Phai doi cach bao cao (ghi nhan mot NGUONG SAN cua phep do) hoac doi metric sang
      loai bat bien voi hoan vi + on dinh so (inertia, ARI).

  (2) KEY VECTOR khong giong nhau giua hai lan forward. Neu la cho nay thi khong phai
      loi cua k-means chut nao, ma la mo hinh/moi truong -> moi con so Phase 2 sinh o
      hai thoi diem khac nhau deu khong doi chieu duoc, va do la van de nang hon nhieu.

  (3) Thu muc reference sinh boi CODE/CONFIG KHAC (transformers khac, rope_scaling khac,
      force_chat khac, fixed_context khac). Neu la cho nay: sinh lai reference bang dung
      cau hinh, khong lien quan gi den seed.

CACH TACH — BA TANG DO, TANG DAN
--------------------------------
  T1  forward HAI LAN cung prompt trong CUNG MOT PROCESS, so key A vs key B bit-for-bit.
        -> tra loi (2). Lech = 0 thi key tai lap duoc, loai (2).
  T2  goi `run_clustering` HAI LAN tren CUNG key A, trong cung process.
        -> tra loi (1). Day la NGUONG SAN: do lech toi thieu ma phep do khong the
           xuong duoi, du moi thu khac giong het nhau.
  T3  so ket qua T2 voi file centroid tren dia trong `--reference_dir`.
        -> chinh la bat bien D. Doc bang cach so voi nguong san cua T2:
             T3 ~= T2  -> do lech LA nhieu cuML, khong phai bug. Loai (3).
             T3 >> T2  -> co gi khac ngoai k-means. Nghi (3), phai truy cau hinh.

Metric dung y het `check_phase2_invariants.py:397` (Hausdorff mot chieu lay max, chuan
hoa theo median norm, bo hang zero) DE SO SANH DUOC TRUC TIEP voi con so 5,7e-03…3,4e-01
da bao cao. Kem theo hai metric BAT BIEN VOI HOAN VI va it nhay outlier hon, vi
Hausdorff-max chi can mot centroid roi khac cho la vot len:
    mean_nearest_rel  trung binh khoang cach toi centroid gan nhat (thay vi max)
    inertia_rel       lech tuong doi cua ham muc tieu k-means -> hai phan hoach co
                      *tot ngang nhau* khong. Day moi la cau hoi khoa hoc.
    ari               Adjusted Rand Index tren nhan -> hai phan hoach co GIONG NHAU
                      khong, bat bien voi cach danh so cluster.

CHI PHI: mac dinh 3 mau, ~2-3 phut GPU. Re hon 15 lan so voi "sinh lai 500 mau 45 phut"
mà ban goc de xuat, va tra loi dung cau hoi hon.

USAGE
-----
    # tang T1 + T2 (khong can reference)
    python scripts/diag_invariant_d.py qwen2.5-coder-7b --dataset lcc \\
        --phase1_dir /workspace/phase1_data --limit 3

    # ca ba tang
    python scripts/diag_invariant_d.py qwen2.5-coder-7b-instruct --force_chat \\
        --dataset lcc --phase1_dir /workspace/phase1_data --limit 3 \\
        --reference_dir /workspace/fixed-prompt-clusters/qwen2.5-coder-7b-instruct/lcc \\
        --out /workspace/diag_invariant_d.json

Ma thoat: 0 = da ket luan duoc nguyen nhan, 1 = khong ket luan duoc (xem canh bao in ra).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

JSONL_HEAD = "Please complete the code given below. " + chr(10)
JSONL_TAIL = "Next line of code:" + chr(10)


# =====================================================================
# METRIC
# =====================================================================

def hausdorff_rel(a, b):
    """
    Y HET `check_phase2_invariants.py:397` — de con so o day so thang duoc voi bang da
    bao cao. Tra ve (max_rel, mean_rel).

    a, b: [1,H,K,D] hoac [H,K,D]. Bo hang toan 0 (cuML cap phat du K hang nhung khong
    dung het; hang khong dung o lai zero va khong mang thong tin — dua vao phep do thi
    median norm co the ra 0 va lam no con so).
    """
    x = a.float().reshape(-1, a.shape[-2], a.shape[-1])
    y = b.float().reshape(-1, b.shape[-2], b.shape[-1])
    worst, means = 0.0, []
    for h in range(x.shape[0]):
        na, nb = x[h].norm(dim=-1), y[h].norm(dim=-1)
        ma, mb = na > 0, nb > 0
        if int(ma.sum()) == 0 or int(mb.sum()) == 0:
            continue
        d = torch.cdist(x[h][ma], y[h][mb])
        nearest = d.min(dim=1).values
        scale = float(nb[mb].median().clamp_min(1e-12))
        worst = max(worst, float(nearest.max()) / scale)
        means.append(float(nearest.mean()) / scale)
    return worst, (float(np.mean(means)) if means else 0.0)


def inertia(keys, cents, labels):
    """
    Ham muc tieu k-means: tong |key - centroid_cua_no|^2, tinh tren key CHUA chuan hoa
    (dung khong gian ma centroid duoc tinh trung binh trong do — xem `run_clustering`:
    fit tren key da L2-normalize nhung centroid la trung binh cua key GOC).

    keys:   [H, S, D]   cents: [H, K, D]   labels: [H, S]
    Bat bien voi hoan vi nhan -> so duoc giua hai lan chay.
    """
    H, S, D = keys.shape
    idx = labels.long().unsqueeze(-1).expand(H, S, D)
    assigned = torch.gather(cents.float(), 1, idx)
    return float(((keys.float() - assigned) ** 2).sum())


def ari(x, y, K):
    """
    Adjusted Rand Index giua hai phan hoach tren cung tap diem. Bat bien voi cach danh
    so cluster, nen tra loi dung cau hoi "hai lan chay co cho ra CUNG mot phan hoach
    khong" — thu ma Hausdorff tren centroid khong tra loi duoc.

    x, y: [S] long, gia tri trong [0, K).
    """
    x, y = x.long().reshape(-1).cpu(), y.long().reshape(-1).cpu()
    n = x.numel()
    if n < 2:
        return 1.0
    cont = torch.bincount(x * K + y, minlength=K * K).double()
    a = cont.reshape(K, K).sum(1)
    b = cont.reshape(K, K).sum(0)

    def c2(v):
        return (v * (v - 1) / 2).sum()

    sum_ij, sum_a, sum_b = float(c2(cont)), float(c2(a)), float(c2(b))
    total = n * (n - 1) / 2
    exp = sum_a * sum_b / total
    mx = 0.5 * (sum_a + sum_b)
    if abs(mx - exp) < 1e-12:
        return 1.0
    return (sum_ij - exp) / (mx - exp)


# =====================================================================
# DU LIEU
# =====================================================================

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


def load_meta(phase1_dir, dataset, model):
    for cand in (os.path.join(phase1_dir, model), phase1_dir):
        mp = os.path.join(cand, f"{dataset}_meta.jsonl")
        if os.path.exists(mp):
            recs = {}
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    recs[d["dataidx"]] = d
            print(f">>> Phase 1.4: {mp} ({len(recs)} mau)")
            return recs
    raise SystemExit(f"[ERROR] thieu meta Phase 1.4 trong {phase1_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dataset", default="lcc")
    ap.add_argument("--data_source", choices=["longbench", "jsonl"], default="longbench")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--fixed_context", choices=["full", "crossfile"], default="full")
    ap.add_argument("--force_chat", action="store_true")
    ap.add_argument("--rope_scaling", default=None)
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--reference_dir", default=None,
                    help="thu muc centroid do offline_clustering.py sinh — de chay tang T3")
    ap.add_argument("--layers", default="first,mid,last",
                    help="'first,mid,last' hoac danh sach so. Tinh moi lop rat cham ma "
                         "khong doi ket luan")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=0.05,
                    help="nguong cua bat bien D, giong check_phase2_invariants.py")
    ap.add_argument("--out", default="diag_invariant_d.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset

    from squeezedattention.clustering import run_clustering
    from squeezedattention.utils import truncate_fn, apply_rope_scaling

    # In ra seed dang dung, ngay dau output: day chinh la con so ma tai lieu noi "khong
    # ghim". Doc thang tu source de khong ai phai tin loi ke.
    src = open(os.path.join(REPO_ROOT, "squeezedattention", "clustering.py"),
               encoding="utf-8").read()
    m = re.search(r"random_state\s*=\s*(\S+)", src)
    print("=" * 72)
    print("  DIAG bat bien D — nguyen nhan do lech nhanh `sa` vs ban goc")
    print("=" * 72)
    print(f"  cuML KMeans random_state trong squeezedattention/clustering.py: "
          f"{m.group(1) if m else 'KHONG DAT'}")
    if m:
        print("  -> seed DA duoc ghim san. 'Ghim seed roi sinh lai' la no-op.")
    print()

    DEV = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    m2p = json.load(open("LongBench/config/model2path.json", encoding="utf-8"))
    m2l = json.load(open("LongBench/config/model2maxlen.json", encoding="utf-8"))
    d2p = json.load(open("LongBench/config/dataset2prompt.json", encoding="utf-8"))
    model_path, max_length = m2p[args.model], m2l[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    config = apply_rope_scaling(config, args.rope_scaling)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    if getattr(config, "use_sliding_window", False):
        config.use_sliding_window = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16).eval().to(DEV)

    if args.data_source == "jsonl":
        if not args.data_dir:
            raise SystemExit("[ERROR] --data_source jsonl can --data_dir")
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

    meta = load_meta(args.phase1_dir, args.dataset, args.model)

    all_q, all_k = [], []
    state = {"sp": 0}

    def hook(module, inp, out):
        _, qkv, _ = out
        q, k, _v = qkv
        all_q.append(q[:, :, :state["sp"]])
        all_k.append(k[:, :, :state["sp"]])

    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(hook)

    def forward_once(prompt, sp_len):
        state["sp"] = sp_len
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear()
        all_k.clear()
        with torch.no_grad():
            model.generate(ids, do_sample=False, max_new_tokens=1, use_cache=False,
                           output_attentions=True)
        return [t.clone() for t in all_k]

    rows = []
    n = min(args.limit, len(data)) if args.limit > 0 else len(data)
    for i in range(n):
        rec = meta.get(i)
        if rec is None:
            continue
        d = data[i]
        prompt, sp_len = truncate_fn(
            prompt_format.format(**d), prompt_only_format.format(**d), tokenizer,
            max_length, ds_trunc, DEV, model_name=args.model,
            force_chat=args.force_chat)
        if sp_len != rec["shared_prefix_length"]:
            raise SystemExit(f"[ERROR] dataidx {i}: sp_len lech voi Phase 1.4 "
                             f"({sp_len} vs {rec['shared_prefix_length']})")
        n_ctx = sp_len - args.observation_window
        if n_ctx <= 0:
            continue
        K = max(1, int(args.percent_clusters / 100.0 * n_ctx))
        row = {"dataidx": i, "n_ctx": n_ctx, "K": K}
        print(f"-- dataidx {i}  n_ctx={n_ctx}  K={K}")

        # ---------------- T1: key co tai lap duoc khong ----------------
        kA = forward_once(prompt, sp_len)
        kB = forward_once(prompt, sp_len)
        L = len(kA)
        layers = (sorted({0, L // 2, L - 1}) if args.layers == "first,mid,last"
                  else [int(x) for x in args.layers.split(",")])
        dmax = max(float((kA[l].float() - kB[l].float()).abs().max()) for l in range(L))
        kscale = max(float(kA[l].float().abs().max()) for l in range(L))
        row["T1_key_absmax_diff"] = dmax
        row["T1_key_rel_diff"] = dmax / max(kscale, 1e-12)
        row["T1_bitwise_identical"] = dmax == 0.0
        print(f"   T1 key A vs B: absmax={dmax:.3e}  rel={row['T1_key_rel_diff']:.3e}"
              f"  {'BIT-FOR-BIT GIONG' if dmax == 0.0 else 'LECH'}")

        # ---------------- T2: cuML co tat dinh khong ----------------
        # Chay hai lan tren CUNG key A -> loai bo hoan toan yeu to key.
        c1, l1 = run_clustering(kA, K, observation_window=args.observation_window,
                                device=DEV)
        c2, l2 = run_clustering(kA, K, observation_window=args.observation_window,
                                device=DEV)
        t2h, t2m, t2i, t2a = [], [], [], []
        for l in layers:
            hmax, hmean = hausdorff_rel(c1[l].cpu(), c2[l].cpu())
            t2h.append(hmax)
            t2m.append(hmean)
            keys = kA[l].squeeze(0).float()[:, :n_ctx, :]
            i1 = inertia(keys, c1[l].squeeze(0), l1[l].squeeze(0)[:, :n_ctx])
            i2 = inertia(keys, c2[l].squeeze(0), l2[l].squeeze(0)[:, :n_ctx])
            t2i.append(abs(i1 - i2) / max(i1, 1e-12))
            t2a.append(float(np.mean([
                ari(l1[l].squeeze(0)[h, :n_ctx], l2[l].squeeze(0)[h, :n_ctx], K)
                for h in range(l1[l].shape[1])])))
        row["T2_hausdorff_max"] = max(t2h)
        row["T2_nearest_mean"] = float(np.mean(t2m))
        row["T2_inertia_rel"] = max(t2i)
        row["T2_ari"] = float(np.mean(t2a))
        print(f"   T2 cuML lan1 vs lan2: hausdorff_max={row['T2_hausdorff_max']:.3e}"
              f"  nearest_mean={row['T2_nearest_mean']:.3e}"
              f"  inertia_rel={row['T2_inertia_rel']:.3e}  ARI={row['T2_ari']:.4f}")

        # ---------------- T3: doi chieu voi file tren dia ----------------
        if args.reference_dir:
            f = os.path.join(args.reference_dir,
                             f"centroids_tensor_dict_{i}_{K}.pt")
            if not os.path.exists(f):
                alt = glob.glob(os.path.join(args.reference_dir,
                                             f"centroids_tensor_dict_{i}_*.pt"))
                if alt:
                    # K khac nghia la n_ctx khac -> prompt khac -> khong phai chuyen cua
                    # k-means. Bao ngay, dung im lang doi file.
                    kref = int(re.search(r"_(\d+)\.pt$", alt[0]).group(1))
                    row["T3_note"] = (f"reference co K={kref} != K={K}: n_ctx khac -> "
                                      f"prompt/config khac, khong phai nhieu k-means")
                    print(f"   T3 ⚠ reference K={kref} != {K} -> CAU HINH KHAC "
                          f"(prompt/tokenizer/maxlen), khong phai nhieu k-means")
                else:
                    row["T3_note"] = "khong co file reference cho dataidx nay"
                    print("   T3 (khong co file reference)")
            else:
                ref = torch.load(f, map_location="cpu")
                # Nap MOT LAN ngoai vong lap lop: moi file nhan la vai tram MB, nap lai
                # cho tung lop se cham hon phep do chinh.
                flab = os.path.join(args.reference_dir,
                                    f"centroids_labels_dict_{i}_{K}.pt")
                lref = torch.load(flab, map_location="cpu") if os.path.exists(flab) else None
                t3h, t3m, t3i = [], [], []
                for l in layers:
                    if l not in ref:
                        continue
                    if ref[l].shape != c1[l].cpu().shape:
                        row["T3_shape_mismatch"] = [list(ref[l].shape),
                                                    list(c1[l].shape)]
                        break
                    hmax, hmean = hausdorff_rel(c1[l].cpu(), ref[l])
                    t3h.append(hmax)
                    t3m.append(hmean)
                    if lref is None or l not in lref:
                        continue           # thieu nhan -> chi so duoc centroid
                    keys = kA[l].squeeze(0).float()[:, :n_ctx, :]
                    i1 = inertia(keys, c1[l].squeeze(0), l1[l].squeeze(0)[:, :n_ctx])
                    ir = inertia(keys, ref[l].squeeze(0).to(DEV),
                                 lref[l].squeeze(0)[:, :n_ctx].to(DEV))
                    t3i.append(abs(i1 - ir) / max(i1, 1e-12))
                if t3h:
                    row["T3_hausdorff_max"] = max(t3h)
                    row["T3_nearest_mean"] = float(np.mean(t3m))
                    row["T3_inertia_rel"] = max(t3i)
                    print(f"   T3 vs ban goc: hausdorff_max={row['T3_hausdorff_max']:.3e}"
                          f"  nearest_mean={row['T3_nearest_mean']:.3e}"
                          f"  inertia_rel={row['T3_inertia_rel']:.3e}")

        rows.append(row)
        del c1, c2, l1, l2, kA, kB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("[ERROR] khong do duoc mau nao")

    # =================================================================
    # KET LUAN
    # =================================================================
    print()
    print("=" * 72)
    print("  KET LUAN")
    print("=" * 72)
    rc = 0

    key_ok = all(r["T1_bitwise_identical"] for r in rows)
    t2 = max(r["T2_hausdorff_max"] for r in rows)
    t2_ari = min(r["T2_ari"] for r in rows)
    has_t3 = any("T3_hausdorff_max" in r for r in rows)
    t3 = max((r["T3_hausdorff_max"] for r in rows if "T3_hausdorff_max" in r),
             default=None)

    if key_ok:
        print("  (2) KEY VECTOR: loai bo. Hai lan forward cho key bit-for-bit giong nhau")
        print("      -> do lech khong den tu mo hinh/moi truong forward.")
    else:
        kr = max(r["T1_key_rel_diff"] for r in rows)
        print(f"  (2) KEY VECTOR: ⚠ LECH (rel toi da {kr:.3e}). Forward KHONG tai lap.")
        print("      -> day la van de nang hon k-means: moi con so sinh o hai thoi diem")
        print("         khac nhau deu khong doi chieu duoc. Truy truoc khi ket luan gi ve D.")
        rc = 1

    print(f"  (1) cuML: NGUONG SAN cua phep do = {t2:.3e} (hausdorff_max, cung key,")
    print(f"      cung process, seed ghim). ARI thap nhat {t2_ari:.4f}.")
    if t2 <= 1e-6:
        print("      -> cuML TAT DINH. Nhieu k-means KHONG giai thich duoc 5,7e-03…3,4e-01.")
    elif t2 >= args.rtol:
        print(f"      -> cuML KHONG tat dinh du ghim seed, va nguong san da vuot {args.rtol}.")
        print("         Bat bien D o dang hien tai KHONG THE PASS. Phai doi metric sang")
        print("         loai bat bien hoan vi + on dinh so (inertia_rel, ARI) va bao cao")
        print("         nguong san nay nhu mot gioi han cua phep do.")
    else:
        print(f"      -> cuML co nhieu nhung con duoi nguong {args.rtol}.")

    if has_t3:
        print(f"  (3) THU MUC REFERENCE: T3={t3:.3e} vs nguong san T2={t2:.3e} "
              f"(ty le {t3 / max(t2, 1e-12):.1f}x)")
        if t3 <= max(3 * t2, 1e-9):
            print("      -> do lech tren dia CUNG BAC voi nhieu cuML. Khong co bug thu ba.")
            print("         Ghi vao bai: bat bien D bi chan boi tinh khong tat dinh cua")
            print("         cuML, khong phai boi seed. Sua bang metric, khong bang seed.")
        else:
            print("      -> do lech tren dia LON HON HAN nhieu cuML. Con mot nguyen nhan")
            print("         thu ba: reference sinh boi code/config khac (transformers,")
            print("         rope_scaling, force_chat, fixed_context, maxlen). Doi chieu")
            print("         cau hinh cua hai lan chay truoc khi ket luan.")
            rc = 1
    else:
        print("  (3) THU MUC REFERENCE: chua chay (thieu --reference_dir).")
        print("      Chua the tach 'nhieu cuML' voi 'reference sinh boi config khac'.")
        rc = 1

    payload = {"model": args.model, "dataset": args.dataset,
               "percent_clusters": args.percent_clusters,
               "observation_window": args.observation_window,
               "cuml_random_state": (m.group(1) if m else None),
               "reference_dir": args.reference_dir,
               "layers": args.layers, "rtol": args.rtol,
               "conclusive": rc == 0, "per_sample": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n>>> Da ghi {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
