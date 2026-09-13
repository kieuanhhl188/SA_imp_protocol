#!/usr/bin/env python
"""
check_phase2_invariants.py — kiem bat bien cua Phase 2 TREN FILE .pt DA SINH.

Chay CPU, khong can GPU, khong can model weight. Doc thang centroid/label da luu +
offset cua Phase 1.4, roi tinh lai unit_id de doi chieu.

VI SAO CAN FILE NAY
-------------------
Toan bo 86 test cua struct_clustering dung `torch.randn`. Chua bat bien nao duoc kiem
tren key vector THAT. Ma dung ba loi da gap o Phase 1 deu thuoc loai *khong crash, chi
sai lech*:
  - nguong NaN  -> Sq-70% 23.05 thay vi 62.55
  - language hardcode -> 59.5% mau LCC con <=2 unit, ablation vo hieu
  - byte vs ky tu -> 107/500 mau lech span cong don

NAM BAT BIEN
------------
  A. RANH GIOI CUNG   moi cluster chi chua token cua DUNG MOT unit.
                      Day la dinh nghia cua hard_boundary. Vo bat bien nay thi
                      "+HardBoundary" khong con la hard boundary.
  B. CUNG BUDGET      tong K bang nhau giua cac nhanh, cung dataidx.
                      Protocol 6.2: khong cung budget thi khong so duoc.
  C. SHAPE + DO DAI   centroids [1,H,K,D] · labels [1,H,n_ctx]
                      n_ctx == shared_prefix_length - observation_window.
  D. NHANH `sa` TRUNG  `--method sa` phai cho ra ket qua y het offline_clustering.py.
                      Lech nghia la script moi tu lam sai gi do, moi so sanh sau vo nghia.
  E. TANG L1          chi struct_hierarchy. K1<=K2 · hierarchy long nhau (moi cluster
                      L2 thuoc dung 1 nhom L1) · labels_l1 khong per-head · centroid L1
                      == trung binh CO TRONG SO cac centroid L2 (khong phai k-means-cua-
                      k-means) · K1 khop k1_stats. Xem EXPERIMENT_LOG entry 10/9.

USAGE
-----
    # kiem mot nhanh
    python scripts/check_phase2_invariants.py \\
        --cluster_dir /workspace/smoke_struct/hard_boundary --method hard_boundary \\
        --phase1_dir /workspace/phase1_data/longchat-v1.5-7b-32k --dataset lcc

    # kiem nhieu nhanh + so cung budget + doi chieu nhanh sa voi ban goc
    python scripts/check_phase2_invariants.py \\
        --cluster_dir sa=/workspace/smoke_struct/sa \\
        --cluster_dir hard_boundary=/workspace/smoke_struct/hard_boundary \\
        --cluster_dir struct_hierarchy=/workspace/smoke_struct/struct_hierarchy \\
        --model longchat-v1.5-7b-32k \\
        --phase1_dir /workspace/phase1_data/longchat-v1.5-7b-32k --dataset lcc \\
        --reference_dir /workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc

Dang TEN=/duong/dan la quy uoc dung o `phase5_recall.py`, `compare_partitions.py` va
`run_phase2_phase5_lcc.sh`. File nay nhan CA HAI thu tu (xem cho parse `--cluster_dir`),
nhung viet theo dang tren de moi script trong du an doc giong nhau.

Ma thoat: 0 = moi bat bien qua, 1 = co vi pham.
"""
import argparse
import glob
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

_NAME = re.compile(r"^centroids_labels_dict_(\d+)_(\d+)\.pt$")
_L1NAME = re.compile(r"^hierarchical_centroids_labels_dict_L1_(\d+)_(\d+)\.pt$")


def discover(cluster_dir):
    """Tra ve {dataidx: K} tu ten file trong thu muc."""
    out = {}
    for p in glob.glob(os.path.join(cluster_dir, "centroids_labels_dict_*.pt")):
        m = _NAME.match(os.path.basename(p))
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def discover_l1(cluster_dir):
    """Tra ve {dataidx: K1} tu ten file tang L1 (chi struct_hierarchy moi co)."""
    out = {}
    for p in glob.glob(os.path.join(cluster_dir,
                                    "hierarchical_centroids_labels_dict_L1_*.pt")):
        m = _L1NAME.match(os.path.basename(p))
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def discover_method(cluster_dir, fallback):
    """Tra ve `method` THAT da dung de sinh thu muc nay, doc tu feasibility_*.json —
    khong doan qua ten nhan `--cluster_dir` (ten do nguoi goi lenh tu dat, co the lech
    quy uoc, xem vu hard_boundary_class 12/9: bi coi la NHOM DOI CHUNG vi ten khong
    khop y het chuoi "hard_boundary" trong HARD set, du du lieu la hard_boundary that).
    Nhanh `sa` khong sinh file nay -> giu nguyen `fallback` (doan qua ten, nhu truoc).
    """
    for p in glob.glob(os.path.join(cluster_dir, "feasibility_*.json")):
        try:
            return json.load(open(p, encoding="utf-8")).get("method", fallback)
        except Exception:
            pass
    return fallback


def load_meta(phase1_dir, dataset):
    path = os.path.join(phase1_dir, f"{dataset}_meta.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] khong thay {path} — chay prepare_code_data.py truoc")
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[int(r["dataidx"])] = r
    return recs


def rebuild_prompts(model, dataset, idxs, meta):
    """Dung lai prompt CUOI CUNG (sau truncation) cho cac dataidx can kiem.

    `prepare_code_data.py` mac dinh KHONG luu `prompt` vao meta — 500 mau x ~40 KB se lam
    meta phinh len ~20 MB. Nen phai dung lai, theo dung tung buoc cua check_phase1_data.py
    de ra chuoi y het: tokenizer CHAM + truncate_fn.

    QUAN TRONG: `fixed_context_mode` (full/crossfile) doi key template dung cho
    `prompt_noquery` — VA CHI KHAC NHAU O REPOBENCH-P (lcc thi `_prompt` == `_prompt_full`,
    xem LongBench/config/dataset2prompt.json). Phai doc dung gia tri DA GHI trong meta cho
    TUNG dataidx (meta["fixed_context_mode"]), khong duoc doan/hard-code — neu khong sp_len
    dung lai se lech voi meta o moi mau co {input} (repobench-p), vi truncate_fn se coi
    {input} la phan "query" thay vi phan "co dinh". Da vap loi nay 12/9: 200/200 mau
    repobench-p bao [!] sp_len lech, ke ca dataidx thuoc lo cu chua he bi dung — chung minh
    loi nam o cho ham nay, khong phai o du lieu.
    """
    # REPO_ROOT chua thu muc `transformers/` — do la CAY NGUON cua fork (package that nam o
    # transformers/src/transformers), khong co __init__.py o cap do. Python coi no la
    # namespace package va che mat ban da cai:
    #     ImportError: cannot import name 'AutoTokenizer' from 'transformers' (unknown location)
    # Nen phai tam go REPO_ROOT khoi sys.path dung luc import hai goi ngoai.
    _saved = [p for p in sys.path if p and os.path.abspath(p) == os.path.abspath(REPO_ROOT)]
    for p in _saved:
        sys.path.remove(p)
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
    except ImportError:
        # Go REPO_ROOT ra khong giup -> khoi phuc roi thu lai. Bat duoc ca hai kieu hong:
        # (a) namespace package che mat ban cai   -> lan thu dau thanh cong
        # (b) chua kich hoat venv, khong co goi nao -> lan thu hai cung hong, bao ro
        for p in _saved:
            sys.path.insert(0, p)
        _saved = []
        for m in [k for k in sys.modules if k == "transformers" or k.startswith("transformers.")]:
            del sys.modules[m]
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
        except ImportError as e2:
            import shutil
            raise SystemExit(
                f"[ERROR] khong import duoc transformers/datasets: {e2}\n"
                f"        python dang dung : {sys.executable}\n"
                f"        phien ban        : {sys.version.split()[0]}\n"
                f"        sys.path[:4]     : {sys.path[:4]}\n"
                f"\n"
                f"        Nguyen nhan thuong gap nhat: CHUA KICH HOAT VENV.\n"
                f"        Dau nhac phai co tien to (venv310). Chay truoc:\n"
                f"            source /workspace/env.sh\n"
                f"        roi kiem:\n"
                f"            python -c \"import transformers; print(transformers.__version__)\"\n"
                f"        (phai ra 4.40.0.dev0, khong phai ban tren PyPI)\n"
                + ("" if shutil.which("python") else "")
            )
    finally:
        for p in _saved:
            sys.path.insert(0, p)

    # cai nay thi CAN REPO_ROOT trong sys.path
    from squeezedattention.utils import truncate_fn

    cfg = os.path.join(REPO_ROOT, "LongBench", "config")
    model2path = json.load(open(os.path.join(cfg, "model2path.json"), encoding="utf-8"))
    model2maxlen = json.load(open(os.path.join(cfg, "model2maxlen.json"), encoding="utf-8"))
    d2p = json.load(open(os.path.join(cfg, "dataset2prompt.json"), encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(model2path[model], use_fast=False)
    max_length = model2maxlen[model]
    fmt = d2p[dataset]

    data = load_dataset("THUDM/LongBench", dataset, split="test")
    out = {}
    for i in sorted(idxs):
        d = data[int(i)]
        rec = meta.get(i, {})
        mode = rec.get("fixed_context_mode", "full")
        key_only = dataset + ("_prompt_full" if mode == "full" else "_prompt")
        fmt_only = d2p[key_only]
        prompt, sp_len = truncate_fn(fmt.format(**d), fmt_only.format(**d),
                                     tok, max_length, dataset, "cpu",
                                     model_name=model, force_chat=rec.get("force_chat", False))
        out[i] = (prompt, int(sp_len))
    return out


def unit_ids_for(rec, offs, n_ctx, level, prompt):
    """Tinh lai unit_id tung token — dung y het check_phase1_data.py."""
    import torch
    from struct_clustering import parse_units, assign_token_units, compact_unit_ids

    code = prompt[rec["code_char_start"]:rec["code_char_end"]]
    spans, _ = parse_units(code, rec["language"], level)
    spans = [(s + rec["code_char_start"], e + rec["code_char_start"]) for s, e in spans]
    spans.append((0, len(prompt) + 1))
    starts = torch.from_numpy(offs[:n_ctx, 0].astype("int64"))
    raw = assign_token_units(starts, spans)
    uid, _ = compact_unit_ids(raw)
    return uid, None


def check_hard_boundary(labels_dict, uid, K):
    """A. Moi cluster chi chua token cua dung mot unit. Tra ve (n_vi_pham, tong_cluster)."""
    import torch

    n_bad = n_tot = 0
    worst = None
    for layer, lab in labels_dict.items():
        t = lab if lab.dim() == 3 else lab.unsqueeze(0)
        t = t[0].to(torch.int64)                     # [H, n_ctx]
        H, S = t.shape
        if S != uid.numel():
            return None, None, f"labels dai {S} nhung unit_id dai {uid.numel()}"
        for h in range(H):
            l = t[h]
            umin = torch.full((K,), 2**30, dtype=torch.int64)
            umax = torch.full((K,), -1, dtype=torch.int64)
            umin.scatter_reduce_(0, l, uid, reduce="amin", include_self=False)
            umax.scatter_reduce_(0, l, uid, reduce="amax", include_self=False)
            used = umax >= 0
            bad = used & (umax > umin)
            nb = int(bad.sum())
            n_bad += nb
            n_tot += int(used.sum())
            if nb and worst is None:
                k = int(torch.nonzero(bad)[0])
                worst = (layer, h, k, int(umin[k]), int(umax[k]))
    return n_bad, n_tot, worst


def check_l1_layer(path, idx, K2, n_ctx, cent_l2, lab_l2, rtol):
    """E. TANG L1 cua struct_hierarchy — doc file hierarchical_*_L1_*.pt DA SINH.

    De xuat 2 = de xuat 1 + mot tang chi muc tho. Tang do chi dung khi truy hoi 2
    tang, va den 9/9 chua bat bien nao cham vao no — moi phep do deu thay no TRUNG
    de xuat 1. Bat bien nay xac nhan file L1 dung la "L1 = trung binh theo
    function/file" chu khong phai k-means-cua-k-means, va hierarchy long nhau.

      E1. K1 <= K2                          L1 phai tho hon L2
      E2. moi cluster L2 thuoc DUNG 1 nhom L1    hierarchy long nhau (khong vat)
      E3. labels_l1 giong nhau moi head     nhom L1 la cau truc, khong per-head
      E4. do dai labels_l1 == n_ctx
      E5. centroid L1 ~ trung binh CO TRONG SO (theo so key) cac centroid L2 thanh
          vien — sai so tuong doi <= rtol. Day la dinh nghia cua struct_hierarchy_l1;
          lech nghia la file L1 sinh sai cong thuc.
      E6. K1 khop hau to ten file + k1_stats.k1_actual

    Tra ve (ok, K1, det). ok=None khi thieu file.
    """
    import torch

    fl = glob.glob(os.path.join(path, f"hierarchical_centroids_labels_dict_L1_{idx}_*.pt"))
    fc = glob.glob(os.path.join(path, f"hierarchical_centroids_tensor_dict_L1_{idx}_*.pt"))
    if not fl or not fc:
        return None, None, {"err": "thieu file tang L1"}

    K1_name = int(_L1NAME.match(os.path.basename(fl[0])).group(2))
    lab_l1 = torch.load(fl[0], map_location="cpu")
    cen_l1 = torch.load(fc[0], map_location="cpu")

    det = {"K1": K1_name, "K2": K2}
    fails = []

    if K1_name > K2:
        fails.append(f"E1: K1={K1_name} > K2={K2}")

    fs = os.path.join(path, f"k1_stats_{idx}.pt")
    if os.path.exists(fs):
        st = torch.load(fs, map_location="cpu")
        det["mode"] = st.get("k1_mode", st.get("mode"))
        if int(st.get("k1_actual", -1)) != K1_name:
            fails.append(f"E6: k1_stats.k1_actual={st.get('k1_actual')} != ten file {K1_name}")

    worst_e5 = 0.0
    for lyr in lab_l1:
        l1 = lab_l1[lyr][0].to(torch.int64)          # [H, S]
        l2 = lab_l2[lyr][0].to(torch.int64)          # [H, S]
        c2 = cent_l2[lyr][0].float()                 # [H, K2, D]
        c1 = cen_l1[lyr][0].float()                  # [H, K1, D]
        H, S = l1.shape
        D = c2.shape[-1]

        if S != n_ctx:
            fails.append(f"E4: layer {lyr} labels_l1 dai {S} != n_ctx {n_ctx}")
        if not bool((l1 == l1[0]).all()):
            fails.append(f"E3: layer {lyr} labels_l1 khac nhau giua cac head")
        K1 = int(l1.max()) + 1
        if K1 != K1_name:
            fails.append(f"layer {lyr}: max(labels_l1)+1={K1} != {K1_name}")

        for h in range(H):
            # E2 — cung ky thuat scatter_reduce nhu check_hard_boundary
            gmin = torch.full((K2,), 2**30, dtype=torch.int64)
            gmax = torch.full((K2,), -1, dtype=torch.int64)
            gmin.scatter_reduce_(0, l2[h], l1[h], reduce="amin", include_self=False)
            gmax.scatter_reduce_(0, l2[h], l1[h], reduce="amax", include_self=False)
            used = gmax >= 0
            if bool((used & (gmax != gmin)).any()):
                k = int(torch.nonzero(used & (gmax != gmin))[0])
                fails.append(f"E2: layer {lyr} head {h} cluster L2 {k} vat qua nhom L1 "
                             f"{int(gmin[k])}..{int(gmax[k])}")
                break

            # E5 — dung lai cong thuc struct_hierarchy_l1 (weighted=True)
            cnt = torch.zeros(K2).scatter_add_(0, l2[h], torch.ones(S))
            cl2_to_l1 = torch.zeros(K2, dtype=torch.int64).scatter_(0, l2[h], l1[h])
            num = torch.zeros(K1, D).scatter_add_(
                0, cl2_to_l1.unsqueeze(-1).expand(K2, D), c2[h] * cnt.unsqueeze(-1))
            den = torch.zeros(K1).scatter_add_(0, cl2_to_l1, cnt)
            recomp = num / den.clamp_min(1e-12).unsqueeze(-1)
            m = den > 0
            rel = ((recomp[m] - c1[h][m]).norm(dim=-1)
                   / c1[h][m].norm(dim=-1).clamp_min(1e-12)).max()
            worst_e5 = max(worst_e5, float(rel))

    if worst_e5 > rtol:
        fails.append(f"E5: centroid L1 lech trung binh trong so, rel={worst_e5:.2e} > {rtol}")
    det["worst_rel_E5"] = worst_e5
    det["fails"] = fails
    return (len(fails) == 0), K1_name, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", action="append", required=True,
                    help="duong_dan, hoac ten_nhanh=duong_dan (dang dung o phase5_recall.py). "
                         "Nhan ca thu tu nguoc duong_dan=ten_nhanh. Lap lai de kiem nhieu nhanh")
    ap.add_argument("--method", default=None, help="ten nhanh khi chi co 1 --cluster_dir")
    ap.add_argument("--phase1_dir", required=True)
    ap.add_argument("--model", default="longchat-v1.5-7b-32k",
                    help="de dung lai prompt sau truncation (meta khong luu san)")
    ap.add_argument("--dataset", default="lcc")
    ap.add_argument("--level", default="function")
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--reference_dir", default=None,
                    help="thu muc centroid do offline_clustering.py sinh — de kiem bat bien D")
    ap.add_argument("--rtol_set", type=float, default=0.05,
                    help="nguong khoang cach tap hop (chuan hoa) cho bat bien D va E5. "
                         "K-means co the khong tat dinh nen dung nguong long, khong doi bang 0")
    ap.add_argument("--checks", default="ABCDE",
                    help="cac bat bien can chay, vd 'BCE' de bo qua [A] (dung lai 200 prompt, "
                         "cham) va [D]. Mac dinh chay het.")
    args = ap.parse_args()
    checks = set(args.checks.upper())

    import numpy as np
    import torch
    # Script nay IO-nang, compute-nhe. De torch mo het core tren may nhieu core gay
    # oversubscription -> scatter/reduce cham gap chuc lan. Ghim vua phai.
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    dirs = {}
    for spec in args.cluster_dir:
        # Chap nhan CA HAI thu tu: "ten=duong_dan" va "duong_dan=ten".
        # Ly do: phase5_recall.py va compare_partitions.py dung thu tu NGUOC lai voi ban
        # dau cua file nay. Hai quy uoc trai chieu trong cung mot du an la loi cho nguoi
        # dung — da vap that ngay 23/8: lenh kiem bat bien im lang bao "khong thay file
        # centroid" sau khi Phase 2 da chay xong 3 gio.
        #
        # Nhan dien bang cach xem VE NAO la thu muc co that. Khi KHONG ve nao la thu muc
        # (duong dan sai chinh ta, hoac chay o may khac) thi mac dinh phai la
        # "ten=duong_dan" — quy uoc chung cua du an — de thong bao loi chi dung vao thu
        # thuc su la duong dan. Neu mac dinh nguoc lai, mot loi chinh ta se bao
        # "khong thay file centroid trong sa", giau mat duong dan that.
        if "=" in spec:
            left, right = spec.rsplit("=", 1)
            if os.path.isdir(left) and not os.path.isdir(right):
                path, name = left, right          # dang cu: duong_dan=ten
            else:
                name, path = left, right          # quy uoc chung: ten=duong_dan
        else:
            path, name = spec, (args.method or os.path.basename(spec.rstrip("/\\")))
        dirs[name] = path

    meta = load_meta(args.phase1_dir, args.dataset)
    npz_path = os.path.join(args.phase1_dir, f"{args.dataset}_offsets.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"[ERROR] khong thay {npz_path}")
    npz = np.load(npz_path)

    found = {n: discover(p) for n, p in dirs.items()}
    for n, d in found.items():
        if not d:
            raise SystemExit(f"[ERROR] khong thay file centroid nao trong {dirs[n]}")

    print("=" * 72)
    print(f"  KIEM BAT BIEN PHASE 2 — dataset={args.dataset} level={args.level}")
    for n, p in dirs.items():
        print(f"    {n:18s} {len(found[n]):3d} mau  <- {p}")
    print("=" * 72)

    rc = 0
    prompts = {}

    # ---------- B. CUNG BUDGET ----------
    if "B" in checks:
        print("\n[B] Cung budget — tong K bang nhau giua cac nhanh")
        common = set.intersection(*(set(d) for d in found.values()))
        if not common:
            print("    [!] khong co dataidx chung giua cac nhanh")
            rc = 1
        for idx in sorted(common):
            ks = {n: found[n][idx] for n in found}
            ok = len(set(ks.values())) == 1
            rc |= 0 if ok else 1
            print(f"    dataidx {idx:3d}: " + " · ".join(f"{n}={k}" for n, k in ks.items())
                  + ("   ✅" if ok else "   ❌ LECH"))

    # ---------- C + A ----------
    HARD = {"hard_boundary", "struct_hierarchy"}
    for name, path in dirs.items():
        if not ({"C", "A"} & checks):
            break
        print(f"\n[C] Shape + do dai — {name}")
        for idx in sorted(found[name]):
            K = found[name][idx]
            rec = meta.get(idx)
            if rec is None:
                print(f"    dataidx {idx}: [!] khong co trong meta"); rc = 1; continue
            n_ctx = rec["shared_prefix_length"] - args.observation_window
            cen = torch.load(os.path.join(path, f"centroids_tensor_dict_{idx}_{K}.pt"),
                             map_location="cpu")
            lab = torch.load(os.path.join(path, f"centroids_labels_dict_{idx}_{K}.pt"),
                             map_location="cpu")
            c0, l0 = cen[0], lab[0]
            okc = c0.dim() == 4 and c0.shape[0] == 1 and c0.shape[2] == K
            okl = l0.dim() == 3 and l0.shape[0] == 1 and l0.shape[2] == n_ctx
            rc |= 0 if (okc and okl) else 1
            # NGAN SACH HIEU DUNG: dem o centroid toan 0 (cuML cap phat du K hang nhung
            # khong dung het). "Cung K danh nghia" khong keo theo "cung so cluster thuc
            # dung" — neu hai nhanh lang phi khac nhau thi so sanh accuracy da lech san.
            nz = 0
            tot = 0
            for lyr in cen:
                v = cen[lyr].float()
                n = v.reshape(-1, v.shape[-1]).norm(dim=-1)
                nz += int((n == 0).sum()); tot += n.numel()
            zpct = 100.0 * nz / max(tot, 1)
            print(f"    dataidx {idx:3d}: centroids {tuple(c0.shape)} {'✅' if okc else '❌'}"
                  f" · labels {tuple(l0.shape)} (mong {n_ctx}) {'✅' if okl else '❌'}"
                  f" · o rong {zpct:.1f}%")

        if "A" not in checks:
            continue
        # Kiem A cho MOI nhanh, ke ca `sa`. Nhanh `sa` la NHOM DOI CHUNG: K-means tu do
        # phai vat qua bien o gan nhu moi cluster. Khong co con so do thi "0 vi pham" cua
        # hard_boundary chua chung minh duoc gi — biet dau du lieu nay von it unit den muc
        # moi cach cluster deu khong vat bien.
        expect0 = discover_method(path, fallback=name) in HARD
        print(f"\n[A] Ranh gioi cung — {name}"
              + ("" if expect0 else "   (NHOM DOI CHUNG — mong doi vat bien NHIEU)"))
        if not prompts:
            print("    (dung lai prompt sau truncation — can dataset + tokenizer)")
            prompts.update(rebuild_prompts(
                args.model, args.dataset, set().union(*(set(d) for d in found.values())), meta))
        for idx in sorted(found[name]):
            K = found[name][idx]
            rec = meta.get(idx)
            offs = npz[f"offsets_{idx}"]
            n_ctx = rec["shared_prefix_length"] - args.observation_window
            pr, sp = prompts[idx]
            if sp != rec["shared_prefix_length"]:
                print(f"    dataidx {idx}: [!] sp_len dung lai {sp} != meta "
                      f"{rec['shared_prefix_length']}")
                rc = 1
                continue
            uid, err = unit_ids_for(rec, offs, n_ctx, args.level, pr)
            if err:
                print(f"    dataidx {idx}: [!] {err}"); rc = 1; continue
            lab = torch.load(os.path.join(path, f"centroids_labels_dict_{idx}_{K}.pt"),
                             map_location="cpu")
            nb, nt, worst = check_hard_boundary(lab, uid, K)
            if nb is None:
                print(f"    dataidx {idx}: [!] {worst}"); rc = 1; continue
            pct = 100.0 * nb / nt if nt else 0.0
            if expect0:
                ok = nb == 0
                rc |= 0 if ok else 1
                mark = "✅" if ok else "❌"
            else:
                # nhanh doi chung: KHONG tinh vao ma thoat, chi de doi chieu
                mark = "(doi chung)"
            print(f"    dataidx {idx:3d}: {nt:7d} cluster · vat qua >1 unit: {nb:6d}"
                  f" ({pct:5.1f}%)  {mark}  (U={int(uid.max())+1})")
            if worst:
                lyr, h, k, lo, hi = worst
                print(f"        vi du: layer {lyr} head {h} cluster {k} chua unit {lo}..{hi}")

    # ---------- E. TANG L1 (chi struct_hierarchy) ----------
    for name, path in (dirs.items() if "E" in checks else []):
        l1_found = discover_l1(path)
        if not l1_found:
            if name == "struct_hierarchy":
                print(f"\n[E] Tang L1 — {name}: [!] khong thay file hierarchical_*_L1_*.pt")
                rc = 1
            continue
        print(f"\n[E] Tang L1 — {name}   ({len(l1_found)} mau co file L1)")
        for idx in sorted(found[name]):
            K2 = found[name][idx]
            rec = meta.get(idx)
            if rec is None:
                continue
            n_ctx = rec["shared_prefix_length"] - args.observation_window
            cen_l2 = torch.load(os.path.join(path, f"centroids_tensor_dict_{idx}_{K2}.pt"),
                                map_location="cpu")
            lab_l2 = torch.load(os.path.join(path, f"centroids_labels_dict_{idx}_{K2}.pt"),
                                map_location="cpu")
            ok, K1, det = check_l1_layer(path, idx, K2, n_ctx, cen_l2, lab_l2, args.rtol_set)
            if ok is None:
                print(f"    dataidx {idx:3d}: [!] {det['err']}"); rc = 1; continue
            rc |= 0 if ok else 1
            meta_pct = 100.0 * (K1 + K2) / n_ctx
            l2_pct = 100.0 * K2 / n_ctx
            print(f"    dataidx {idx:3d}: K1={K1:4d} K2={K2:4d} (K1<=K2 {'✅' if K1 <= K2 else '❌'})"
                  f" · mode={det.get('mode','?'):5s}"
                  f" · centroid L1 vs tb-trong-so rel={det['worst_rel_E5']:.1e}"
                  f" · metadata (K1+K2)/n_ctx={meta_pct:.1f}% (L2 rieng {l2_pct:.1f}%)"
                  f"   {'✅' if ok else '❌'}")
            for f in det["fails"]:
                print(f"        {f}")

    # ---------- D. NHANH sa TRUNG BAN GOC ----------
    if "D" in checks and args.reference_dir and "sa" in dirs:
        print("\n[D] Nhanh `sa` so voi offline_clustering.py")
        for idx in sorted(found["sa"]):
            K = found["sa"][idx]
            fa = os.path.join(dirs["sa"], f"centroids_tensor_dict_{idx}_{K}.pt")
            fb = os.path.join(args.reference_dir, f"centroids_tensor_dict_{idx}_{K}.pt")
            if not os.path.exists(fb):
                print(f"    dataidx {idx:3d}: (khong co ban goc de doi chieu)"); continue
            a = torch.load(fa, map_location="cpu")
            b = torch.load(fb, map_location="cpu")
            # Centroid la TAP HOP, khong phai day co thu tu: K-means danh so cluster tuy y,
            # nen hai lan chay cho cung phan hoach van xep centroid khac thu tu. So theo vi
            # tri se luon lech du hai ben giong het nhau ve mat toan hoc.
            # Do dung: voi moi centroid ben A, tim centroid GAN NHAT ben B, lay max cac
            # khoang cach do — bat bien voi hoan vi.
            # O CENTROID RONG: cuML cap phat du K hang nhung khong phai hang nao cung duoc
            # dung; hang khong dung o lai toan 0. Do tren Qwen2 (dataidx 0): lop 27 co
            # 21.7% hang zero, lop 0 co 3.3%, cac lop khac 0% — VA CA HAI BEN deu vay, tuc
            # day la hanh vi cua code goc chu khong phai loi cua ban port. (LongChat la MHA
            # 32 head KV nen ty le co the khac; con so tren chi la vi du.)
            #
            # Hai he qua:
            #   1. Khong duoc dua hang zero vao phep so: chung khong mang thong tin, va neu
            #      mot head co qua nua hang zero thi median-norm = 0 -> chia cho 1e-12 ->
            #      "khoang cach" nhay len 1e13 va bao FAIL gia. Chinh la ca gap 21/8.
            #   2. Ty le hang zero LA MOT SO LIEU CAN BAO CAO: ngan sach danh nghia K khong
            #      bang ngan sach hieu dung. Neu hai nhanh lang phi khac nhau thi so sanh
            #      "cung budget" da lech san truoc khi do accuracy.
            worst_rel = 0.0
            shape_bad = False
            n_zero_a = n_zero_b = n_row = 0
            for lyr in a:
                x, y = a[lyr].float(), b[lyr].float()
                if x.shape != y.shape:
                    shape_bad = True
                    break
                x2, y2 = x.reshape(-1, x.shape[-2], x.shape[-1]), y.reshape(-1, y.shape[-2], y.shape[-1])
                for h in range(x2.shape[0]):
                    na, nb = x2[h].norm(dim=-1), y2[h].norm(dim=-1)
                    ma, mb = na > 0, nb > 0
                    n_zero_a += int((~ma).sum()); n_zero_b += int((~mb).sum())
                    n_row += ma.numel()
                    if int(ma.sum()) == 0 or int(mb.sum()) == 0:
                        continue                      # head rong hoan toan, khong so duoc
                    xa, yb = x2[h][ma], y2[h][mb]
                    d = torch.cdist(xa, yb)                  # [na, nb]
                    nearest = d.min(dim=1).values.max()      # Hausdorff mot chieu
                    scale = nb[mb].median().clamp_min(1e-12)
                    worst_rel = max(worst_rel, float(nearest / scale))
            if shape_bad:
                print(f"    dataidx {idx:3d}: SHAPE LECH  ❌"); rc = 1; continue
            ok = worst_rel <= args.rtol_set
            rc |= 0 if ok else 1
            za = 100.0 * n_zero_a / max(n_row, 1)
            zb = 100.0 * n_zero_b / max(n_row, 1)
            print(f"    dataidx {idx:3d}: khoang cach tap hop (chuan hoa) {worst_rel:.3e}"
                  f"  {'✅' if ok else '❌'}   o centroid rong: sa {za:.1f}% · goc {zb:.1f}%")

    print("\n" + "=" * 72)
    print("  ✅ MOI BAT BIEN QUA" if rc == 0 else "  ❌ CO BAT BIEN BI VI PHAM")
    print("=" * 72)
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
