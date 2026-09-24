"""
phase4_incremental.py — Phase 4 / C3: đo t_incr so với t_full khi sửa MỘT class.

CLAIM C3 (protocol): sửa code thì chỉ cluster lại unit bị đụng -> nhanh hơn nhiều lần, mất
< 0.3 điểm. Script này đo CẢ HAI vế, không chỉ vế thời gian. Một con số speedup không đi kèm
cái giá về chất lượng thì không phải bằng chứng cho C3.

HAI CHẾ ĐỘ
----------
  --mode real       Dữ liệu thật (RepoBench-P / LCC), model thật, GPU. Với mỗi mẫu:
                    chọn một class, chèn `--edit_lines` dòng comment vào thân class, rồi đo
                    cập nhật centroid cho bản mới theo 3 cách. Giới hạn bởi context của model
                    (LongChat: 31.500 token).
  --mode synthetic  CHỈ đo phần clustering, key ngẫu nhiên, tới 128K token. Dùng để trả lời
                    "ở 128K thì sao" khi model không chạy được tới đó. Phải ghi rõ là tổng hợp.

CÁC KHOẢN THỜI GIAN (mode real), mỗi khoản đo riêng, cuda.synchronize hai đầu
---------------------------------------------------------------------------
  fwd_full      forward toàn bộ prompt mới (để có key)            — bắt buộc cho full
  fwd_incr      forward CHỈ phần từ token sửa đầu tiên e, dùng lại KV cache của tiền tố
                (tiền tố [0, e) không đổi vì LLM là causal)       — bắt buộc cho incr
  hb_full       hard_boundary trên mọi unit
  sa_full       run_clustering (cuML, bài gốc) trên mọi key      — chỉ khi --with_sa
  hb_local      P1: chỉ cluster lại unit chứa chỗ sửa (XẤP XỈ, xem incremental_clustering.py)
  hb_suffix     P2: cluster lại mọi unit có token >= e (chính xác, cùng k_u)
  thr_full / thr_incr   run_global_threshold — τ là quantile trên MỌI token nên luôn phải
                tính lại, incremental không tránh được

  speedup cluster-only = hb_full / hb_local           <- con số "vài giây vs vài phút"
  speedup end-to-end   = (fwd_full + hb_full + thr_full) / (fwd_incr + hb_local + thr_incr)

Báo cáo cả hai. Chỉ báo con số cluster-only là giấu phần forward mà incremental không bỏ được.

CHẤT LƯỢNG (cùng giao thức C2 của phase5_recall.py: top-N, lớp first/mid/last, sp 70/80/90)
-------------------------------------------------------------------------------------------
  hb_full vs hb_local vs hb_suffix, và (nếu --with_sa) sa_full vs sa_stale (SA không cluster
  lại, token mới gán về centroid cũ gần nhất). Δrecall = cái giá của phép xấp xỉ P1.
  Cộng thêm độ trôi key: key của token SAU chỗ sửa thay đổi bao nhiêu (%) — đó là lý do P1
  là xấp xỉ.

Không chạy cái gì trên pod mà chưa qua `python scripts/test_incremental.py` (CPU).

Ví dụ:
    python phase4_incremental.py longchat-v1.5-7b-32k --mode real --dataset repobench-p \\
        --limit 100 --edit_pos middle --edit_lines 1 --out phase4_out/rb_mid_l1.jsonl
    python phase4_incremental.py longchat-v1.5-7b-32k --mode synthetic \\
        --lengths 16384 32768 65536 131072 --out phase4_out/synthetic.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from struct_clustering import parse_units, hard_boundary_kmeans, compact_unit_ids  # noqa: E402
from incremental_clustering import (  # noqa: E402
    make_edit, diff_region, token_map_new_to_old, map_units, unit_layout, incremental_k,
    incremental_hard_boundary, sa_assign_stale,
)


# =====================================================================
# tiện ích
# =====================================================================

def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev):
    _sync(dev)
    t = time.perf_counter()
    out = fn()
    _sync(dev)
    return out, time.perf_counter() - t


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(path):
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.add(json.loads(line)["key"])
                except Exception:
                    pass  # dòng cuối ghi dở
    return out


def eval_layers(L):
    return sorted({0, L // 2, L - 1})


class Capture:
    """
    Hook trên layer.self_attn, giống offline_clustering_struct.py nhưng:
      - q chỉ giữ observation window (run_global_threshold và phase5 chỉ dùng phần đó)
      - hiểu được forward incremental: q chỉ phủ [q_off, S), còn k (sau cache.update) phủ [0, S)
    """

    def __init__(self, W):
        self.W, self.active, self.sp, self.q_off = W, False, 0, 0
        self.k, self.q = [], []

    def arm(self, sp, q_off=0):
        self.sp, self.q_off, self.active = sp, q_off, True
        self.k, self.q = [], []

    def hook(self, module, inp, out):
        if not self.active:
            return
        _, qkv, _ = out
        q, k, _v = qkv
        assert k.shape[2] >= self.sp, (k.shape, self.sp)
        lo = self.sp - self.W - self.q_off
        assert lo >= 0, "chỗ sửa nằm trong observation window — không hợp lệ"
        self.k.append(k[:, :, :self.sp].detach())
        self.q.append(q[:, :, lo:lo + self.W].detach().clone())

    def take(self):
        self.active = False
        k, q = self.k, self.q
        self.k, self.q = [], []
        return k, q


# =====================================================================
# MODE REAL
# =====================================================================

def pick_class_span(prompt, rec, offs, n_ctx, edit_pos, min_tokens):
    """Chọn class để sửa. Chỉ nhận class nằm gọn trong phần được cluster (< n_ctx)."""
    cs, ce = rec["code_char_start"], rec["code_char_end"]
    code = prompt[cs:ce]
    spans, _ = parse_units(code, rec["language"], "class")
    starts = offs[:, 0]
    ctx_end_char = int(offs[n_ctx - 1, 0])
    cands = []
    for s, e in spans:
        if (s, e) == (0, len(code)):
            continue
        s, e = s + cs, e + cs
        if e >= ctx_end_char:
            continue
        ntok = int(((starts >= s) & (starts < e)).sum())
        if ntok >= min_tokens:
            cands.append((s, e, ntok))
    if not cands:
        return None
    cands.sort()
    i = {"first": 0, "middle": len(cands) // 2, "last": len(cands) - 1}[edit_pos]
    return cands[i]


def run_real(args, dev):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset
    from squeezedattention.utils import truncate_fn, apply_rope_scaling
    from squeezedattention.clustering import run_clustering, run_global_threshold
    from offline_clustering_struct import load_phase1, build_unit_ids
    from phase5_recall import recall_one_sample

    m2p = json.load(open("LongBench/config/model2path.json", encoding="utf-8"))
    m2l = json.load(open("LongBench/config/model2maxlen.json", encoding="utf-8"))
    d2p = json.load(open("LongBench/config/dataset2prompt.json", encoding="utf-8"))
    model_path, max_length = m2p[args.model], m2l[args.model]

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tok_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    config = apply_rope_scaling(config, args.rope_scaling)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    if getattr(config, "use_sliding_window", False):
        config.use_sliding_window = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                 torch_dtype=torch.bfloat16).eval().to(dev)
    L = config.num_hidden_layers
    EL = eval_layers(L)
    W = args.observation_window

    cap = Capture(W)
    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(cap.hook)

    prompt_format = d2p[args.dataset]
    prompt_only_format = d2p[args.dataset + "_prompt_full"]
    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    n = min(args.limit, len(data)) if args.limit > 0 else len(data)
    meta, _ = load_phase1(args.phase1_dir, args.dataset, args.model, expect_n=n,
                          expect_mode="full", expect_chat=False)
    ns = SimpleNamespace(token_weights=False)

    def ids_of(text):
        return tok(text, truncation=False, return_tensors="pt").input_ids

    def offsets_of(text, ids):
        enc = tok_fast(text, truncation=False, return_offsets_mapping=True, add_special_tokens=True)
        if list(enc["input_ids"]) != ids[0].tolist():
            return None
        return np.asarray(enc["offset_mapping"], dtype=np.int64)

    @torch.no_grad()
    def fwd_full(ids, sp):
        cap.arm(sp, 0)
        model.model(ids.to(dev), use_cache=False)
        return cap.take()

    @torch.no_grad()
    def prefix_cache(ids, e):
        was = cap.active
        cap.active = False
        out = model.model(ids[:, :e].to(dev), use_cache=True)
        cap.active = was
        return out.past_key_values                      # legacy tuple, không bị update làm bẩn

    @torch.no_grad()
    def fwd_incr(ids, e, cache, sp):
        cap.arm(sp, e)
        model.model(ids[:, e:].to(dev), past_key_values=cache, use_cache=True)
        return cap.take()

    def hb_all(ks, uid, K, n_ctx, k_per_unit=None):
        cent, lab = {}, {}
        for li, k in enumerate(ks):
            c, l, _ = hard_boundary_kmeans(k[0, :, :n_ctx].float(), uid, K, n_iter=args.n_iter,
                                           device=dev, k_per_unit=k_per_unit)
            cent[li], lab[li] = c, l
        return cent, lab

    def hb_incr(ks, uid_new, k_new, recl, cent_old, lab_old, k_old, new2old, tmap, n_ctx):
        cent, lab, st = {}, {}, None
        for li, k in enumerate(ks):
            c, l, st = incremental_hard_boundary(
                k[0, :, :n_ctx].float(), uid_new, k_new, recl, cent_old[li], lab_old[li],
                k_old, new2old, tmap, n_iter=args.n_iter, device=dev)
            cent[li], lab[li] = c, l
        return cent, lab, st

    def recall(q_obs, k_ctx, cent, lab):
        r = recall_one_sample(q_obs, k_ctx, cent.squeeze(0).float(), lab.squeeze(0), args.sparsity)
        return {str(sp): v[0] for sp, v in r.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = done_keys(args.out)
    warmed = False
    n_ok = 0

    for idx in range(n):
        key = f"{args.dataset}:{idx}:{args.edit_pos}:{args.edit_lines}"
        if key in done:
            continue
        rec = meta[idx]
        base = {"key": key, "dataidx": idx, "dataset": args.dataset, "model": args.model,
                "edit_pos": args.edit_pos, "edit_lines": args.edit_lines,
                "language": rec["language"]}

        def skip(reason, **kw):
            append_jsonl(args.out, {**base, "status": "skip", "reason": reason, **kw})

        if rec["truncated"]:
            skip("truncated")
            continue
        d = data[idx]
        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, sp_old = truncate_fn(prompt, prompt_only, tok, max_length, args.dataset, dev,
                                     model_name=args.model)
        if sp_old != rec["shared_prefix_length"]:
            raise SystemExit(f"[ERROR] idx {idx}: sp {sp_old} != Phase 1.4 "
                             f"{rec['shared_prefix_length']}")
        ids_old = ids_of(prompt)
        offs_old = offsets_of(prompt, ids_old)
        if offs_old is None:
            skip("fast_slow_mismatch_old")
            continue
        n_ctx_old = sp_old - W

        pick = pick_class_span(prompt, rec, offs_old, n_ctx_old, args.edit_pos,
                               args.min_class_tokens)
        if pick is None:
            skip("no_class")
            continue
        cls_s, cls_e, cls_tok = pick
        ed = make_edit(prompt, (cls_s, cls_e), rec["language"], args.edit_lines)
        if ed is None or ed[0] >= len(prompt_only):
            skip("no_edit_point")
            continue
        pos, ins = ed
        prompt_new = prompt[:pos] + ins + prompt[pos:]
        prompt_only_new = prompt_only[:pos] + ins + prompt_only[pos:]
        ids_new = ids_of(prompt_new)
        if ids_new.shape[1] > max_length:
            skip("edited_exceeds_max_length")
            continue
        _, sp_new = truncate_fn(prompt_new, prompt_only_new, tok, max_length, args.dataset, dev,
                                model_name=args.model)
        offs_new = offsets_of(prompt_new, ids_new)
        if offs_new is None:
            skip("fast_slow_mismatch_new")
            continue
        n_ctx_new = sp_new - W

        e, old_end, new_end = diff_region(ids_old[0], ids_new[0])
        if not (0 < e and new_end <= n_ctx_new and old_end <= n_ctx_old):
            skip("edit_outside_ctx", e=e, new_end=new_end, n_ctx_new=n_ctx_new)
            continue

        rec_new = dict(rec, code_char_end=rec["code_char_end"] + len(ins),
                       shared_prefix_length=sp_new)
        uid_old, _, _ = build_unit_ids(prompt, rec, offs_old, sp_old, W, "class", ns)
        uid_new, _, _ = build_unit_ids(prompt_new, rec_new, offs_new, sp_new, W, "class", ns)
        uid_old, _ = compact_unit_ids(uid_old)
        uid_new, _ = compact_unit_ids(uid_new)
        tmap = token_map_new_to_old(n_ctx_new, n_ctx_old, e, old_end, new_end)
        new2old, changed = map_units(uid_old, uid_new, tmap)

        K_old = max(1, int(args.percent_clusters / 100.0 * n_ctx_old))
        K_new = max(1, int(args.percent_clusters / 100.0 * n_ctx_new))
        try:
            uid_old, sizes_old, k_old = unit_layout(uid_old, K_old)
            _, sizes_new, _ = unit_layout(uid_new, K_new)
        except ValueError as ex:          # BudgetExceeded — ở level class gần như không gặp
            skip("budget_exceeded", msg=str(ex))
            continue
        k_inc = incremental_k(k_old, sizes_old, sizes_new, new2old, changed, n_ctx_new,
                              args.percent_clusters)
        recl_local = changed.clone()
        recl_suffix = changed.clone()
        recl_suffix[torch.unique(uid_new[e:])] = True

        # ---- khởi động GPU một lần (kernel, allocator, cuML) — không tính ----
        if not warmed:
            ks, _ = fwd_full(ids_new, sp_new)
            hb_all(ks[:1], uid_new, K_new, n_ctx_new)
            del ks
            warmed = True
        torch.cuda.reset_peak_memory_stats(dev)

        # ================= BẢN CŨ (không tính giờ: đã có sẵn trên đĩa khi sửa code) =========
        ks_old, _ = fwd_full(ids_old, sp_old)
        cent_old, lab_old = hb_all(ks_old, uid_old, K_old, n_ctx_old, k_per_unit=k_old)
        if args.with_sa:
            sa_cent_old, sa_lab_old = run_clustering(ks_old, K_old, observation_window=W,
                                                     device=dev, seed=0)
        k_old_eval = {li: ks_old[li][0, :, :n_ctx_old].float() for li in EL}
        del ks_old

        T = {}
        # ================= FULL: làm lại từ đầu trên bản mới ================================
        (ks_new, qs_new), T["fwd_full"] = timed(lambda: fwd_full(ids_new, sp_new), dev)
        (cent_f, lab_f), T["hb_full"] = timed(lambda: hb_all(ks_new, uid_new, K_new, n_ctx_new), dev)
        _, T["thr_full"] = timed(lambda: run_global_threshold(
            ks_new, qs_new, cent_f, lab_f, K_new, observation_window=W, device=dev), dev)
        if args.with_sa:
            (sa_cent_f, sa_lab_f), T["sa_full"] = timed(lambda: run_clustering(
                ks_new, K_new, observation_window=W, device=dev, seed=0), dev)

        # ================= INCREMENTAL ======================================================
        cache = prefix_cache(ids_new, e)
        (ks_inc, qs_inc), T["fwd_incr"] = timed(lambda: fwd_incr(ids_new, e, cache, sp_new), dev)
        del cache
        (cent_l, lab_l, st_l), T["hb_local"] = timed(lambda: hb_incr(
            ks_inc, uid_new, k_inc, recl_local, cent_old, lab_old, k_old, new2old, tmap,
            n_ctx_new), dev)
        (cent_s, lab_s, st_s), T["hb_suffix"] = timed(lambda: hb_incr(
            ks_inc, uid_new, k_inc, recl_suffix, cent_old, lab_old, k_old, new2old, tmap,
            n_ctx_new), dev)
        K_inc = int(k_inc.sum())
        _, T["thr_incr"] = timed(lambda: run_global_threshold(
            ks_inc, qs_inc, cent_l, lab_l, K_inc, observation_window=W, device=dev), dev)

        # ================= KIỂM TRA + CHẤT LƯỢNG (không tính giờ) ===========================
        chk, drift, Q = {}, {}, {}
        for li in EL:
            kf = ks_new[li][0, :, :n_ctx_new].float()
            ki = ks_inc[li][0, :, :n_ctx_new].float()
            ko = k_old_eval[li]
            chk[f"L{li}_incr_vs_full_key_relmax"] = float(
                (ki - kf).norm(dim=-1).max() / kf.norm(dim=-1).mean())
            pre = torch.arange(e, device=dev)
            chk[f"L{li}_prefix_key_relmax"] = float(
                (kf[:, pre] - ko[:, pre]).norm(dim=-1).max() / ko.norm(dim=-1).mean())
            delta = new_end - old_end
            j = torch.arange(new_end, min(n_ctx_new, n_ctx_old + delta), device=dev)
            if j.numel():
                dk = (kf[:, j] - ko[:, j - delta]).norm(dim=-1)
                drift[f"L{li}"] = float((dk / ko[:, j - delta].norm(dim=-1)).mean())

            # P2 phải khớp full chạy với CÙNG k_u (cùng key incremental)
            c_ref, l_ref, _ = hard_boundary_kmeans(ki, uid_new, K_inc, n_iter=args.n_iter,
                                                   device=dev, k_per_unit=k_inc)
            chk[f"L{li}_suffix_label_agree"] = float((lab_s[li] == l_ref).float().mean())

            q_obs = qs_new[li][0].float()
            Q[f"L{li}"] = {"hb_full": recall(q_obs, kf, cent_f[li], lab_f[li]),
                           "hb_local": recall(q_obs, kf, cent_l[li], lab_l[li]),
                           "hb_suffix": recall(q_obs, kf, cent_s[li], lab_s[li])}
            if args.with_sa:
                sa_stale = sa_assign_stale(kf, sa_cent_old[li], sa_lab_old[li], tmap)
                Q[f"L{li}"]["sa_full"] = recall(q_obs, kf, sa_cent_f[li], sa_lab_f[li])
                Q[f"L{li}"]["sa_stale"] = recall(q_obs, kf, sa_cent_old[li], sa_stale)

        out = {**base, "status": "ok",
               "n_ctx_old": n_ctx_old, "n_ctx_new": n_ctx_new, "e": e,
               "edit_frac": e / n_ctx_old, "inserted_tokens": new_end - e,
               "class_tokens": cls_tok, "units": int(sizes_new.numel()),
               "units_changed": int(changed.sum()),
               "units_local": st_l["units_reclustered"], "units_suffix": st_s["units_reclustered"],
               "tokens_local": st_l["tokens_reclustered"], "tokens_suffix": st_s["tokens_reclustered"],
               "K_new": K_new, "K_incr": K_inc, "times_s": T, "checks": chk,
               "key_drift_after_edit": drift, "recall": Q,
               "peak_alloc_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30}
        append_jsonl(args.out, out)
        n_ok += 1
        sp = (T["fwd_full"] + T["hb_full"] + T["thr_full"]) / (
            T["fwd_incr"] + T["hb_local"] + T["thr_incr"])
        print(f"[{idx}] n_ctx={n_ctx_new} e={e} ({100 * e / n_ctx_old:.0f}%) "
              f"hb_full={T['hb_full']:.2f}s hb_local={T['hb_local']:.3f}s "
              f"(x{T['hb_full'] / max(T['hb_local'], 1e-9):.1f}) e2e x{sp:.2f}")

        del ks_new, qs_new, ks_inc, qs_inc, cent_f, lab_f, cent_l, lab_l, cent_s, lab_s
        del cent_old, lab_old, k_old_eval
        if args.with_sa:
            del sa_cent_old, sa_lab_old, sa_cent_f, sa_lab_f
        torch.cuda.empty_cache()

    print(f"\n>>> Xong {n_ok} mẫu mới -> {args.out}")


# =====================================================================
# MODE SYNTHETIC — chỉ clustering, tới 128K
# =====================================================================

def run_synthetic(args, dev):
    """
    Key ngẫu nhiên N(0,1), unit liền kề cỡ `--syn_unit_tokens`. Đo MỘT layer rồi nhân L.
    hard_boundary chạy đúng n_iter vòng không phụ thuộc dữ liệu, nên thời gian của nó ít phụ
    thuộc phân phối key. cuML (SA) dừng theo hội tụ nên PHỤ THUỘC dữ liệu — số SA ở đây chỉ là
    cỡ độ lớn. Không có forward: model 32K không chạy được 128K.
    """
    try:
        from squeezedattention.clustering import run_clustering, run_global_threshold
        have_cuml = True
    except Exception as ex:                                   # CPU / máy không có RAPIDS
        print(f"[WARN] không import được squeezedattention.clustering ({ex}) -> bỏ SA + threshold")
        have_cuml = False

    H, D, L, W = args.syn_heads, args.syn_head_dim, args.syn_layers, args.observation_window
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for S in args.lengths:
        n_ctx = S - W
        g = torch.Generator().manual_seed(0)
        keys = torch.randn(H, S, D, generator=g).to(dev)
        q = torch.randn(1, H, W, D, generator=g).to(dev)
        n_units = max(1, n_ctx // args.syn_unit_tokens)
        uid = torch.minimum(torch.arange(n_ctx) // args.syn_unit_tokens,
                            torch.tensor(n_units - 1))
        K = max(1, int(args.percent_clusters / 100.0 * n_ctx))
        _, sizes, k = unit_layout(uid, K)
        mid = n_units // 2
        tok_mid = (uid == mid).nonzero(as_tuple=True)[0].to(dev)
        kc = keys[:, :n_ctx].float()

        hard_boundary_kmeans(kc[:, tok_mid], torch.zeros_like(tok_mid), int(k[mid]),
                             n_iter=args.n_iter, device=dev)                      # warmup
        T = {}
        (c, l, _), T["hb_full_1layer"] = timed(lambda: hard_boundary_kmeans(
            kc, uid, K, n_iter=args.n_iter, device=dev, k_per_unit=k), dev)
        _, T["hb_local_1layer"] = timed(lambda: hard_boundary_kmeans(
            kc[:, tok_mid], torch.zeros(tok_mid.numel(), dtype=torch.long), int(k[mid]),
            n_iter=args.n_iter, device=dev), dev)
        if have_cuml:
            _, T["thr_1layer"] = timed(lambda: run_global_threshold(
                {0: keys.unsqueeze(0)}, {0: q}, {0: c}, {0: l}, K, observation_window=W,
                device=dev), dev)
            _, t_sa = timed(lambda: run_clustering({0: keys[:args.syn_sa_heads].unsqueeze(0)},
                                                   K, observation_window=W, device=dev), dev)
            T["sa_full_1layer_est"] = t_sa * H / args.syn_sa_heads
        rec = {"mode": "synthetic", "S": S, "n_ctx": n_ctx, "H": H, "L": L, "K": K,
               "units": n_units, "unit_tokens": args.syn_unit_tokens, "times_s": T,
               "est_all_layers_s": {k_: v * L for k_, v in T.items()},
               "sa_heads_measured": args.syn_sa_heads if have_cuml else 0,
               "device": str(dev)}
        append_jsonl(args.out, rec)
        e2 = rec["est_all_layers_s"]
        line = (f"S={S:>6} K={K:>5} hb_full≈{e2['hb_full_1layer']:.1f}s "
                f"hb_local≈{e2['hb_local_1layer']:.3f}s "
                f"(x{T['hb_full_1layer'] / max(T['hb_local_1layer'], 1e-9):.0f})")
        if have_cuml:
            line += f" sa_full≈{e2['sa_full_1layer_est']:.0f}s thr≈{e2['thr_1layer']:.1f}s"
        print(line)
        del keys, kc, c, l
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# =====================================================================
# TỔNG HỢP
# =====================================================================

def summarize(path, sparsity, B=20000, seed=0):
    """Trung vị thời gian/speedup + Δrecall ghép cặp theo mẫu, bootstrap như Phase 5/6."""
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    ok = [r for r in rows if r.get("status") == "ok"]
    skips = {}
    for r in rows:
        if r.get("status") == "skip":
            skips[r["reason"]] = skips.get(r["reason"], 0) + 1
    print(f"\n=== {path}: {len(ok)} mẫu ok, bỏ qua {skips} ===")
    if not ok:
        return

    def med(xs):
        return statistics.median(xs), np.percentile(xs, 25), np.percentile(xs, 75)

    T = lambda r, k: r["times_s"].get(k)  # noqa: E731
    print("\nThời gian (s), trung vị [p25, p75]:")
    for k in ["fwd_full", "fwd_incr", "hb_full", "hb_local", "hb_suffix", "thr_full",
              "thr_incr", "sa_full"]:
        xs = [T(r, k) for r in ok if T(r, k) is not None]
        if xs:
            m, a, b = med(xs)
            print(f"  {k:10s} {m:9.3f}  [{a:.3f}, {b:.3f}]")

    def spd(num, den):
        return [sum(T(r, k) for k in num) / sum(T(r, k) for k in den) for r in ok
                if all(T(r, k) is not None for k in num + den)]

    print("\nSpeedup, trung vị [p25, p75] (tính theo từng mẫu):")
    for name, num, den in [
        ("cluster-only  hb_full/hb_local", ["hb_full"], ["hb_local"]),
        ("cluster-only  hb_full/hb_suffix", ["hb_full"], ["hb_suffix"]),
        ("cluster-only  sa_full/hb_local", ["sa_full"], ["hb_local"]),
        ("end-to-end    local  (vs HB full)", ["fwd_full", "hb_full", "thr_full"],
         ["fwd_incr", "hb_local", "thr_incr"]),
        ("end-to-end    suffix (vs HB full)", ["fwd_full", "hb_full", "thr_full"],
         ["fwd_incr", "hb_suffix", "thr_incr"]),
        ("end-to-end    local  (vs SA full)", ["fwd_full", "sa_full", "thr_full"],
         ["fwd_incr", "hb_local", "thr_incr"]),
    ]:
        xs = spd(num, den)
        if xs:
            m, a, b = med(xs)
            print(f"  {name:36s} x{m:8.2f}  [x{a:.2f}, x{b:.2f}]")

    ef = [r["edit_frac"] for r in ok]
    print(f"\nVị trí sửa e/n_ctx: trung vị {statistics.median(ef):.2f} · "
          f"token cluster lại P1 / P2 (trung vị): "
          f"{statistics.median([r['tokens_local'] for r in ok])} / "
          f"{statistics.median([r['tokens_suffix'] for r in ok])} "
          f"trên {statistics.median([r['n_ctx_new'] for r in ok])}")
    drift = [np.mean(list(r["key_drift_after_edit"].values())) for r in ok
             if r["key_drift_after_edit"]]
    if drift:
        print(f"Độ trôi key sau chỗ sửa (‖Δk‖/‖k‖, TB lớp): trung vị {100 * np.median(drift):.2f}%")
    agree = [v for r in ok for k, v in r["checks"].items() if k.endswith("suffix_label_agree")]
    print(f"Kiểm tra P2 == full cùng k_u (tỉ lệ nhãn trùng): min {min(agree):.4f}")

    rng = np.random.default_rng(seed)
    print(f"\nΔrecall ghép cặp (điểm %), TB 3 lớp, bootstrap B={B}:")
    for a, b in [("hb_local", "hb_full"), ("hb_suffix", "hb_full"), ("sa_stale", "sa_full"),
                 ("hb_local", "sa_full")]:
        for sp in sparsity:
            diffs = []
            for r in ok:
                vals = [(v[a][str(sp)], v[b][str(sp)]) for v in r["recall"].values()
                        if a in v and b in v]
                if vals:
                    diffs.append(100 * np.mean([x - y for x, y in vals]))
            if not diffs:
                continue
            d = np.asarray(diffs)
            boot = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {a:9s} − {b:8s} sp{sp}: {d.mean():+.3f}  [{lo:+.3f}; {hi:+.3f}]  n={len(d)}")


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="longchat-v1.5-7b-32k")
    ap.add_argument("--mode", choices=["real", "synthetic", "summary"], default="real")
    ap.add_argument("--out", default="phase4_out/phase4.jsonl")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--percent_clusters", type=float, default=5)
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--sparsity", type=int, nargs="+", default=[70, 80, 90])
    # real
    ap.add_argument("--dataset", default="repobench-p", choices=["repobench-p", "lcc"])
    ap.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    ap.add_argument("--rope_scaling", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--edit_pos", choices=["first", "middle", "last"], default="middle",
                    help="class thứ mấy (theo vị trí) bị sửa. Quyết định kích thước phần phải "
                         "forward lại: first -> gần như toàn bộ, last -> ít")
    ap.add_argument("--edit_lines", type=int, default=1, help="số dòng comment chèn vào class")
    ap.add_argument("--min_class_tokens", type=int, default=50)
    ap.add_argument("--with_sa", action="store_true",
                    help="đo cả SA (cuML) full + SA stale. Tốn ~2 lượt cuML mỗi mẫu")
    # synthetic
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--syn_heads", type=int, default=32)
    ap.add_argument("--syn_head_dim", type=int, default=128)
    ap.add_argument("--syn_layers", type=int, default=32)
    ap.add_argument("--syn_unit_tokens", type=int, default=800,
                    help="cỡ unit tổng hợp. ~800 = cỡ class trung bình RepoBench-P "
                         "(~14K token / 17 class)")
    ap.add_argument("--syn_sa_heads", type=int, default=1,
                    help="số head cuML đo thật rồi nhân lên H — cuML ở 128K rất chậm")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize(args.out, args.sparsity)
        return
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.mode == "real":
        if dev.type != "cuda":
            raise SystemExit("[ERROR] --mode real cần GPU (flash-attention)")
        run_real(args, dev)
        summarize(args.out, args.sparsity)
    else:
        run_synthetic(args, dev)


if __name__ == "__main__":
    main()
