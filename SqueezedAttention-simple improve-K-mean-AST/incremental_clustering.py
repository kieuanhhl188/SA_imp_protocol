"""
incremental_clustering.py — Phase 4 (C3): tái phân cụm CỤC BỘ cho hard_boundary khi code bị sửa.

Ý tưởng (claim C3): với ranh giới cứng, cluster không bao giờ vắt qua hai unit, nên sửa một
class thì về NGUYÊN TẮC chỉ cần cluster lại class đó. Module này cài đúng thao tác đó và
không làm gì khác. Đo thời gian là việc của `phase4_incremental.py`.

ĐIỀU PHẢI NÓI RÕ KHI BÁO CÁO — incremental ở đây KHÔNG chính xác tuyệt đối
------------------------------------------------------------------------
LLM là causal: sửa token ở vị trí e làm đổi key của MỌI token sau e (qua attention của
các layer trên, và dịch vị trí RoPE nếu số token thay đổi). Nên có hai chính sách:

  local   (P1)  chỉ cluster lại unit chứa chỗ sửa. Các unit khác GIỮ centroid cũ, tính trên
                key cũ. Đây là phép XẤP XỈ. Cái giá phải đo bằng Δrecall so với cluster lại
                từ đầu, không được bỏ qua.
  suffix  (P2)  cluster lại mọi unit có token ở vị trí >= e. Chính xác (khớp cluster-lại-toàn-
                bộ với cùng k_u), nhưng ở level=class unit "phần còn lại của file" (import,
                code top-level) gần như luôn nằm sau e, nên P2 thường gần bằng full.

Ngoài ra, dù dùng chính sách nào thì vẫn phải chạy lại forward cho phần sau e (để có key mới)
và tính lại ngưỡng toàn cục τ (nó là quantile trên MỌI token). Hai khoản đó incremental không
tránh được. `phase4_incremental.py` đo riêng từng khoản.

Mọi hàm ở đây thuần tensor, chạy trên CPU được — xem scripts/test_incremental.py.
"""
from typing import Dict, Optional, Tuple

import torch

from struct_clustering import allocate_centroids, compact_unit_ids, hard_boundary_kmeans

# Comment một dòng theo ngôn ngữ. Chèn comment thay vì code thật để cây AST không đổi (không
# thêm/bớt unit) — thứ bị đo là chi phí, không phải ngữ nghĩa của chỗ sửa.
LINE_COMMENT = {"python": "#", "java": "//", "javascript": "//", "typescript": "//",
                "csharp": "//"}


def make_edit(text: str, span: Tuple[int, int], language: str, n_lines: int = 1,
              tag: str = "phase4 edit probe") -> Optional[Tuple[int, str]]:
    """
    Chèn `n_lines` dòng comment vào THÂN của unit có span [s, e) trong `text`.

    Vị trí chèn: đầu dòng ngay sau header — sau dòng đầu tiên với Python, sau dòng chứa `{`
    đầu tiên với ngôn ngữ dùng ngoặc nhọn. Thụt lề copy từ dòng kế tiếp nên code vẫn đúng
    cú pháp, và chỗ chèn nằm TRONG span của unit.

    Trả về (pos, chuỗi_chèn) hoặc None nếu unit không có thân nhiều dòng.
    """
    if language not in LINE_COMMENT:
        raise ValueError(f"chưa hỗ trợ ngôn ngữ '{language}'")
    s, e = span
    start = s
    if language != "python":
        brace = text.find("{", s, e)
        if brace < 0:
            return None
        start = brace
    nl = text.find("\n", start, e)
    if nl < 0 or nl + 1 >= e:
        return None
    pos = nl + 1
    k = pos
    while k < e and text[k] in " \t":
        k += 1
    indent = text[pos:k]
    c = LINE_COMMENT[language]
    ins = "".join(f"{indent}{c} {tag} {i}\n" for i in range(n_lines))
    return pos, ins


def diff_region(old_ids: torch.Tensor, new_ids: torch.Tensor) -> Tuple[int, int, int]:
    """
    Vùng token khác nhau giữa hai chuỗi id.

    Trả về (e, old_end, new_end):
      old[:e] == new[:e]                 (tiền tố chung dài nhất)
      old[old_end:] == new[new_end:]     (hậu tố chung dài nhất, không chồng lên tiền tố)
    Vùng đã đổi là old[e:old_end] -> new[e:new_end].
    """
    old_ids = old_ids.reshape(-1).cpu()
    new_ids = new_ids.reshape(-1).cpu()
    n_o, n_n = old_ids.numel(), new_ids.numel()
    m = min(n_o, n_n)
    neq = (old_ids[:m] != new_ids[:m]).nonzero()
    e = int(neq[0]) if neq.numel() else m
    rem = m - e
    o_rev = old_ids.flip(0)[:rem]
    n_rev = new_ids.flip(0)[:rem]
    neq = (o_rev != n_rev).nonzero()
    suf = int(neq[0]) if neq.numel() else rem
    return e, n_o - suf, n_n - suf


def token_map_new_to_old(n_new: int, n_old: int, e: int, old_end: int, new_end: int) -> torch.Tensor:
    """
    [n_new] long: chỉ số token cũ tương ứng với token mới j, hoặc -1 nếu j nằm trong vùng đã
    đổi (hoặc ngoài dải cũ). Tiền tố giữ nguyên chỉ số, hậu tố dịch đi Δ = new_end - old_end.
    """
    j = torch.arange(n_new)
    out = torch.full((n_new,), -1, dtype=torch.long)
    pre = j < e
    out[pre] = j[pre]
    suf = j >= new_end
    shifted = j - (new_end - old_end)
    ok = suf & (shifted >= 0) & (shifted < n_old)
    out[ok] = shifted[ok]
    return out


def map_units(uid_old: torch.Tensor, uid_new: torch.Tensor,
              tok_new2old: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ghép unit mới <-> unit cũ qua các cặp token tương ứng.

    uid_old [n_old], uid_new [n_new]: unit_id đã compact.
    Trả về:
      new2old [U_new] long, -1 nếu unit mới không có token tương ứng nào
      changed [U_new] bool, True nếu unit có token trong vùng đã đổi, hoặc số token khác unit
              cũ, hoặc token của nó ánh xạ về nhiều unit cũ. Unit changed phải cluster lại.
    """
    U_new = int(uid_new.max()) + 1
    U_old = int(uid_old.max()) + 1
    new2old = torch.full((U_new,), -1, dtype=torch.long)
    changed = torch.zeros(U_new, dtype=torch.bool)

    has = tok_new2old >= 0
    # token trong vùng đổi -> unit của nó changed
    changed[torch.unique(uid_new[~has])] = True

    pairs_new = uid_new[has]
    pairs_old = uid_old[tok_new2old[has]]
    for u in torch.unique(pairs_new).tolist():
        olds = torch.unique(pairs_old[pairs_new == u])
        if olds.numel() != 1:
            changed[u] = True
        new2old[u] = int(olds[0])

    sizes_new = torch.bincount(uid_new, minlength=U_new)
    sizes_old = torch.bincount(uid_old, minlength=U_old)
    mapped = new2old >= 0
    diff_size = torch.zeros(U_new, dtype=torch.bool)
    diff_size[mapped] = sizes_new[mapped] != sizes_old[new2old[mapped]]
    changed |= diff_size | ~mapped
    return new2old, changed


def unit_layout(unit_ids: torch.Tensor, num_centroids: int,
                max_k_per_unit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(uid compact, sizes, k_per_unit) — ĐÚNG phép chia mà hard_boundary_kmeans tự làm."""
    uid, _ = compact_unit_ids(unit_ids.cpu())
    U = int(uid.max()) + 1
    sizes = torch.bincount(uid, minlength=U)
    k = allocate_centroids(sizes, num_centroids, max_k_per_unit)
    return uid, sizes, k


def incremental_k(k_old: torch.Tensor, sizes_old: torch.Tensor, sizes_new: torch.Tensor,
                  new2old: torch.Tensor, changed: torch.Tensor, n_ctx_new: int,
                  percent_clusters: float) -> torch.Tensor:
    """
    Số centroid cho từng unit mới khi cập nhật incremental.

    Unit giữ nguyên: k cũ (bắt buộc — centroid cũ được chép sang nguyên khối).
    Unit changed có unit cũ tương ứng: co giãn k cũ theo tỉ lệ số token.
    Unit changed không có unit cũ: chia theo ngân sách danh nghĩa percent_clusters.
    Mọi k bị kẹp vào [1, số token của unit].

    Tổng thường lệch K danh nghĩa của bản mới vài centroid — người gọi phải ghi lại cả hai.
    """
    U = sizes_new.numel()
    k = torch.zeros(U, dtype=torch.long)
    mapped = new2old >= 0
    keep = mapped & ~changed
    k[keep] = k_old[new2old[keep]]
    re_m = changed & mapped
    if bool(re_m.any()):
        so = sizes_old[new2old[re_m]].double()
        k[re_m] = (k_old[new2old[re_m]].double() * sizes_new[re_m].double() / so).round().long()
    re_u = changed & ~mapped
    if bool(re_u.any()):
        k[re_u] = (percent_clusters / 100.0 * sizes_new[re_u].double()).floor().long()
    k = torch.minimum(k.clamp(min=1), sizes_new)
    return k


def incremental_hard_boundary(
    keys_new: torch.Tensor,
    uid_new: torch.Tensor,
    k_new: torch.Tensor,
    recluster: torch.Tensor,
    old_cent: torch.Tensor,
    old_lab: torch.Tensor,
    k_old: torch.Tensor,
    new2old: torch.Tensor,
    tok_new2old: torch.Tensor,
    n_iter: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Ghép centroid/label cho bản code MỚI: unit `recluster` chạy lại hard_boundary_kmeans,
    unit còn lại chép nguyên centroid + label từ bản cũ.

    keys_new    [H, S, D]      key của bản mới (fixed context, đã bỏ observation window)
    uid_new     [S]            unit_id compact của bản mới
    k_new       [U_new]        số centroid mỗi unit (xem `incremental_k`)
    recluster   [U_new] bool   unit phải cluster lại
    old_cent    [1, H, K_old, D], old_lab [1, H, S_old]   kết quả hard_boundary của bản cũ
    k_old       [U_old]        k_per_unit của bản cũ
    new2old     [U_new], tok_new2old [S]                  từ `map_units` / `token_map_new_to_old`

    Trả về (centroids [1, H, K, D], labels [1, H, S], stats) — cùng layout với
    hard_boundary_kmeans, K = k_new.sum().
    """
    if device is None:
        device = keys_new.device
    H, S, D = keys_new.shape
    uid_new = uid_new.to(device)
    k_new = k_new.to(device)
    recluster = recluster.to(device)
    new2old = new2old.to(device)
    tok_new2old = tok_new2old.to(device)
    k_old = k_old.to(device)
    old_cent = old_cent.squeeze(0).to(device)
    old_lab = old_lab.squeeze(0).to(device)

    K = int(k_new.sum())
    off_new = torch.cumsum(k_new, 0) - k_new
    off_old = torch.cumsum(k_old, 0) - k_old

    cent = torch.zeros(H, K, D, dtype=keys_new.dtype, device=device)
    lab = torch.full((H, S), -1, dtype=torch.long, device=device)

    # ---- unit giữ nguyên: chép khối centroid + dịch nhãn ----
    keep_units = (~recluster).nonzero(as_tuple=True)[0]
    if keep_units.numel():
        if bool((new2old[keep_units] < 0).any()):
            raise ValueError("unit giữ nguyên nhưng không có unit cũ tương ứng")
        ko = k_old[new2old[keep_units]]
        if not torch.equal(ko, k_new[keep_units]):
            raise ValueError("unit giữ nguyên phải có k_new == k_old")
        rep = torch.repeat_interleave(torch.arange(keep_units.numel(), device=device), ko)
        within = torch.arange(int(ko.sum()), device=device) - torch.repeat_interleave(
            torch.cumsum(ko, 0) - ko, ko)
        src = off_old[new2old[keep_units]][rep] + within
        dst = off_new[keep_units][rep] + within
        cent[:, dst, :] = old_cent[:, src, :].to(cent.dtype)

        tok_keep = (~recluster[uid_new]).nonzero(as_tuple=True)[0]
        src_tok = tok_new2old[tok_keep]
        if bool((src_tok < 0).any()):
            raise ValueError("token của unit giữ nguyên không có token cũ tương ứng")
        u = uid_new[tok_keep]
        shift = off_new[u] - off_old[new2old[u]]                      # [T]
        lab[:, tok_keep] = old_lab[:, src_tok] + shift.unsqueeze(0)

    # ---- unit phải cluster lại: chạy hard_boundary trên tập con, đúng k đã định ----
    n_re_tok = 0
    re_units = recluster.nonzero(as_tuple=True)[0]
    if re_units.numel():
        mask = recluster[uid_new]
        tok = mask.nonzero(as_tuple=True)[0]
        n_re_tok = tok.numel()
        sub_uid, _ = compact_unit_ids(uid_new[tok])               # giữ thứ tự tăng của unit
        k_sub = k_new[re_units]
        c_sub, l_sub, _ = hard_boundary_kmeans(
            keys_new[:, tok, :], sub_uid, int(k_sub.sum()), n_iter=n_iter,
            device=device, k_per_unit=k_sub)
        off_sub = torch.cumsum(k_sub, 0) - k_sub
        rep = torch.repeat_interleave(torch.arange(re_units.numel(), device=device), k_sub)
        within = torch.arange(int(k_sub.sum()), device=device) - off_sub[rep]
        sub2new = off_new[re_units][rep] + within                       # [K_sub]
        cent[:, sub2new, :] = c_sub.squeeze(0)
        lab[:, tok] = sub2new[l_sub.squeeze(0)]

    assert bool((lab >= 0).all()), "còn token chưa có nhãn"
    stats = {"K": K, "units_reclustered": int(re_units.numel()),
             "units_total": int(k_new.numel()), "tokens_reclustered": int(n_re_tok),
             "tokens_total": int(S)}
    return cent.unsqueeze(0), lab.unsqueeze(0), stats


def sa_assign_stale(keys_new: torch.Tensor, old_cent: torch.Tensor, old_lab: torch.Tensor,
                    tok_new2old: torch.Tensor) -> torch.Tensor:
    """
    Baseline incremental "ngây thơ" cho SA: KHÔNG chạy lại K-means. Token có token cũ tương
    ứng giữ nhãn cũ; token mới gán vào centroid cũ gần nhất theo cosine (đúng thước đo mà
    run_clustering dùng để gán). Centroid giữ nguyên. Trả về labels [1, H, S].
    """
    dev = keys_new.device
    old_cent = old_cent.squeeze(0).to(dev)
    old_lab = old_lab.squeeze(0).to(dev)
    tok_new2old = tok_new2old.to(dev)
    H, S, _ = keys_new.shape
    lab = torch.empty(H, S, dtype=torch.long, device=dev)
    has = tok_new2old >= 0
    lab[:, has] = old_lab[:, tok_new2old[has]]
    new_tok = (~has).nonzero(as_tuple=True)[0]
    if new_tok.numel():
        kn = torch.nn.functional.normalize(keys_new[:, new_tok, :].float(), dim=-1)
        cn = torch.nn.functional.normalize(old_cent.float(), dim=-1)
        lab[:, new_tok] = (kn @ cn.transpose(1, 2)).argmax(-1)
    return lab.unsqueeze(0)
