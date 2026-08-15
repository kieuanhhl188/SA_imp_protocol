#!/usr/bin/env python
"""
test_struct_clustering.py — kiểm tra module Phase 2 trên CPU, không cần GPU.

Bất biến quan trọng nhất được kiểm: **không cluster nào chứa token của hai unit khác nhau**.
Đó chính là ranh giới cứng — thứ mà `ast_clustering.py` cũ KHÔNG có, và là toàn bộ nội dung
của đề xuất 1. Nếu bất biến này hỏng thì thí nghiệm Phase 5/6 đo nhầm thứ.

Usage:
    python scripts/test_struct_clustering.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from struct_clustering import (  # noqa: E402
    LEVELS, parse_units, assign_token_units, compact_unit_ids,
    allocate_centroids, hard_boundary_kmeans, struct_hierarchy_l1,
    compute_token_type_weights, classify_leaf, DEFAULT_TYPE_WEIGHTS,
)

OK = True


def check(name, cond, extra=""):
    global OK
    s = str(extra) if extra != "" else ""
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + s) if s else ''}")
    if not cond:
        OK = False


SAMPLE = '''import os
import sys


def alpha(a, b):
    total = a + b
    if total > 10:
        return total * 2
    return total


def beta(items):
    out = []
    for it in items:
        out.append(it)
    return out


class Gamma:
    def method_one(self):
        return 1

    def method_two(self, x):
        y = x + 1
        return y
'''


def test_parse():
    print("=== 1. parse_units theo từng level ===")
    prev = 0
    for lv in LEVELS:
        spans, st = parse_units(SAMPLE, "python", lv)
        print(f"    {lv:10} -> {len(spans):3} span, ERROR node = {st['num_error_nodes']}")
        check(f"level '{lv}' sinh ít nhất 1 span", len(spans) >= 1)
        if lv != "file":
            check(f"level '{lv}' mịn hơn hoặc bằng level trước", len(spans) >= prev,
                  f"{len(spans)} vs {prev}")
        prev = len(spans)

    spans_f, _ = parse_units(SAMPLE, "python", "function")
    # 2 hàm top-level + 2 method + 1 class + 1 span cả file
    check("level function bắt được 4 hàm + 1 class + 1 file", len(spans_f) == 6,
          str(len(spans_f)))

    print("\n=== 2. Code hỏng cú pháp (mô phỏng sample bị truncate) ===")
    broken = "def foo(a):\n    return a +" + "\n\ndef bar(:\n    pass\n"
    spans_b, st_b = parse_units(broken, "python", "function")
    check("vẫn parse được, không ném exception", len(spans_b) >= 1, f"{len(spans_b)} span")
    check("có báo số ERROR node để Phase 2 lọc", st_b["num_error_nodes"] > 0,
          f"n_error={st_b['num_error_nodes']}")


def test_assign():
    print("\n=== 3. assign_token_units: chọn span NHỎ NHẤT bao token ===")
    # span lồng nhau: file [0,100), function [10,50), block [20,30)
    spans = [(0, 100), (10, 50), (20, 30)]
    starts = torch.tensor([0, 5, 12, 22, 25, 35, 60, 90])
    u = assign_token_units(starts, spans)
    exp = torch.tensor([0, 0, 1, 2, 2, 1, 0, 0])
    check("gán đúng span nhỏ nhất", torch.equal(u, exp), f"{u.tolist()} vs {exp.tolist()}")

    # Thứ tự span trong list không được ảnh hưởng kết quả. So sánh PHÂN HOẠCH chứ không
    # so id, vì id phụ thuộc vị trí span trong list — đổi thứ tự thì id đổi là đương nhiên.
    spans_rev = [(20, 30), (0, 100), (10, 50)]
    u2 = assign_token_units(starts, spans_rev)

    def partition(t):
        return sorted(tuple(sorted((t == v).nonzero(as_tuple=True)[0].tolist()))
                      for v in torch.unique(t))

    check("bất biến với thứ tự span đầu vào (cùng phân hoạch)",
          partition(u) == partition(u2), f"{partition(u)} vs {partition(u2)}")

    check("mọi token đều có unit", bool((u >= 0).all()))

    print("\n=== 4. Gán trên code thật ===")
    spans_f, _ = parse_units(SAMPLE, "python", "function")
    # giả lập tokenizer: mỗi từ là một token
    starts, i = [], 0
    for line in SAMPLE.split("\n"):
        j = 0
        while j < len(line):
            if not line[j].isspace():
                starts.append(i + j)
                while j < len(line) and not line[j].isspace():
                    j += 1
            else:
                j += 1
        i += len(line) + 1
    ts = torch.tensor(starts)
    u = assign_token_units(ts, spans_f)
    uc, _ = compact_unit_ids(u)
    check("số unit có token > 1", int(uc.max()) + 1 > 1, f"U={int(uc.max())+1}")
    check("unit_id liên tục từ 0", sorted(set(uc.tolist())) == list(range(int(uc.max()) + 1)))


def test_allocate():
    print("\n=== 5. allocate_centroids ===")
    sizes = torch.tensor([100, 50, 10, 1, 300])
    k = allocate_centroids(sizes, 60, max_k_per_unit=64)
    check("tổng đúng bằng ngân sách", int(k.sum()) == 60, str(int(k.sum())))
    check("mỗi unit >= 1 centroid", bool((k >= 1).all()), k.tolist())
    check("không vượt số token của unit", bool((k <= sizes).all()), k.tolist())
    check("unit lớn nhất được nhiều centroid nhất", int(k.argmax()) == 4, k.tolist())

    k2 = allocate_centroids(sizes, 5, max_k_per_unit=64)
    check("ngân sách = số unit -> mỗi unit đúng 1", bool((k2 == 1).all()), k2.tolist())

    try:
        allocate_centroids(sizes, 4, max_k_per_unit=64)
        check("ngân sách < số unit phải raise", False)
    except ValueError:
        check("ngân sách < số unit phải raise", True)

    big = torch.full((3,), 1000)
    k3 = allocate_centroids(big, 20, max_k_per_unit=8)   # 20 <= 3*8, vừa đủ chỗ
    check("tôn trọng max_k_per_unit", bool((k3 <= 8).all()), k3.tolist())
    check("vẫn tiêu hết ngân sách", int(k3.sum()) == 20, int(k3.sum()))

    # Ngân sách vượt tổng sức chứa thì PHẢI raise, không được im lặng cắt bớt: cắt bớt sẽ
    # phá tính "cùng budget" — nền tảng của mọi so sánh trong protocol.
    try:
        allocate_centroids(big, 30, max_k_per_unit=8)    # 30 > 3*8 = 24
        check("ngân sách vượt sức chứa phải raise", False)
    except ValueError:
        check("ngân sách vượt sức chứa phải raise", True)


def test_hard_boundary():
    print("\n=== 6. hard_boundary_kmeans — BẤT BIẾN RANH GIỚI CỨNG ===")
    torch.manual_seed(0)
    H, D = 4, 8
    # 6 unit kích thước lệch nhau, để chạm nhiều bucket
    unit_sizes = [40, 3, 17, 1, 64, 25]
    unit_ids = torch.cat([torch.full((n,), i) for i, n in enumerate(unit_sizes)])
    S = unit_ids.numel()
    keys = torch.randn(H, S, D)

    K = 30
    cent, lab, st = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5)
    check("shape centroid [1,H,K,D]", tuple(cent.shape) == (1, H, K, D), str(tuple(cent.shape)))
    check("shape label [1,H,S]", tuple(lab.shape) == (1, H, S), str(tuple(lab.shape)))
    check("label nằm trong [0,K)", bool((lab >= 0).all() and (lab < K).all()))

    # BẤT BIẾN LÕI: mỗi cluster chỉ chứa token của đúng một unit
    l = lab.squeeze(0)
    viol = 0
    for h in range(H):
        for c in torch.unique(l[h]):
            us = torch.unique(unit_ids[l[h] == c])
            if us.numel() != 1:
                viol += 1
    check("KHÔNG cluster nào vắt qua hai unit", viol == 0, f"{viol} vi phạm")

    # mỗi unit dùng đúng dải cluster của nó, không chồng lấn giữa các unit
    ranges = {}
    for u in range(len(unit_sizes)):
        cs = torch.unique(l[0][unit_ids == u])
        ranges[u] = (int(cs.min()), int(cs.max()))
    overlap = any(
        ranges[a][1] >= ranges[b][0] and ranges[b][1] >= ranges[a][0]
        for a in ranges for b in ranges if a < b
    )
    check("dải cluster của các unit không chồng lấn", not overlap, str(ranges))

    print(f"    thống kê: {st['num_units']} unit, {st['num_buckets']} bucket, "
          f"{st['num_kernel_calls']} lần gọi kernel")
    check("số lần gọi kernel là O(bucket x iter), không phải O(U x H)",
          st["num_kernel_calls"] <= st["num_buckets"] * 5 * 2,
          f"{st['num_kernel_calls']} với U={st['num_units']}, H={H}")

    print("\n=== 7. Tính tất định ===")
    c2, l2, _ = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5)
    check("chạy lại ra kết quả y hệt",
          torch.equal(cent, c2) and torch.equal(lab, l2))

    print("\n=== 8. Ca biên ===")
    # unit 1 token
    u1 = torch.tensor([0, 1, 1, 1])
    k1 = torch.randn(2, 4, D)
    c3, l3, _ = hard_boundary_kmeans(k1, u1, 2, n_iter=3)
    check("unit 1 token -> centroid chính là token đó",
          torch.allclose(c3[0, :, 0, :], k1[:, 0, :], atol=1e-5))
    check("không NaN", not bool(torch.isnan(c3).any()))

    # mọi token cùng một unit -> quy về K-means thường
    u_all = torch.zeros(S, dtype=torch.long)
    c4, l4, _ = hard_boundary_kmeans(keys, u_all, 8, n_iter=5)
    check("một unit duy nhất vẫn chạy", tuple(c4.shape) == (1, H, 8, D))
    check("không NaN", not bool(torch.isnan(c4).any()))

    # K == S: mỗi token một cluster
    u_s = torch.tensor([0, 0, 1, 1])
    k_s = torch.randn(2, 4, D)
    c5, l5, _ = hard_boundary_kmeans(k_s, u_s, 4, n_iter=3)
    check("K == S -> mỗi token một cluster",
          len(torch.unique(l5[0, 0])) == 4, str(torch.unique(l5[0, 0]).tolist()))


def test_hierarchy():
    print("\n=== 9. struct_hierarchy_l1 ===")
    torch.manual_seed(1)
    H, D = 3, 6
    # 4 unit L2 (function), gộp thành 2 unit L1 (class)
    l2_sizes = [10, 6, 8, 12]
    unit_l2 = torch.cat([torch.full((n,), i) for i, n in enumerate(l2_sizes)])
    unit_l1 = torch.cat([torch.full((n,), 0 if i < 2 else 1) for i, n in enumerate(l2_sizes)])
    S = unit_l2.numel()
    keys = torch.randn(H, S, D)

    c2, l2, _ = hard_boundary_kmeans(keys, unit_l2, 12, n_iter=5)
    c1, l1 = struct_hierarchy_l1(c2, l2, unit_l1, weighted=False, unit_ids_l2=unit_l2)

    check("K1 == số unit L1", c1.shape[2] == 2, str(tuple(c1.shape)))
    check("labels_l1 ánh xạ thẳng key -> cluster L1", tuple(l1.shape) == (1, H, S))
    check("labels_l1 khớp unit_l1", torch.equal(l1[0, 0], unit_l1))
    check("không NaN", not bool(torch.isnan(c1).any()))

    # weighted=False: L1 centroid == trung bình cộng các L2 centroid thuộc nó
    cl2_l1 = torch.zeros(c2.shape[2], dtype=torch.long)
    cl2_l1.scatter_(0, l2[0, 0], unit_l1)
    for g in range(2):
        want = c2[0, :, cl2_l1 == g, :].mean(dim=1)
        check(f"L1 centroid nhóm {g} == trung bình cộng L2 của nhóm",
              torch.allclose(c1[0, :, g, :], want, atol=1e-5))

    # weighted=True: L1 centroid == trung bình của TOÀN BỘ KEY trong nhóm.
    # Đây là điểm mấu chốt — chỉ khi có trọng số thì L1 mới thật sự đại diện cho nhóm key.
    c1w, _ = struct_hierarchy_l1(c2, l2, unit_l1, weighted=True)
    check("weighted khác unweighted", not torch.allclose(c1, c1w, atol=1e-4))
    for g in range(2):
        want = keys[:, unit_l1 == g, :].mean(dim=1)
        check(f"weighted L1 nhóm {g} == trung bình toàn bộ key của nhóm",
              torch.allclose(c1w[0, :, g, :], want, atol=1e-4),
              f"lệch max {float((c1w[0, :, g, :] - want).abs().max()):.2e}")

    check("phát hiện hierarchy không lồng nhau", _raises(
        lambda: struct_hierarchy_l1(c2, l2, unit_l1,
                                    unit_ids_l2=torch.zeros(S, dtype=torch.long))))


def test_l1_groups():
    from struct_clustering import build_l1_groups
    print("\n=== 10b. build_l1_groups — ép K1 về đúng mục tiêu ===")
    # 6 unit L2, gộp thành 3 unit cha
    l2_sizes = [10, 6, 8, 12, 5, 9]
    unit_l2 = torch.cat([torch.full((n,), i) for i, n in enumerate(l2_sizes)])
    parent = [0, 0, 1, 1, 2, 2]
    unit_l1 = torch.cat([torch.full((n,), parent[i]) for i, n in enumerate(l2_sizes)])

    g, st = build_l1_groups(unit_l2, unit_l1, target_k1=3)
    check("K1 == số unit cha -> giữ nguyên", st["k1_mode"] == "as-is" and st["k1_actual"] == 3,
          st)

    g, st = build_l1_groups(unit_l2, unit_l1, target_k1=2)
    check("K1 < số unit cha -> gộp, đạt đúng mục tiêu",
          st["k1_mode"] == "merge" and st["k1_actual"] == 2, st)
    check("gộp vẫn giữ unit L2 nguyên vẹn",
          all(torch.unique(g[unit_l2 == u]).numel() == 1 for u in torch.unique(unit_l2)))

    g, st = build_l1_groups(unit_l2, unit_l1, target_k1=6)
    check("K1 > số unit cha -> tách, đạt đúng mục tiêu",
          st["k1_mode"] == "split" and st["k1_actual"] == 6, st)
    check("tách vẫn giữ unit L2 nguyên vẹn",
          all(torch.unique(g[unit_l2 == u]).numel() == 1 for u in torch.unique(unit_l2)))
    check("mỗi nhóm sau tách vẫn nằm trong đúng một unit cha",
          all(torch.unique(unit_l1[g == v]).numel() == 1 for v in torch.unique(g)))

    g, st = build_l1_groups(unit_l2, unit_l1, target_k1=5)
    check("mục tiêu không chia hết vẫn đạt đúng", st["k1_actual"] == 5, st)

    print("\n=== 10. Hierarchy phải TÔN TRỌNG lồng nhau ===")
    # mỗi unit L2 phải nằm gọn trong đúng một unit L1
    bad = 0
    for u in torch.unique(unit_l2):
        if torch.unique(unit_l1[unit_l2 == u]).numel() != 1:
            bad += 1
    check("mỗi unit L2 nằm trong đúng một unit L1", bad == 0, f"{bad} vi phạm")


def test_token_weights():
    print("\n=== 11. Hướng 2(b) token-type weighting — CỜ TẮT SẴN ===")
    torch.manual_seed(2)
    H, D = 3, 8
    unit_sizes = [20, 12, 30]
    unit_ids = torch.cat([torch.full((n,), i) for i, n in enumerate(unit_sizes)])
    S = unit_ids.numel()
    keys = torch.randn(H, S, D)
    K = 12

    base_c, base_l, base_st = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5)
    check("mặc định là KHÔNG trọng số", base_st["token_weighted"] is False)

    # BẤT BIẾN QUAN TRỌNG NHẤT: tắt cờ phải ra kết quả Y HỆT bản chưa có tính năng này.
    # Nếu không, con số baseline +HardBoundary bị nhiễm và ablation vô nghĩa.
    none_c, none_l, _ = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5, token_weights=None)
    check("token_weights=None trùng khớp bit-for-bit với mặc định",
          torch.equal(base_c, none_c) and torch.equal(base_l, none_l))

    # trọng số đều = 1 cũng phải ra y hệt (kiểm mẫu số dùng tổng trọng số, không phải count)
    ones_c, ones_l, ones_st = hard_boundary_kmeans(
        keys, unit_ids, K, n_iter=5, token_weights=torch.ones(S))
    check("trọng số toàn 1 cho cùng kết quả",
          torch.allclose(base_c, ones_c, atol=1e-6) and torch.equal(base_l, ones_l))
    check("stats đánh dấu đã bật", ones_st["token_weighted"] is True)

    # trọng số < 1: đây là ca mà clamp(min=1) cũ sẽ làm sai
    small = torch.full((S,), 0.5)
    small_c, _, _ = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5, token_weights=small)
    check("trọng số hằng 0.5 vẫn ra cùng centroid (trung bình bất biến khi nhân hằng số)",
          torch.allclose(base_c, small_c, atol=1e-5),
          f"lệch max {float((base_c - small_c).abs().max()):.2e}")

    # trọng số không đều thì phải đổi kết quả
    wv = torch.ones(S)
    wv[:S // 2] = 3.0
    var_c, _, _ = hard_boundary_kmeans(keys, unit_ids, K, n_iter=5, token_weights=wv)
    check("trọng số không đều làm centroid đổi", not torch.allclose(base_c, var_c, atol=1e-4))

    check("trọng số âm bị chặn", _raises(lambda: hard_boundary_kmeans(
        keys, unit_ids, K, n_iter=2, token_weights=-torch.ones(S))))

    print("\n=== 12. compute_token_type_weights ===")
    check("phân loại identifier", classify_leaf("identifier") == "identifier")
    check("phân loại literal", classify_leaf("integer") == "literal")
    check("phân loại dấu câu", classify_leaf(":") == "punctuation")
    check("keyword KHÔNG bị coi là identifier (bản regex cũ nhầm chỗ này)",
          classify_leaf("return") == "keyword")

    starts, i = [], 0
    for line in SAMPLE.split("\n"):
        j = 0
        while j < len(line):
            if not line[j].isspace():
                starts.append(i + j)
                while j < len(line) and not line[j].isspace():
                    j += 1
            else:
                j += 1
        i += len(line) + 1
    ts = torch.tensor(starts)
    w = compute_token_type_weights(SAMPLE, ts, "python")
    check("trả về đúng độ dài", w.numel() == ts.numel(), f"{w.numel()} vs {ts.numel()}")
    check("có token được boost lên 1.5 (identifier)",
          bool((w == DEFAULT_TYPE_WEIGHTS["identifier"]).any()))
    check("mọi trọng số dương", bool((w > 0).all()))
    # So sánh có dung sai: float32 lưu 1.2 thành 1.2000000476837158, so bằng set sẽ trượt.
    allowed = sorted(set(DEFAULT_TYPE_WEIGHTS.values()))
    got = sorted(w.unique().tolist())
    check("nằm trong dải giá trị đã khai báo",
          all(any(abs(g - a) < 1e-5 for a in allowed) for g in got),
          f"{got} vs {allowed}")


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def main():
    test_parse()
    test_assign()
    test_allocate()
    test_hard_boundary()
    test_hierarchy()
    test_l1_groups()
    test_token_weights()
    print("\n" + ("TẤT CẢ PASS" if OK else "CÓ TEST FAIL"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
