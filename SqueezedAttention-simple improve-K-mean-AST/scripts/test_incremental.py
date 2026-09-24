#!/usr/bin/env python
"""
test_incremental.py — kiểm tra module Phase 4 (incremental_clustering.py) trên CPU, không cần GPU.

Ba điều phải đúng trước khi đo thời gian trên pod, vì cả ba đều KHÔNG crash nếu sai:
  1. Ghép đúng: unit giữ nguyên nhận lại y hệt centroid + nhãn cũ; cluster lại MỌI unit với
     cùng k_u thì trùng bit với hard_boundary_kmeans chạy trên toàn bộ.
  2. Không phá ranh giới cứng: không cluster nào chứa token của hai unit.
  3. Tham số mới `k_per_unit` của hard_boundary_kmeans không đổi đường mặc định.

Usage:
    python scripts/test_incremental.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from struct_clustering import (  # noqa: E402
    parse_units, assign_token_units, compact_unit_ids, hard_boundary_kmeans,
)
from incremental_clustering import (  # noqa: E402
    make_edit, diff_region, token_map_new_to_old, map_units, unit_layout,
    incremental_k, incremental_hard_boundary, sa_assign_stale,
)

OK = True


def check(name, cond, extra=""):
    global OK
    s = str(extra) if extra != "" else ""
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + s) if s else ''}")
    if not cond:
        OK = False


def no_cross_unit(lab, uid):
    """Mọi cluster chỉ chứa token của một unit, ở mọi head."""
    H = lab.shape[1]
    for h in range(H):
        l = lab[0, h]
        for c in torch.unique(l):
            if torch.unique(uid[l == c]).numel() != 1:
                return False
    return True


def make_units(sizes):
    return torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])


def test_k_per_unit_default_unchanged():
    print("\n=== hard_boundary_kmeans: k_per_unit mặc định không đổi hành vi ===")
    torch.manual_seed(0)
    uid = make_units([40, 7, 130, 3, 60])
    keys = torch.randn(4, uid.numel(), 16)
    K = 20
    _, _, k = unit_layout(uid, K)
    c1, l1, _ = hard_boundary_kmeans(keys, uid, K)
    c2, l2, _ = hard_boundary_kmeans(keys, uid, K, k_per_unit=k)
    check("centroid trùng bit", torch.equal(c1, c2))
    check("nhãn trùng bit", torch.equal(l1, l2))
    bad = k.clone()
    bad[0] += 1
    try:
        hard_boundary_kmeans(keys, uid, K, k_per_unit=bad)
        check("k_per_unit sai tổng thì assert", False)
    except AssertionError:
        check("k_per_unit sai tổng thì assert", True)


def test_make_edit():
    print("\n=== make_edit: chèn vào thân class, cây AST không thêm unit ===")
    py = "import os\n\nclass A(object):\n    def f(self):\n        return 1\n\nx = 2\n"
    spans, _ = parse_units(py, "python", "class")
    cls = [s for s in spans if s != (0, len(py))]
    pos, ins = make_edit(py, cls[0], "python", n_lines=2)
    new = py[:pos] + ins + py[pos:]
    check("python: chỗ chèn nằm trong span class", cls[0][0] < pos < cls[0][1])
    check("python: giữ thụt lề", ins.startswith("    # "), repr(ins))
    spans2, st2 = parse_units(new, "python", "class")
    check("python: số unit không đổi, không node ERROR",
          len(spans2) == len(spans) and st2["num_error_nodes"] == 0)

    java = ("package p;\nimport q.R;\n@Anno\npublic class B extends C {\n"
            "    private int x;\n    public int g() { return x; }\n}\n")
    spans, _ = parse_units(java, "java", "class")
    cls = [s for s in spans if s != (0, len(java))]
    pos, ins = make_edit(java, cls[0], "java", n_lines=1)
    new = java[:pos] + ins + java[pos:]
    check("java: chèn SAU dấu '{' đầu tiên", java.find("{", cls[0][0]) < pos)
    spans2, st2 = parse_units(new, "java", "class")
    check("java: số unit không đổi, không node ERROR",
          len(spans2) == len(spans) and st2["num_error_nodes"] == 0)
    check("unit một dòng thì trả None", make_edit("class A: pass", (0, 13), "python") is None)


def test_diff_region():
    print("\n=== diff_region / token_map_new_to_old ===")
    old = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    new = torch.tensor([1, 2, 3, 9, 9, 9, 4, 5, 6, 7])
    e, oe, ne = diff_region(old, new)
    check("chèn thuần: (e, old_end, new_end)", (e, oe, ne) == (3, 3, 6), (e, oe, ne))
    m = token_map_new_to_old(len(new), len(old), e, oe, ne)
    check("ánh xạ token", m.tolist() == [0, 1, 2, -1, -1, -1, 3, 4, 5, 6], m.tolist())
    old = torch.tensor([1, 2, 8, 4, 5])
    new = torch.tensor([1, 2, 9, 9, 4, 5])
    check("thay thế (token biên bị gộp lại)", diff_region(old, new) == (2, 3, 4))
    check("giống hệt", diff_region(old, old) == (5, 5, 5))


def _scenario(seed=0):
    """Bản cũ 5 unit; bản mới chèn 6 token vào giữa unit 2. Key ngoài vùng sửa giữ nguyên."""
    torch.manual_seed(seed)
    H, D = 3, 16
    sizes_old = [50, 20, 90, 15, 70]
    uid_old = make_units(sizes_old)
    keys_old = torch.randn(H, uid_old.numel(), D)
    ins_at, n_ins = 50 + 20 + 40, 6
    keys_new = torch.cat([keys_old[:, :ins_at], torch.randn(H, n_ins, D),
                          keys_old[:, ins_at:]], dim=1)
    uid_new = torch.cat([uid_old[:ins_at], torch.full((n_ins,), 2), uid_old[ins_at:]])
    ids_old = torch.arange(uid_old.numel()) + 1000
    ids_new = torch.cat([ids_old[:ins_at], torch.arange(n_ins), ids_old[ins_at:]])
    return H, keys_old, keys_new, uid_old, uid_new, ids_old, ids_new


def test_incremental_local():
    print("\n=== P1 local: chỉ cluster lại unit bị sửa ===")
    H, keys_old, keys_new, uid_old, uid_new, ids_old, ids_new = _scenario()
    K_old = 12
    uid_old, sizes_old, k_old = unit_layout(uid_old, K_old)
    c_old, l_old, _ = hard_boundary_kmeans(keys_old, uid_old, K_old, k_per_unit=k_old)

    e, oe, ne = diff_region(ids_old, ids_new)
    t = token_map_new_to_old(ids_new.numel(), ids_old.numel(), e, oe, ne)
    uid_new, _ = compact_unit_ids(uid_new)
    new2old, changed = map_units(uid_old, uid_new, t)
    check("chỉ unit 2 changed", changed.tolist() == [False, False, True, False, False],
          changed.tolist())
    sizes_new = torch.bincount(uid_new)
    k_new = incremental_k(k_old, sizes_old, sizes_new, new2old, changed, uid_new.numel(), 5)
    check("unit giữ nguyên giữ k cũ", torch.equal(k_new[~changed], k_old[~changed]))

    c, l, st = incremental_hard_boundary(keys_new, uid_new, k_new, changed, c_old, l_old,
                                         k_old, new2old, t, n_iter=10)
    check("shape [1,H,K,D] / [1,H,S]",
          c.shape == (1, H, int(k_new.sum()), 16) and l.shape == (1, H, uid_new.numel()))
    check("không cluster nào vắt qua unit", no_cross_unit(l, uid_new))
    check("chỉ cluster lại 1 unit", st["units_reclustered"] == 1, st)

    # unit 0,1 (trước chỗ sửa) và 3,4 (sau) phải y hệt bản cũ
    off_o = torch.cumsum(k_old, 0) - k_old
    off_n = torch.cumsum(k_new, 0) - k_new
    same = True
    for u in [0, 1, 3, 4]:
        a = c_old[0, :, off_o[u]:off_o[u] + k_old[u]]
        b = c[0, :, off_n[u]:off_n[u] + k_new[u]]
        same &= torch.equal(a, b)
    check("centroid unit giữ nguyên trùng bit với bản cũ", same)
    j_new = (uid_new != 2).nonzero(as_tuple=True)[0]
    rel_old = l_old[0][:, t[j_new]] - off_o[uid_old[t[j_new]]]
    rel_new = l[0][:, j_new] - off_n[uid_new[j_new]]
    check("nhãn tương đối trong unit giữ nguyên trùng", torch.equal(rel_old, rel_new))

    # unit 2 phải đúng bằng hard_boundary chạy riêng unit đó
    tok2 = (uid_new == 2).nonzero(as_tuple=True)[0]
    c2, l2, _ = hard_boundary_kmeans(keys_new[:, tok2], torch.zeros(tok2.numel(), dtype=torch.long),
                                     int(k_new[2]), k_per_unit=k_new[2:3])
    check("centroid unit sửa == hard_boundary riêng unit đó",
          torch.equal(c[0, :, off_n[2]:off_n[2] + k_new[2]], c2[0]))


def test_incremental_all_equals_full():
    print("\n=== P2/giới hạn: cluster lại MỌI unit == hard_boundary toàn bộ, cùng k ===")
    H, keys_old, keys_new, uid_old, uid_new, ids_old, ids_new = _scenario(seed=1)
    K_old = 12
    uid_old, sizes_old, k_old = unit_layout(uid_old, K_old)
    c_old, l_old, _ = hard_boundary_kmeans(keys_old, uid_old, K_old, k_per_unit=k_old)
    e, oe, ne = diff_region(ids_old, ids_new)
    t = token_map_new_to_old(ids_new.numel(), ids_old.numel(), e, oe, ne)
    uid_new, _ = compact_unit_ids(uid_new)
    new2old, changed = map_units(uid_old, uid_new, t)
    k_new = incremental_k(k_old, sizes_old, torch.bincount(uid_new), new2old, changed,
                          uid_new.numel(), 5)
    everything = torch.ones_like(changed)
    c, l, _ = incremental_hard_boundary(keys_new, uid_new, k_new, everything, c_old, l_old,
                                        k_old, new2old, t)
    cf, lf, _ = hard_boundary_kmeans(keys_new, uid_new, int(k_new.sum()), k_per_unit=k_new)
    check("centroid trùng bit", torch.equal(c, cf))
    check("nhãn trùng bit", torch.equal(l, lf))

    # suffix: mọi unit có token >= e
    suffix = changed.clone()
    suffix[torch.unique(uid_new[e:])] = True
    c, l, st = incremental_hard_boundary(keys_new, uid_new, k_new, suffix, c_old, l_old,
                                         k_old, new2old, t)
    check("suffix: key trước chỗ sửa không đổi -> trùng bit với full cùng k",
          torch.equal(c, cf) and torch.equal(l, lf), st)


def test_no_change_roundtrip():
    print("\n=== không sửa gì: incremental trả lại đúng bản cũ ===")
    torch.manual_seed(2)
    uid = make_units([30, 30, 30])
    keys = torch.randn(2, 90, 8)
    uid, sizes, k = unit_layout(uid, 9)
    c0, l0, _ = hard_boundary_kmeans(keys, uid, 9, k_per_unit=k)
    t = torch.arange(90)
    new2old, changed = map_units(uid, uid, t)
    check("không unit nào changed", not bool(changed.any()))
    c, l, _ = incremental_hard_boundary(keys, uid, k, changed, c0, l0, k, new2old, t)
    check("trùng bit", torch.equal(c, c0) and torch.equal(l, l0))


def test_sa_stale():
    print("\n=== sa_assign_stale ===")
    torch.manual_seed(3)
    cent = torch.randn(1, 2, 4, 8)
    lab_old = torch.randint(0, 4, (1, 2, 10))
    keys_new = torch.randn(2, 12, 8)
    keys_new[:, 5] = cent[0, :, 3] * 7.0         # token mới trùng hướng centroid 3
    t = torch.tensor([0, 1, 2, 3, 4, -1, -1, 5, 6, 7, 8, 9])
    lab = sa_assign_stale(keys_new, cent, lab_old, t)
    check("token cũ giữ nhãn", torch.equal(lab[0][:, t >= 0], lab_old[0][:, t[t >= 0]]))
    check("token mới gán theo cosine", bool((lab[0, :, 5] == 3).all()), lab[0, :, 5].tolist())


def main():
    test_k_per_unit_default_unchanged()
    test_make_edit()
    test_diff_region()
    test_incremental_local()
    test_incremental_all_equals_full()
    test_no_change_roundtrip()
    test_sa_stale()
    print("\n" + ("TẤT CẢ PASS" if OK else "CÓ TEST FAIL"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
