#!/usr/bin/env python
"""
check_cluster_integrity.py — quet CRC toan bo file centroid sau moi luot clustering.

VI SAO CAN FILE NAY
-------------------
Phase 0 mat ~5 gio GPU vi hai file centroid hong (mau 122 va 458): `pred.py` chay
nhieu gio roi moi chet khi doc trung file do. Toi da kiem ba lan voi ba muc chat dan:

  (1) file co ton tai khong          -> bo sot file bi cat cut
  (2) `zipfile.ZipFile()` mo duoc khong -> chi doc muc luc, bo sot du lieu hong ben trong
  (3) `zipfile.testzip()` kiem CRC tung entry -> moi bat duoc

Script nay lam thang muc (3). Quet ~1.500 file mat vai phut voi 8 tien trinh — re hon
rat nhieu so voi mot luot `pred.py` hong.

`torch.save` (torch >= 1.6) ghi ra file dinh dang zip, nen `testzip()` kiem duoc CRC cua
moi tensor ben trong ma khong phai nap tensor len RAM.

NO KIEM HAI THU
---------------
  A. TOAN VEN  — moi file .pt mo duoc va CRC dung.
  B. DAY DU    — voi moi dataidx trong [0, N), co du bo ba file ma modeling_*.py se doc:
                 centroids_tensor_dict / centroids_labels_dict / global_threshold.
                 Thieu mot file la `pred.py` chet giua chung.

USAGE
-----
    # kiem toan ven
    python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/

    # kiem them tinh day du cho 500 mau
    python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --expect 500

    # xoa file hong de offline_clustering.py sinh lai (no bo qua mau da co ket qua)
    python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --delete

Exit code: 0 = sach, 1 = co file hong hoac thieu.
"""
import argparse
import os
import re
import sys
import zipfile
from multiprocessing import Pool

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Bo ba file ma modeling_llama.py / modeling_qwen2.py doc cho moi sample o che do
# single-level. `global_threshold` la file duoc ghi SAU CUNG -> cung la file ma logic
# "bo qua mau da co ket qua" cua offline_clustering.py dung lam dau hieu hoan thanh.
SINGLE_LEVEL_PREFIXES = (
    "centroids_tensor_dict",
    "centroids_labels_dict",
    "global_threshold",
)

# vd: centroids_tensor_dict_113_209.pt -> (prefix, dataidx=113, K=209)
_NAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<dataidx>\d+)_(?P<k>\d+)\.pt$")


def check_one(path):
    """Tra ve (path, ok, ly_do). Chay trong tien trinh con -> khong dung state ngoai."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return (path, False, f"khong stat duoc: {e}")

    if size == 0:
        return (path, False, "file rong (0 byte)")

    if not zipfile.is_zipfile(path):
        # torch.save >= 1.6 luon ghi zip. Khong phai zip -> hoac cat cut, hoac dinh dang cu.
        return (path, False, "khong phai file zip (torch.save cat cut?)")

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()          # tra ve ten entry hong dau tien, None neu sach
            if bad is not None:
                return (path, False, f"CRC sai o entry '{bad}'")
            if not zf.namelist():
                return (path, False, "zip rong, khong co entry nao")
    except zipfile.BadZipFile as e:
        return (path, False, f"BadZipFile: {e}")
    except Exception as e:
        return (path, False, f"{type(e).__name__}: {e}")

    return (path, True, "")


def parse_name(filename):
    m = _NAME_RE.match(filename)
    if not m:
        return None
    return m.group("prefix"), int(m.group("dataidx")), int(m.group("k"))


def check_completeness(files, expect):
    """Voi moi dataidx trong [0, expect), kiem du bo ba file single-level."""
    seen = {}   # dataidx -> set(prefix)
    for fn in files:
        parsed = parse_name(os.path.basename(fn))
        if parsed is None:
            continue
        prefix, dataidx, _ = parsed
        if prefix in SINGLE_LEVEL_PREFIXES:
            seen.setdefault(dataidx, set()).add(prefix)

    missing = []
    for idx in range(expect):
        have = seen.get(idx, set())
        lack = [p for p in SINGLE_LEVEL_PREFIXES if p not in have]
        if lack:
            missing.append((idx, lack))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_dir", help="thu muc chua file .pt (vd fixed-prompt-clusters/lcc/)")
    ap.add_argument("--jobs", type=int, default=8, help="so tien trinh quet song song")
    ap.add_argument("--expect", type=int, default=-1,
                    help="so sample ky vong; kiem them tinh day du cho dataidx 0..N-1")
    ap.add_argument("--delete", action="store_true",
                    help="xoa file hong. offline_clustering.py se sinh lai o luot sau, "
                         "vi no bo qua mau da co global_threshold")
    args = ap.parse_args()

    if not os.path.isdir(args.cluster_dir):
        print(f"[ERROR] khong phai thu muc: {args.cluster_dir}")
        return 1

    files = sorted(
        os.path.join(args.cluster_dir, fn)
        for fn in os.listdir(args.cluster_dir)
        if fn.endswith(".pt")
    )
    if not files:
        print(f"[ERROR] khong tim thay file .pt nao trong {args.cluster_dir}")
        return 1

    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f">>> Thu muc: {args.cluster_dir}")
    print(f">>> {len(files)} file .pt, {total_bytes / 1e9:.1f} GB, quet bang {args.jobs} tien trinh")
    print()

    with Pool(processes=args.jobs) as pool:
        results = pool.map(check_one, files, chunksize=4)

    corrupt = [(p, why) for p, ok, why in results if not ok]

    for path, why in corrupt:
        print(f"  [HONG] {os.path.basename(path)} — {why}")

    rc = 0
    if corrupt:
        rc = 1
        print()
        print(f">>> {len(corrupt)}/{len(files)} file HONG")
        if args.delete:
            # Phai xoa CA BO BA cua mau hong, khong chi rieng file hong.
            # Ly do: offline_clustering.py bo qua mau da co `global_threshold_<idx>_<K>.pt`.
            # Neu file hong la centroids_labels/tensor ma ta chi xoa no, thi threshold van
            # con -> luot sau skip luon mau do -> file thieu KHONG BAO GIO duoc sinh lai,
            # va lan kiem sau lai bao hong y het. Gap that 21/8: 2 file 0 byte do container
            # bi dung lai giua luc ghi.
            PREFIXES = ("centroids_labels_dict", "centroids_tensor_dict", "global_threshold")
            tags = set()
            for path, _ in corrupt:
                base = os.path.basename(path)[:-3]          # bo duoi .pt
                for p in PREFIXES:
                    if base.startswith(p + "_"):
                        tags.add(base[len(p) + 1:])         # con lai "<dataidx>_<K>"
                        break
                else:
                    os.remove(path)                          # ten la -> xoa rieng
                    print(f"  [da xoa] {base}.pt")
            for tag in sorted(tags):
                for p in PREFIXES:
                    f = os.path.join(args.cluster_dir, f"{p}_{tag}.pt")
                    if os.path.exists(f):
                        os.remove(f)
                        print(f"  [da xoa] {p}_{tag}.pt")
            print()
            print("  Chay lai offline_clustering.py voi dung tham so cu de sinh lai.")
            print("  Da xoa ca bo ba nen luot sau se thuc su tinh lai nhung mau nay.")
        else:
            print("  Dung --delete de xoa roi chay lai offline_clustering.py sinh lai.")
    else:
        print(f">>> Toan ven: {len(files)}/{len(files)} file CRC dung")

    if args.expect > 0:
        print()
        missing = check_completeness(files, args.expect)
        if missing:
            rc = 1
            print(f">>> THIEU FILE cho {len(missing)}/{args.expect} sample:")
            for idx, lack in missing[:20]:
                print(f"  dataidx {idx}: thieu {', '.join(lack)}")
            if len(missing) > 20:
                print(f"  ... va {len(missing) - 20} sample nua")
            print()
            print("  pred.py se chet khi doc toi cac dataidx nay. Chay lai clustering truoc.")
        else:
            print(f">>> Day du: ca {args.expect} sample deu co du bo ba file")

    return rc


if __name__ == "__main__":
    sys.exit(main())
