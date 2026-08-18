# -*- coding: utf-8 -*-
"""
test_eval_detail.py — kiem tra ban va `LongBench/eval.py` + `LongBench/pred.py`.

Chay tren CPU, khong can GPU, khong can dataset that:

    python scripts/test_eval_detail.py

Kiem 5 dieu:
  1. jsonl CU (khong co `dataidx`) van doc duoc -> khong pha du lieu da co
  2. jsonl MOI (co `dataidx`)      -> `per_sample` key theo dataidx, ghep duoc giua config
  3. diem trung binh trong `result.json` KHONG doi (bit-for-bit voi cong thuc goc)
  4. `dataidx` trung lap bi phat hien -> bug append khong con am tham
  5. jsonl rong -> bao loi ro rang, khong NameError/ZeroDivisionError

Diem 3 la diem quan trong nhat: moi so da cong bo (vd LCC All-KV 54.83 / Sq-70% 56.08)
phai tai lap y nguyen sau ban va, neu khong thi ban va da lam sai lech ket qua.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LB = os.path.join(REPO_ROOT, "LongBench")

sys.path.insert(0, LB)
os.chdir(LB)

# jieba/rouge chi phuc vu task tieng Trung + summarization, khong lien quan
# code_sim_score. Stub de test chay duoc du chua cai hai goi do.
import types  # noqa: E402

for _m in ("jieba", "rouge"):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except ImportError:
            _mod = types.ModuleType(_m)
            if _m == "rouge":
                _mod.Rouge = object
            sys.modules[_m] = _mod

from metrics import code_sim_score  # noqa: E402

# (prediction, [ground truth]) — phu cac ca cua code_sim_score
SAMPLES = [
    ("    return self.value", ["    return self.value"]),   # khop hoan toan
    ("    x = 1", ["    y = 2"]),                            # khac han
    ("", ["    foo()"]),                                     # prediction rong
    ("def f(a, b):", ["def f(a, b, c):"]),                   # gan giong
    ("# comment\n    real = 1", ["    real = 1"]),           # dong dau la comment -> bi bo
]

fails = []
created = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def old_formula(preds, answs):
    """Cong thuc `scorer` NGUYEN BAN, de doi chieu bit-for-bit."""
    total = 0.0
    for p, gts in zip(preds, answs):
        s = 0.0
        for gt in gts:
            s = max(s, code_sim_score(p, gt))
        total += s
    return round(100 * total / len(preds), 2)


def write_jsonl(cfg, with_dataidx, dup=False, empty=False):
    d = os.path.join("pred", cfg + "_baseline")
    os.makedirs(d, exist_ok=True)
    created.append(d)
    with open(os.path.join(d, "lcc.jsonl"), "w", encoding="utf-8") as f:
        if empty:
            return
        for i, (p, a) in enumerate(SAMPLES):
            rec = {"pred": p, "answers": a, "all_classes": None, "length": 1000 + i}
            if with_dataidx:
                # dup: mau 1 lay lai dataidx 0 -> mo phong jsonl bi append hai luot
                rec = dict(dataidx=(0 if dup and i == 1 else i), **rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_eval(cfg):
    return subprocess.run([sys.executable, "eval.py", "--model", cfg],
                          capture_output=True, text=True)


def load(cfg, name):
    with open(os.path.join("pred", cfg + "_baseline", name), encoding="utf-8") as f:
        return json.load(f)


expected = old_formula([s[0] for s in SAMPLES], [s[1] for s in SAMPLES])
print("diem theo cong thuc NGUYEN BAN = %s\n" % expected)

# ---------- 1 + 3. jsonl cu ----------
print("[1] jsonl CU (khong co dataidx) — tuong thich nguoc")
write_jsonl("_ttOLD", with_dataidx=False)
r = run_eval("_ttOLD")
check(r.returncode == 0, "chay khong loi (rc=%s)" % r.returncode)
if r.returncode == 0:
    res, det = load("_ttOLD", "result.json"), load("_ttOLD", "result_detail.json")
    check(res == {"lcc": expected},
          "result.json = {'lcc': %s} — BIT-FOR-BIT voi cong thuc goc" % expected)
    check(det["lcc"]["n_samples"] == 5, "n_samples = 5")
    check(det["lcc"]["dataidx_available"] is False, "dataidx_available = False")
    check(isinstance(det["lcc"]["per_sample"], list), "per_sample la LIST khi thieu dataidx")
    check(det["lcc"]["n_empty_pred"] == 1, "dem duoc 1 prediction rong")
    check("n=5" in r.stdout, "in n=5 ra stdout (ban cu khong in gi)")
else:
    print(r.stdout[-800:], r.stderr[-800:])

# ---------- 2. jsonl moi ----------
print("\n[2] jsonl MOI (co dataidx)")
write_jsonl("_ttNEW", with_dataidx=True)
r = run_eval("_ttNEW")
check(r.returncode == 0, "chay khong loi")
if r.returncode == 0:
    res, det = load("_ttNEW", "result.json"), load("_ttNEW", "result_detail.json")
    check(res == {"lcc": expected}, "diem KHONG doi khi them dataidx (%s)" % expected)
    ps = det["lcc"]["per_sample"]
    check(isinstance(ps, dict), "per_sample la DICT — ghep duoc theo mau giua cac config")
    check(sorted(ps.keys(), key=int) == ["0", "1", "2", "3", "4"], "key = dataidx 0..4")
    check(det["lcc"]["n_duplicate_dataidx"] == 0, "khong bao trung lap gia")

# ---------- 4. dataidx trung lap ----------
print("\n[4] dataidx trung lap (mo phong bug append cua pred.py)")
write_jsonl("_ttDUP", with_dataidx=True, dup=True)
r = run_eval("_ttDUP")
if r.returncode == 0:
    det = load("_ttDUP", "result_detail.json")
    check(det["lcc"]["n_duplicate_dataidx"] == 1, "phat hien 1 dataidx trung lap")
    check("CANH BAO" in r.stdout, "in CANH BAO ra stdout")
else:
    check(False, "chay khong loi (rc=%s)" % r.returncode)

# ---------- 5. jsonl rong ----------
print("\n[5] jsonl rong")
write_jsonl("_ttEMPTY", with_dataidx=True, empty=True)
r = run_eval("_ttEMPTY")
out = r.stdout + r.stderr
check(r.returncode != 0, "thoat voi ma loi (rc=%s), khong PASS am tham" % r.returncode)
check("0 prediction" in out, "bao ro '0 prediction'")
check("NameError" not in out, "khong con NameError: all_classes (bug co san cua ban goc)")
check("ZeroDivisionError" not in out, "khong ZeroDivisionError")

for d in created:
    shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 62)
if fails:
    print("FAIL: %d" % len(fails))
    for m in fails:
        print("  - " + m)
else:
    print("TAT CA PASS")
sys.exit(1 if fails else 0)
