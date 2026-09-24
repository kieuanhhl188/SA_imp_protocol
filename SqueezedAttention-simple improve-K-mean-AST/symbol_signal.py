"""
symbol_signal.py — Phase 3 (V1): chi so ky hieu (symbol / def-use) doc lap voi SA.

CHI DOC. Khong sua clustering, khong tao centroid moi, khong dung ranh gioi AST, khong dung LLM.
V1 = khop CHUOI dinh danh (identifier string matching). KHONG phan giai scope / def-use that (V2).

INDEX (moi mau)
    identifier -> [vi tri token trong fixed context (< n_ctx)]
  - Dinh danh = regex `[A-Za-z_][A-Za-z0-9_]*` tren vung code cua prompt (code_char_start:code_char_end
    tu Phase 1.4), KHONG dinh danh la tu khoa cua ngon ngu (Python / Java, lay tu meta Phase 1).
  - Vi tri token lay tu offset Phase 1.4: token j thuoc dinh danh [s, e) neu khoang ky tu cua token
    giao [s, e). Khong tokenize lai, khong dung tokenizer thu hai.

QUERY
  Trong C2, "query" = cua so quan sat: 100 token cuoi cua shared prefix (q[:, n_ctx:sp_len]).
  Dinh danh cua query = cac dinh danh (khac nhau) trong vung ky tu cua cua so do. Mot dinh danh
  cua query chi co tin hieu neu no xuat hien o fixed context (vi tri < n_ctx).

symbol_hit(cluster) (cho moi head, vi labels SA khac nhau theo head)
    = #{ dinh danh khop m : ton tai key j thuoc vi tri cua m co label[h, j] == cluster } / #dinh danh khop
  in [0, 1]. = 0 voi moi cluster neu khong co dinh danh nao khop. Chuan hoa theo so dinh danh khop
  de lambda co nghia on dinh giua cac mau. Cung mot vector cho ca 100 query cua cua so.
"""
import keyword
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")

JAVA_KEYWORDS = frozenset("""
abstract assert boolean break byte case catch char class const continue default do double else enum
extends final finally float for goto if implements import instanceof int interface long native new
package private protected public return short static strictfp super switch synchronized this throw
throws transient try void volatile while true false null var
""".split())
PY_KEYWORDS = frozenset(keyword.kwlist) | frozenset({"True", "False", "None"})
KEYWORDS = {"python": PY_KEYWORDS, "java": JAVA_KEYWORDS}


def extract_identifiers(text: str, language: str, base: int = 0) -> List[Tuple[str, int, int]]:
    """[(name, start_char, end_char)] theo thu tu xuat hien, toa do = base + vi tri trong text."""
    kw = KEYWORDS[language]
    out = []
    for m in IDENT_RE.finditer(text):
        name = m.group(0)
        if name in kw:
            continue
        out.append((name, base + m.start(), base + m.end()))
    return out


def build_symbol_index(prompt: str, rec: dict, offsets: np.ndarray, n_ctx: int, sp_len: int) -> Dict:
    """
    prompt:  prompt cuoi (sau truncation) — cung chuoi ma offset Phase 1.4 duoc tinh tren do.
    rec:     dong meta Phase 1.4 (language, code_char_start, code_char_end).
    offsets: [num_tokens, 2] (start, end) ky tu cua tung token.
    """
    lang = rec["language"]
    cs, ce = rec["code_char_start"], rec["code_char_end"]
    starts = offsets[:, 0].astype(np.int64)
    ends = offsets[:, 1].astype(np.int64)
    idents = extract_identifiers(prompt[cs:ce], lang, base=cs)

    index: Dict[str, List[int]] = {}
    query_names = set()
    for name, s, e in idents:
        lo = int(np.searchsorted(ends, s, side="right"))     # token dau tien co end > s
        hi = int(np.searchsorted(starts, e, side="left"))    # token cuoi (exclusive) co start < e
        if hi <= lo:
            continue
        if lo < sp_len and hi > n_ctx:                       # cham cua so quan sat
            query_names.add(name)
        hi_c = min(hi, n_ctx)
        if lo < hi_c:                                        # vi tri trong fixed context (< n_ctx)
            index.setdefault(name, []).extend(range(lo, hi_c))
    index = {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in index.items()}
    q_ids = sorted(query_names)
    matched = [n for n in q_ids if n in index]
    return {"index": index, "query_ids": q_ids, "matched": matched,
            "n_index_ids": len(index),
            "n_matched_positions": int(sum(len(index[n]) for n in matched))}


def symbol_hit_from_labels(matched_positions: List[np.ndarray], labels: torch.Tensor, K: int) -> torch.Tensor:
    """
    matched_positions: list (moi dinh danh khop) cac vi tri token < n_ctx.
    labels: [H, S] long (label cluster SA cua tung key, theo tung head).
    Tra ve [H, K] float32 in [0,1]: ty le dinh danh khop ma cluster c dai dien (co it nhat mot vi tri nam trong c).
    """
    H = labels.shape[0]
    M = len(matched_positions)
    if M == 0:
        return torch.zeros(H, K, dtype=torch.float32, device=labels.device)
    pos = torch.from_numpy(np.concatenate(matched_positions)).to(labels.device)
    mid = torch.from_numpy(np.concatenate([np.full(len(p), i, dtype=np.int64)
                                           for i, p in enumerate(matched_positions)])).to(labels.device)
    lab_at = labels[:, pos]                                            # [H, T]
    key = mid.unsqueeze(0) * K + lab_at                                # [H, T]  (m, c)
    key = key + torch.arange(H, device=labels.device).unsqueeze(1) * (M * K)
    u = torch.unique(key.reshape(-1))                                  # cap (h, m, c) khac nhau
    h = u // (M * K)
    c = (u % (M * K)) % K
    cnt = torch.bincount(h * K + c, minlength=H * K).view(H, K).float()
    return cnt / M
