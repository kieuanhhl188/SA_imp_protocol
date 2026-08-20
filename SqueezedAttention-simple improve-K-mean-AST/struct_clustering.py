"""
struct_clustering.py — Phase 2: structure-aware clustering cho Squeezed Attention.

Ý tưởng (protocol Phase 2): đặt ranh giới CỨNG theo AST, chỉ cluster embedding BÊN TRONG
mỗi đơn vị cấu trúc. Hierarchy = token -> statement/block -> function -> file.

Khác biệt then chốt so với `ast_clustering.py` cũ
-------------------------------------------------
File cũ dùng AST để *khởi tạo* centroid rồi thả K-means chạy tự do trên toàn bộ token, nên
assignment vẫn vượt biên function — đó là "AST-aware init", KHÔNG phải ranh giới cứng.
Ở đây K-means chạy độc lập trong từng unit, token không bao giờ bị gán sang cluster của
unit khác. Đó mới là đề xuất 1 của protocol.

Ba phương pháp để ablation tách bạch được (protocol 2.5):
    sa               K-means thuần trên toàn bộ key (baseline, dùng squeezedattention/clustering.py)
    hard_boundary    K-means độc lập trong từng unit AST                      <- đề xuất 1
    struct_hierarchy hard_boundary ở L2 + L1 là trung bình theo unit cha       <- đề xuất 2

Giữ nguyên Si, threshold, kernel (protocol 2.6): module này CHỈ sinh centroid + label, cùng
layout với `squeezedattention.clustering.run_clustering` ([1,H,K,D] và [1,H,S]). Không đụng
tới cách tính điểm hay ngưỡng. KHÔNG có token-type weighting — thứ đó làm nhiễu ablation.

Phụ thuộc: tree_sitter + tree_sitter_<lang> (API mới, không cần tree_sitter_languages).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# =====================================================================
# 1. PARSE AST THEO NHIỀU LEVEL
# =====================================================================

# Level xếp từ thô tới mịn. Phase 7 quét cái này để vẽ hình sensitivity.
LEVELS = ("file", "class", "function", "block", "statement")

# Node type của tree-sitter theo từng ngôn ngữ, cho từng level.
NODE_TYPES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "python": {
        "class": ("class_definition",),
        "function": ("function_definition",),
        "block": ("block",),
        "statement": (
            "expression_statement", "if_statement", "for_statement", "while_statement",
            "return_statement", "assignment", "import_statement", "import_from_statement",
            "with_statement", "try_statement", "raise_statement", "assert_statement",
            "delete_statement", "global_statement", "decorated_definition",
        ),
    },
    "java": {
        "class": ("class_declaration", "interface_declaration", "enum_declaration"),
        "function": ("method_declaration", "constructor_declaration"),
        "block": ("block",),
        "statement": (
            "expression_statement", "if_statement", "for_statement", "while_statement",
            "return_statement", "local_variable_declaration", "import_declaration",
            "try_statement", "throw_statement", "switch_expression",
        ),
    },
    "javascript": {
        "class": ("class_declaration",),
        "function": ("function_declaration", "method_definition", "arrow_function",
                     "function_expression"),
        "block": ("statement_block",),
        "statement": (
            "expression_statement", "if_statement", "for_statement", "while_statement",
            "return_statement", "variable_declaration", "lexical_declaration",
            "import_statement", "try_statement",
        ),
    },
    "typescript": {
        "class": ("class_declaration", "interface_declaration"),
        "function": ("function_declaration", "method_definition", "arrow_function",
                     "function_expression"),
        "block": ("statement_block",),
        "statement": (
            "expression_statement", "if_statement", "for_statement", "while_statement",
            "return_statement", "variable_declaration", "lexical_declaration",
            "import_statement", "try_statement",
        ),
    },
    "csharp": {
        "class": ("class_declaration", "interface_declaration", "struct_declaration"),
        "function": ("method_declaration", "constructor_declaration"),
        "block": ("block",),
        "statement": (
            "expression_statement", "if_statement", "for_statement", "while_statement",
            "return_statement", "local_declaration_statement", "using_directive",
            "try_statement",
        ),
    },
}

_PARSER_CACHE: Dict[str, object] = {}


def get_parser(language: str):
    """Tạo (và cache) tree-sitter parser. Dùng API mới của tree_sitter >= 0.22."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    from tree_sitter import Language, Parser

    modname = {
        "python": "tree_sitter_python",
        "java": "tree_sitter_java",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "csharp": "tree_sitter_c_sharp",
    }[language]
    mod = __import__(modname)
    if language == "typescript":
        lang = Language(mod.language_typescript())
    else:
        lang = Language(mod.language())

    parser = Parser(lang)
    _PARSER_CACHE[language] = parser
    return parser


def byte_to_char_index(code: str) -> Optional[List[int]]:
    """
    Bảng tra `b2c[i]` = chỉ số KÝ TỰ của byte thứ i trong `code`. Trả None nếu code thuần
    ASCII (khi đó byte và ký tự trùng nhau, khỏi tốn bộ nhớ).

    VÌ SAO CẦN: tree-sitter đánh địa chỉ theo BYTE (`node.start_byte`), còn offset của
    Phase 1.4 là KÝ TỰ (`return_offsets_mapping` của tokenizer nhanh). Cộng thẳng hai loại
    vào nhau thì mọi span sau ký tự non-ASCII đầu tiên đều lệch, và độ lệch cộng dồn.
    Đo trên dữ liệu thật: LCC 0/500 mẫu có non-ASCII (không ảnh hưởng), RepoBench-P
    **88/500 mẫu (17,6%)** có — tức gần 1/5 dataset sẽ gán token sai unit nếu không quy đổi.
    """
    code_bytes = code.encode("utf-8")
    n_bytes = len(code_bytes)
    if n_bytes == len(code):
        return None
    b2c = [0] * (n_bytes + 1)
    b = 0
    for ci, ch in enumerate(code):
        w = len(ch.encode("utf-8"))
        for k in range(w):
            b2c[b + k] = ci
        b += w
    b2c[n_bytes] = len(code)
    return b2c


def parse_units(
    code: str,
    language: str = "python",
    level: str = "function",
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    """
    Trả về các span [start_char, end_char) của đơn vị cấu trúc ở `level`, cộng thống kê.

    ĐƠN VỊ LÀ KÝ TỰ, không phải byte — để khớp thẳng với offset Phase 1.4. tree-sitter làm
    việc theo byte, hàm này quy đổi lại qua `byte_to_char_index`. Với code thuần ASCII hai
    đơn vị trùng nhau nên không có chi phí gì.

    Level thô hơn được gộp vào: level="function" cũng nhận class (một class không có method
    vẫn là một đơn vị), level="block" nhận cả function/class, v.v. Nhờ vậy mọi token đều có
    đơn vị bao nó, và các level xếp thành một hierarchy thật.

    level="file" trả về đúng một span phủ toàn bộ code.
    """
    if level not in LEVELS:
        raise ValueError(f"level phải thuộc {LEVELS}, nhận '{level}'")

    code_bytes = code.encode("utf-8")
    if level == "file":
        return [(0, len(code))], {"num_error_nodes": 0, "num_units": 1}

    if language not in NODE_TYPES:
        raise ValueError(f"chưa hỗ trợ ngôn ngữ '{language}', có: {sorted(NODE_TYPES)}")

    # gộp mọi level thô hơn hoặc bằng level yêu cầu
    order = list(LEVELS)
    wanted = set()
    for lv in order[1:order.index(level) + 1]:
        wanted.update(NODE_TYPES[language].get(lv, ()))

    parser = get_parser(language)
    tree = parser.parse(code_bytes)

    spans: List[Tuple[int, int]] = []
    n_error = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            n_error += 1
        if node.type in wanted and node.end_byte > node.start_byte:
            spans.append((node.start_byte, node.end_byte))
        stack.extend(node.children)

    # byte -> ký tự, để khớp với offset của Phase 1.4
    b2c = byte_to_char_index(code)
    if b2c is not None:
        spans = [(b2c[s], b2c[e]) for s, e in spans]

    # luôn có span phủ toàn bộ file, để token ngoài mọi unit (import rời, code top-level)
    # vẫn có đơn vị bao — không token nào bị bỏ rơi.
    spans.append((0, len(code)))

    return spans, {"num_error_nodes": n_error, "num_units": len(spans)}


# =====================================================================
# 2. GÁN unit_id CHO TỪNG TOKEN
# =====================================================================

def assign_token_units(
    token_starts: torch.Tensor,
    spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """
    Gán mỗi token vào span NHỎ NHẤT bao nó.

    token_starts: [S] long, offset ký tự/byte của đầu mỗi token, ĐÃ SẮP TĂNG DẦN.
    spans:        list [(start, end)).

    Trả về [S] long, giá trị trong [0, len(spans)).

    Thuật toán: sắp span theo kích thước giảm dần rồi ghi đè. Span nhỏ xử lý sau nên
    thắng — đúng nghĩa "nhỏ nhất bao nó". Mỗi span chỉ tốn 2 lần searchsorted, tổng
    O(U log S). Bản cũ `map_tokens_to_scopes` lặp Python O(S x U); với S=31K, U=500 là
    15 triệu vòng — đó là lý do nó không dùng được.
    """
    S = token_starts.numel()
    unit_ids = torch.zeros(S, dtype=torch.long)
    if S == 0:
        return unit_ids

    order = sorted(range(len(spans)), key=lambda i: -(spans[i][1] - spans[i][0]))
    for idx in order:
        s, e = spans[idx]
        lo = int(torch.searchsorted(token_starts, torch.tensor(s), right=False))
        hi = int(torch.searchsorted(token_starts, torch.tensor(e), right=False))
        if hi > lo:
            unit_ids[lo:hi] = idx
    return unit_ids


def compact_unit_ids(unit_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ép unit_id về dải liên tục [0, U). Bỏ các span không chứa token nào.
    Trả về (unit_ids_mới, unit_ids_gốc_theo_thứ_tự_mới).
    """
    uniq, inv = torch.unique(unit_ids, return_inverse=True)
    return inv, uniq


# =====================================================================
# 3. PHÂN BỔ NGÂN SÁCH CENTROID
# =====================================================================

class BudgetExceeded(ValueError):
    """
    Số unit cấu trúc vượt ngân sách centroid — mẫu này KHÔNG biểu diễn được ở level và
    ngân sách hiện tại.

    Là exception riêng (không phải ValueError trần) để người gọi phân biệt được "mẫu không
    khả thi" với "code có bug", và quyết định chính sách: bỏ qua-và-ghi-nhận, hay gộp unit.
    Mặc định của Phase 2 là BỎ QUA: gộp rồi báo cáo như không có gì xảy ra sẽ biến một giới
    hạn ngân sách thành một kết luận về cấu trúc.
    """

    def __init__(self, msg, num_units=None, budget=None):
        super().__init__(msg)
        self.num_units = num_units
        self.budget = budget


def merge_units_to_budget(
    unit_ids: torch.Tensor,
    target_units: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Gộp unit LIỀN KỀ THEO VỊ TRÍ trong code cho tới khi còn `target_units` unit.

    CHỈ dùng cho nhánh biến thể có-ràng-buộc-ngân-sách, KHÔNG dùng trong thí nghiệm chính.
    Người gọi phải ghi lại `frac_units_merged` và báo cáo — gộp 60% số unit rồi vẫn gọi đó
    là "statement level" là sai sự thật.

    Cách gộp giống hệt nhánh "merge" của `build_l1_groups`: sắp unit theo lần xuất hiện đầu
    tiên trong chuỗi token, rồi cắt thành `target_units` đoạn liền kề cân theo số token.
    Tất định tuyệt đối, không RNG. Nhờ liền kề theo vị trí, unit gộp vẫn là một vùng code
    liền mạch chứ không phải một tập rời rạc.
    """
    unit_ids, _ = compact_unit_ids(unit_ids)
    U = int(unit_ids.max()) + 1
    if target_units >= U:
        return unit_ids, {"merged": False, "u_before": U, "u_after": U,
                          "frac_units_merged": 0.0}
    if target_units < 1:
        raise ValueError(f"target_units phải >= 1, nhận {target_units}")

    sizes = torch.bincount(unit_ids, minlength=U)
    first_pos = torch.tensor([int((unit_ids == u).nonzero()[0]) for u in range(U)])
    order = torch.argsort(first_pos)          # unit theo thứ tự xuất hiện trong code
    sizes_ord = sizes[order]
    total = float(sizes_ord.sum())
    quota = total / target_units

    group_of_unit = torch.zeros(U, dtype=torch.long)
    g, acc = 0, 0.0
    for i, u in enumerate(order.tolist()):
        group_of_unit[u] = min(g, target_units - 1)
        acc += float(sizes_ord[i])
        # sang nhóm mới khi đủ hạn ngạch, nhưng phải chừa đủ unit cho các nhóm còn lại
        if (acc >= quota * (g + 1) and g < target_units - 1
                and (U - i - 1) >= (target_units - g - 1)):
            g += 1

    out, _ = compact_unit_ids(group_of_unit[unit_ids])
    u_after = int(out.max()) + 1
    return out, {"merged": True, "u_before": U, "u_after": u_after,
                 "frac_units_merged": (U - u_after) / U}

def allocate_centroids(
    unit_sizes: torch.Tensor,
    num_centroids_total: int,
    max_k_per_unit: Optional[int] = None,
    caps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Chia `num_centroids_total` centroid cho các unit, tỉ lệ theo số token.

    Ràng buộc (protocol 2.3):
      - mỗi unit ít nhất 1 centroid  ("unit nhỏ (< ngưỡng) -> 1 centroid")
      - không quá số token của unit  (K-means không thể có nhiều cluster hơn điểm)
      - không quá max_k_per_unit     (nếu đặt; xem bên dưới)
      - TỔNG đúng bằng num_centroids_total, để so công bằng với SA ở cùng budget

    `max_k_per_unit=None` (MẶC ĐỊNH) nghĩa là chỉ chặn bởi số token của unit.

    VÌ SAO MẶC ĐỊNH KHÔNG CÒN LÀ 64: trần cứng 64 làm `sum(cap)` có thể NHỎ HƠN ngân sách
    khi context dài mà ít unit — lúc đó hàm không tiêu hết ngân sách và raise, dù ngân sách
    còn thừa mênh mông. Đo trên dữ liệu thật (5% budget): LCC **236/499 mẫu** ở level=class
    và **22/499** ở level=function rơi vào đúng nhánh này. Đó là hằng số cài đặt chặn thí
    nghiệm, không phải ràng buộc của phương pháp. Bỏ trần thì `sum(cap) = sum(unit_sizes)
    = n_ctx`, mà ngân sách chỉ là vài phần trăm của `n_ctx`, nên nhánh đó không chạm tới
    được nữa. Vẫn giữ tham số để chặn chi phí padding khi cần.

    Nếu số unit > ngân sách thì không thể cho mỗi unit 1 centroid — hàm raise
    `BudgetExceeded`, vì im lặng cắt bớt sẽ làm hỏng tính "cùng budget" của thí nghiệm.
    Người gọi quyết định làm gì tiếp (bỏ qua mẫu, hay gộp unit): xem `merge_units_to_budget`.
    """
    U = unit_sizes.numel()
    if num_centroids_total < U:
        raise BudgetExceeded(
            f"ngân sách {num_centroids_total} centroid < {U} unit. Mỗi unit cần tối thiểu "
            f"1 centroid. Dùng level thô hơn (function thay vì statement), tăng "
            f"percent_clusters, hoặc gộp unit bằng merge_units_to_budget().",
            num_units=U, budget=num_centroids_total,
        )

    # `caps` cho phép caller đặt trần riêng cho từng unit. build_l1_groups cần: một unit cha
    # chỉ tách được tối đa bằng số unit con của nó, không liên quan tới số token.
    base = unit_sizes if caps is None else caps
    if max_k_per_unit is None:
        cap = base.clone()
    else:
        cap = torch.minimum(base, torch.full_like(base, max_k_per_unit))
    k = torch.ones(U, dtype=torch.long)

    remaining = num_centroids_total - U
    if remaining > 0:
        headroom = cap - k
        share = unit_sizes.double() / max(float(unit_sizes.sum()), 1.0)
        extra = torch.minimum((share * remaining).floor().long(), headroom)
        k = k + extra
        remaining = num_centroids_total - int(k.sum())

        # chia phần dư theo thứ tự unit lớn trước, lặp tới khi hết hoặc hết chỗ
        while remaining > 0:
            headroom = cap - k
            cand = torch.nonzero(headroom > 0, as_tuple=True)[0]
            if cand.numel() == 0:
                raise BudgetExceeded(
                    f"không phân bổ hết {remaining} centroid: mọi unit đã chạm trần "
                    f"(max_k_per_unit={max_k_per_unit} hoặc số token). Đặt "
                    f"max_k_per_unit=None để chỉ chặn theo số token.",
                    num_units=U, budget=num_centroids_total,
                )
            take = cand[torch.argsort(unit_sizes[cand], descending=True)][:remaining]
            k[take] += 1
            remaining = num_centroids_total - int(k.sum())

    assert int(k.sum()) == num_centroids_total, (int(k.sum()), num_centroids_total)
    assert bool((k >= 1).all()) and bool((k <= cap).all())
    return k


# =====================================================================
# 4. K-MEANS RANH GIỚI CỨNG
# =====================================================================

def _bucket_by_size(sizes: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Gom unit theo kích thước làm tròn lên lũy thừa 2, để batch mà không phí padding quá."""
    buckets: Dict[int, List[int]] = {}
    for i, n in enumerate(sizes.tolist()):
        p = 1
        while p < n:
            p <<= 1
        buckets.setdefault(p, []).append(i)
    return {p: torch.tensor(v, dtype=torch.long) for p, v in buckets.items()}


def hard_boundary_kmeans(
    keys: torch.Tensor,
    unit_ids: torch.Tensor,
    num_centroids_total: int,
    n_iter: int = 10,
    max_k_per_unit: Optional[int] = None,
    max_batch_elems: int = 64_000_000,
    device: Optional[torch.device] = None,
    token_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    K-means chạy ĐỘC LẬP trong từng unit. Token không bao giờ nhảy sang cluster của unit khác.

    keys:      [H, S, D]  key vector của fixed context (đã bỏ observation window)
    unit_ids:  [S]        unit của từng token, giá trị trong [0, U)
    token_weights: [S] hoặc None. **Mặc định None = không trọng số.**
        Đây là cờ tắt sẵn cho Hướng 2(b) (token-type weighted distance) đã có trong repo.
        Trọng số chỉ tác động vào bước CẬP NHẬT centroid (token nặng kéo centroid mạnh hơn),
        KHÔNG tác động vào bước gán cluster — đúng như mô tả gốc của Hướng 2(b).
        Protocol không có nhánh này trong bảng ablation, nên mặc định phải tắt: bật lên là
        thêm một biến vào thí nghiệm, phải báo cáo riêng.

    Trả về:
        centroids [1, H, K, D], labels [1, H, S]  — cùng layout với run_clustering của SA,
        cộng dict thống kê.

    Cài đặt: gom unit theo bucket kích thước lũy-thừa-2 rồi chạy K-means batch trên
    [H, U_b, P, D]. Số lần gọi kernel là O(log(max_unit_size)) mỗi vòng lặp, không phải
    O(U) hay O(U x H).

    Khởi tạo tất định (linspace trong unit) — không dùng RNG, nên tái lập được tuyệt đối.
    """
    if device is None:
        device = keys.device
    keys = keys.to(device)
    unit_ids = unit_ids.to(device)

    H, S, D = keys.shape
    assert unit_ids.shape == (S,), (unit_ids.shape, S)

    if token_weights is not None:
        token_weights = token_weights.to(device=device, dtype=keys.dtype)
        assert token_weights.shape == (S,), (token_weights.shape, S)
        if bool((token_weights < 0).any()):
            raise ValueError("token_weights phải không âm")

    unit_ids, _ = compact_unit_ids(unit_ids)
    U = int(unit_ids.max()) + 1
    sizes = torch.bincount(unit_ids, minlength=U)

    k_per_unit = allocate_centroids(sizes.cpu(), num_centroids_total, max_k_per_unit).to(device)
    cluster_offset = torch.cumsum(k_per_unit, 0) - k_per_unit

    # sắp token theo unit -> mỗi unit là một lát liên tục
    order = torch.argsort(unit_ids, stable=True)
    tok_offset = torch.cumsum(sizes, 0) - sizes

    centroids = torch.zeros(H, num_centroids_total, D, dtype=keys.dtype, device=device)
    labels = torch.zeros(H, S, dtype=torch.long, device=device)

    buckets = _bucket_by_size(sizes.cpu())
    n_kernel_calls = 0

    for pad_n, unit_idx in sorted(buckets.items()):
        unit_idx = unit_idx.to(device)
        pad_k = int(k_per_unit[unit_idx].max())

        # chia nhỏ bucket để không nổ bộ nhớ
        chunk = max(1, min(len(unit_idx), max_batch_elems // max(H * pad_n * D, 1)))
        for c0 in range(0, len(unit_idx), chunk):
            ui = unit_idx[c0:c0 + chunk]
            B = ui.numel()
            n_u = sizes[ui]                                   # [B]
            k_u = k_per_unit[ui]                              # [B]

            # ---- gom key của các unit trong chunk vào tensor có padding ----
            pos = torch.arange(pad_n, device=device).unsqueeze(0)          # [1, P]
            valid = pos < n_u.unsqueeze(1)                                 # [B, P]
            gather_idx = (tok_offset[ui].unsqueeze(1) + pos).clamp(
                max=S - 1)                                                 # [B, P]
            tok_idx = order[gather_idx]                                    # [B, P]
            blk = keys[:, tok_idx, :]                                      # [H, B, P, D]

            # ---- khởi tạo centroid: linspace trong từng unit, tất định ----
            kpos = torch.arange(pad_k, device=device).unsqueeze(0)         # [1, Kp]
            k_valid = kpos < k_u.unsqueeze(1)                              # [B, Kp]
            denom = (k_u - 1).clamp(min=1).unsqueeze(1)
            init_rel = (kpos * (n_u - 1).unsqueeze(1) // denom).clamp(min=0)
            init_rel = torch.where(k_u.unsqueeze(1) == 1,
                                   torch.zeros_like(init_rel), init_rel)
            init_rel = init_rel.clamp(max=(n_u - 1).clamp(min=0).unsqueeze(1))
            cent = torch.gather(
                blk, 2, init_rel.unsqueeze(0).unsqueeze(-1).expand(H, B, pad_k, D)
            )                                                              # [H, B, Kp, D]

            NEG = torch.finfo(blk.dtype).max
            for _ in range(n_iter):
                # [H, B, P, Kp]
                dist = torch.cdist(blk.reshape(H * B, pad_n, D),
                                   cent.reshape(H * B, pad_k, D)).reshape(H, B, pad_n, pad_k)
                dist = dist.masked_fill(~k_valid.unsqueeze(0).unsqueeze(2), NEG)
                asg = dist.argmin(dim=-1)                                  # [H, B, P]
                asg = torch.where(valid.unsqueeze(0), asg, torch.zeros_like(asg))

                new = torch.zeros_like(cent)
                cnt = torch.zeros(H, B, pad_k, dtype=blk.dtype, device=device)
                w = valid.to(blk.dtype)                                    # [B, P]
                if token_weights is not None:
                    w = w * token_weights[tok_idx]                         # [B, P]
                w = w.unsqueeze(0)                                         # [1, B, P]
                new.scatter_add_(2, asg.unsqueeze(-1).expand(H, B, pad_n, D),
                                 blk * w.unsqueeze(-1))
                cnt.scatter_add_(2, asg, w.expand(H, B, pad_n))
                # Mẫu số là TỔNG TRỌNG SỐ, không phải số đếm. Không được clamp(min=1):
                # với trọng số nhỏ hơn 1 (Hướng 2(b) giảm dấu câu còn 0.5) thì clamp sẽ
                # bóp méo trung bình. Chỉ chặn chia cho 0.
                eps = torch.finfo(blk.dtype).tiny
                nonempty = cnt > eps
                # cluster rỗng thì giữ nguyên centroid cũ, tránh NaN
                cent = torch.where(nonempty.unsqueeze(-1),
                                   new / cnt.clamp(min=eps).unsqueeze(-1), cent)
                n_kernel_calls += 1

            # ---- ghi kết quả về mảng toàn cục ----
            g_cent = cluster_offset[ui].unsqueeze(1) + kpos                # [B, Kp]
            sel = k_valid.reshape(-1)
            centroids[:, g_cent.reshape(-1)[sel], :] = cent.reshape(H, -1, D)[:, sel, :]

            g_lab = cluster_offset[ui].unsqueeze(0).unsqueeze(-1) + asg    # [H, B, P]
            flat_tok = tok_idx.reshape(-1)
            flat_val = valid.reshape(-1)
            labels[:, flat_tok[flat_val]] = g_lab.reshape(H, -1)[:, flat_val]

    stats = {
        "token_weighted": token_weights is not None,
        "num_units": U,
        "num_centroids": num_centroids_total,
        "unit_size_min": int(sizes.min()),
        "unit_size_mean": float(sizes.double().mean()),
        "unit_size_max": int(sizes.max()),
        "k_per_unit_min": int(k_per_unit.min()),
        "k_per_unit_max": int(k_per_unit.max()),
        "num_buckets": len(buckets),
        "num_kernel_calls": n_kernel_calls,
    }
    return centroids.unsqueeze(0), labels.unsqueeze(0), stats


# =====================================================================
# 4b. TRỌNG SỐ THEO LOẠI TOKEN — Hướng 2(b), TẮT SẴN
# =====================================================================
#
# Đây KHÔNG phải một mục của protocol. Protocol Phase 2 chỉ có 3 nhánh ablation
# (SA / +HardBoundary / +StructHierarchy) và mục 2.6 yêu cầu giữ nguyên Si/threshold/kernel.
# Giữ lại vì đó là ý tưởng đã có sẵn trong repo (Hướng 2(b) trong README_EXTENSIONS.md).
# Bật lên là thêm một biến vào thí nghiệm -> phải báo cáo thành nhánh riêng, không được
# trộn vào con số của +HardBoundary.

# Trọng số theo lớp node của tree-sitter. Số mặc định lấy đúng theo bản Hướng 2(b) cũ
# (`ast_clustering.py::compute_token_type_weights`): identifier 1.5x, literal 1.2x,
# dấu câu 0.5x, còn lại 1.0x.
DEFAULT_TYPE_WEIGHTS = {
    "identifier": 1.5,
    "literal": 1.2,
    "punctuation": 0.5,
    "comment": 0.5,
    "keyword": 1.0,
    "other": 1.0,
}

_IDENTIFIER_TYPES = {
    "identifier", "type_identifier", "field_identifier", "property_identifier",
    "attribute", "dotted_name", "scoped_identifier",
}
_LITERAL_TYPES = {
    "string", "string_literal", "integer", "float", "number", "decimal_integer_literal",
    "true", "false", "none", "null", "string_content", "character_literal",
}
_COMMENT_TYPES = {"comment", "line_comment", "block_comment"}


def classify_leaf(node_type: str) -> str:
    """Xếp một leaf node của tree-sitter vào lớp trọng số."""
    if node_type in _IDENTIFIER_TYPES:
        return "identifier"
    if node_type in _LITERAL_TYPES:
        return "literal"
    if node_type in _COMMENT_TYPES:
        return "comment"
    if not node_type[:1].isalpha():
        return "punctuation"
    return "keyword"


def compute_token_type_weights(
    code: str,
    token_starts: torch.Tensor,
    language: str = "python",
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Trọng số cho từng token theo loại node AST bao nó. Trả về [S] float.

    Khác bản cũ ở chỗ phân loại bằng **node type của tree-sitter** thay vì regex trên chuỗi
    token đã decode. Bản cũ decode từng token một (`tokenizer.decode([tok_id])` trong vòng
    lặp Python, S lần gọi) và không phân biệt được identifier với keyword — `return` và
    `total` đều khớp `^[a-zA-Z_][a-zA-Z0-9_]{1,}$`. Ở đây tận dụng cây đã parse sẵn nên
    gần như miễn phí và phân loại đúng hơn.
    """
    w = dict(DEFAULT_TYPE_WEIGHTS)
    if weights:
        w.update(weights)

    S = token_starts.numel()
    out = torch.full((S,), float(w["other"]))
    if S == 0:
        return out

    code_bytes = code.encode("utf-8")
    parser = get_parser(language)
    tree = parser.parse(code_bytes)
    # `token_starts` là offset KÝ TỰ (Phase 1.4) còn tree-sitter đếm byte -> quy đổi,
    # giống hệt lý do ở `parse_units`.
    b2c = byte_to_char_index(code)

    # Duyệt leaf node, gán trọng số cho mọi token nằm trong khoảng của leaf đó.
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
            continue
        if node.end_byte <= node.start_byte:
            continue
        cls = classify_leaf(node.type)
        s_char = node.start_byte if b2c is None else b2c[node.start_byte]
        e_char = node.end_byte if b2c is None else b2c[node.end_byte]
        lo = int(torch.searchsorted(token_starts, torch.tensor(s_char), right=False))
        hi = int(torch.searchsorted(token_starts, torch.tensor(e_char), right=False))
        if hi > lo:
            out[lo:hi] = float(w[cls])
    return out


# =====================================================================
# 5. HIERARCHY THEO CẤU TRÚC (đề xuất 2)
# =====================================================================

def build_l1_groups(
    unit_ids_l2: torch.Tensor,
    unit_ids_l1_raw: torch.Tensor,
    target_k1: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Ép số nhóm L1 về đúng `target_k1`, vẫn tôn trọng ranh giới cấu trúc.

    VÌ SAO CẦN: protocol đặt L1 = 1% context length. Nhưng "trung bình theo function/file"
    cho ra số nhóm = số unit cha mà code TÌNH CỜ có — 5 class thì K1=5 (bộ lọc L1 vô dụng),
    300 function thì K1 vượt xa 1%. Mà metadata centroid ĐƯỢC TÍNH VÀO KV budget (bài: đơn
    tầng 2.5%, phân tầng 3%), nên K1 lệch là budget lệch, so sánh mất công bằng — đúng điều
    "Ghi chú kiểm soát" của protocol cấm.

    Cách xử lý, giữ nguyên tính cấu trúc:
      - nhiều unit cha hơn mục tiêu -> GỘP các unit cha LIỀN KỀ (code liền nhau), cân theo
        số token
      - ít unit cha hơn mục tiêu    -> TÁCH mỗi unit cha thành nhiều nhóm con, mỗi nhóm là
        một dãy unit L2 liền kề; số nhóm chia theo `allocate_centroids`

    Trả về (l1_group_of_token [S], thống kê).
    """
    l2, _ = compact_unit_ids(unit_ids_l2)
    l1, _ = compact_unit_ids(unit_ids_l1_raw)
    U1 = int(l1.max()) + 1
    stats: Dict[str, object] = {"k1_raw": U1, "k1_target": target_k1}

    if target_k1 <= 0 or target_k1 == U1:
        stats["k1_mode"] = "as-is"
        stats["k1_actual"] = U1
        return l1, stats

    parent_sizes = torch.bincount(l1, minlength=U1)

    if target_k1 < U1:
        # ---- GỘP unit cha liền kề, cân theo số token ----
        order = torch.argsort(
            torch.tensor([int((l1 == u).nonzero()[0]) for u in range(U1)])
        )  # unit cha theo thứ tự xuất hiện trong code
        sizes_ord = parent_sizes[order]
        total = int(sizes_ord.sum())
        quota = total / target_k1

        group_of_parent = torch.zeros(U1, dtype=torch.long)
        g, acc = 0, 0.0
        for i, u in enumerate(order.tolist()):
            group_of_parent[u] = min(g, target_k1 - 1)
            acc += float(sizes_ord[i])
            # sang nhóm mới khi đã đủ hạn ngạch, nhưng phải chừa đủ unit cho nhóm còn lại
            if acc >= quota * (g + 1) and g < target_k1 - 1 and (U1 - i - 1) >= (target_k1 - g - 1):
                g += 1
        out = group_of_parent[l1]
        stats["k1_mode"] = "merge"
    else:
        # ---- TÁCH mỗi unit cha thành nhiều nhóm con theo dãy unit L2 liền kề ----
        # Trần của mỗi unit cha là SỐ UNIT CON của nó: không thể tách 2 function thành 3 nhóm.
        child_count = torch.tensor(
            [int(torch.unique(l2[l1 == p]).numel()) for p in range(U1)], dtype=torch.long
        )
        max_possible = int(child_count.sum())
        if target_k1 > max_possible:
            stats["k1_clamped_from"] = target_k1
            target_k1 = max_possible
            stats["k1_target"] = target_k1
        n_groups = allocate_centroids(parent_sizes, target_k1,
                                      max_k_per_unit=max_possible, caps=child_count)
        offset = torch.cumsum(n_groups, 0) - n_groups
        out = torch.zeros_like(l1)
        for p in range(U1):
            mask = l1 == p
            child = torch.unique(l2[mask])          # các unit L2 nằm trong unit cha p
            n_child, n_g = child.numel(), int(n_groups[p])
            # chia dãy unit con thành n_g đoạn liền kề, càng đều càng tốt
            edges = (torch.arange(n_child) * n_g) // max(n_child, 1)
            sub_of_child = {int(c): int(e) for c, e in zip(child.tolist(), edges.tolist())}
            idx = mask.nonzero(as_tuple=True)[0]
            out[idx] = int(offset[p]) + torch.tensor(
                [sub_of_child[int(v)] for v in l2[idx].tolist()], dtype=torch.long)
        stats["k1_mode"] = "split"

    out, _ = compact_unit_ids(out)
    stats["k1_actual"] = int(out.max()) + 1
    return out, stats


def struct_hierarchy_l1(
    centroids_l2: torch.Tensor,
    labels_l2: torch.Tensor,
    unit_ids_l1: torch.Tensor,
    weighted: bool = True,
    unit_ids_l2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dựng tầng L1 bằng cấu trúc thật, thay cho "K-means của K-means" của bài.

    Protocol: "Level-2 centroid = trong-function; Level-1 centroid = trung bình theo
    function/file."

    centroids_l2: [1, H, K2, D]
    labels_l2:    [1, H, S]      key -> cluster L2
    unit_ids_l1:  [S]            key -> nhóm L1 (nên lấy từ `build_l1_groups`)
    weighted:     True  -> trung bình CÓ TRỌNG SỐ theo số key mỗi cluster L2. Khi đó L1
                           centroid đúng bằng trung bình của toàn bộ key trong nhóm — đại
                           diện đúng cho nhóm.
                  False -> trung bình cộng đơn thuần các L2 centroid; cluster 3 token nặng
                           ngang cluster 200 token. Giữ lại để ablation.
    unit_ids_l2:  tuỳ chọn, chỉ dùng để assert mỗi unit L2 nằm gọn trong một nhóm L1.

    Trả về (centroids_l1 [1, H, K1, D], labels_l1 [1, H, S]).
    labels_l1 ánh xạ THẲNG key -> cluster L1, đúng dạng code eval cần (bài gốc phải gather
    L1->L2->key, ở đây có sẵn nên bỏ được bước đó).
    """
    c2 = centroids_l2.squeeze(0)                    # [H, K2, D]
    l2 = labels_l2.squeeze(0)                       # [H, S]
    H, K2, D = c2.shape
    dev = c2.device

    l1_compact, _ = compact_unit_ids(unit_ids_l1)
    l1_compact = l1_compact.to(dev)
    K1 = int(l1_compact.max()) + 1

    if unit_ids_l2 is not None:
        u2, _ = compact_unit_ids(unit_ids_l2)
        for u in torch.unique(u2):
            if torch.unique(l1_compact[u2.to(dev) == u.to(dev)]).numel() != 1:
                raise ValueError(
                    f"unit L2 {int(u)} nằm vắt qua nhiều nhóm L1 — hierarchy không lồng nhau"
                )

    # cluster L2 -> nhóm L1. Mọi key của một cluster L2 thuộc cùng unit L2, mà mỗi unit L2
    # nằm gọn trong một nhóm L1, nên ánh xạ này xác định duy nhất.
    cl2_to_l1 = torch.zeros(K2, dtype=torch.long, device=dev)
    cl2_to_l1.scatter_(0, l2[0], l1_compact)

    # Trọng số mỗi cluster L2 = số key của nó, tính RIÊNG TỪNG HEAD.
    # Ranh giới cứng chỉ đảm bảo mọi head có cùng phân hoạch theo UNIT; bên trong một unit
    # thì mỗi head chia cluster khác nhau, nên số key mỗi cluster khác nhau theo head.
    # Dùng số đếm của head 0 cho mọi head là sai — L1 centroid sẽ lệch khỏi trung bình key.
    if weighted:
        w = torch.zeros(H, K2, dtype=c2.dtype, device=dev)
        w.scatter_add_(1, l2, torch.ones_like(l2, dtype=c2.dtype))
    else:
        w = torch.ones(H, K2, dtype=c2.dtype, device=dev)

    c1 = torch.zeros(H, K1, D, dtype=c2.dtype, device=dev)
    den = torch.zeros(H, K1, dtype=c2.dtype, device=dev)
    idx = cl2_to_l1.unsqueeze(0).unsqueeze(-1).expand(H, K2, D)
    c1.scatter_add_(1, idx, c2 * w.unsqueeze(-1))
    den.scatter_add_(1, cl2_to_l1.unsqueeze(0).expand(H, K2), w)

    eps = torch.finfo(c2.dtype).tiny
    c1 = c1 / den.clamp(min=eps).unsqueeze(-1)

    labels_l1 = l1_compact.unsqueeze(0).expand(H, -1).contiguous()
    return c1.unsqueeze(0), labels_l1.unsqueeze(0)
