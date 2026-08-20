"""
AST-aware clustering initialization for Squeezed Attention on code.

Hướng 2: Code-aware Clustering
==============================
Quan sát: K-means thuần ngữ nghĩa không capture được code structure:
  - Identifier (variable, function name) lặp nhiều → semantically related
  - Code trong cùng function/class scope → semantically related
  - Comment vs code: hai distribution rất khác

Đề xuất: dùng AST của code để init centroids tốt hơn, thay vì random init.
Hai approach:
  (a) Scope-based init: 1 centroid per function/class
  (b) Token-type weighted distance: weight K-means distance khác nhau theo token type

Author: <your name>

Dependencies:
    pip install tree-sitter tree-sitter-languages
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


# =====================================================================
# TREE-SITTER WRAPPER (optional, fallback to regex if unavailable)
# =====================================================================

def _try_import_tree_sitter():
    """Tree-sitter dependency là optional. Fallback nếu không có."""
    try:
        import tree_sitter_languages
        return tree_sitter_languages
    except ImportError:
        print("[WARN] tree_sitter_languages chưa cài. Fallback dùng regex.")
        print("       pip install tree-sitter tree-sitter-languages")
        return None


def parse_code_to_scopes(code: str, language: str = "python") -> List[Tuple[int, int, str]]:
    """
    Parse code, trả về list các scope (function, class, block).

    ⚠️ File này là bản CŨ, Phase 2 KHÔNG dùng (xem `struct_clustering.py`). Nó trả toạ độ
    BYTE của tree-sitter rồi so thẳng với `offset_mapping` của tokenizer — vốn là KÝ TỰ.
    Hai đơn vị chỉ trùng nhau khi code thuần ASCII; với mẫu có Unicode (17,6% RepoBench-P)
    span sẽ lệch dần. `struct_clustering.byte_to_char_index` xử lý đúng chỗ này.

    Returns: list of (start_byte, end_byte, scope_type)
             - scope_type: 'function' | 'class' | 'block'

    Nếu tree-sitter không có, fallback regex đơn giản (chỉ cho Python).
    """
    tsl = _try_import_tree_sitter()

    if tsl is None:
        return _fallback_regex_scopes(code, language)

    parser = tsl.get_parser(language)
    tree = parser.parse(bytes(code, "utf-8"))

    scopes = []
    # Mapping cho từng language; mở rộng tùy ngôn ngữ
    scope_nodes_map = {
        "python":     ["function_definition", "class_definition"],
        "javascript": ["function_declaration", "class_declaration", "method_definition"],
        "java":       ["method_declaration", "class_declaration"],
        "cpp":        ["function_definition", "class_specifier"],
        "go":         ["function_declaration", "method_declaration"],
    }
    target_types = scope_nodes_map.get(language, ["function_definition", "class_definition"])

    def walk(node):
        if node.type in target_types:
            scope_type = "function" if "function" in node.type or "method" in node.type else "class"
            scopes.append((node.start_byte, node.end_byte, scope_type))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return scopes


def _fallback_regex_scopes(code: str, language: str) -> List[Tuple[int, int, str]]:
    """Regex-based fallback. Kém chính xác hơn AST nhưng không cần dependency."""
    import re
    scopes = []
    if language == "python":
        # Tìm 'def' và 'class' (đơn giản, không xử lý nested chính xác)
        for m in re.finditer(r"^def\s+\w+", code, re.MULTILINE):
            scopes.append((m.start(), m.end() + 100, "function"))  # heuristic
        for m in re.finditer(r"^class\s+\w+", code, re.MULTILINE):
            scopes.append((m.start(), m.end() + 100, "class"))
    return scopes


# =====================================================================
# TOKEN-TO-SCOPE MAPPING
# =====================================================================

def map_tokens_to_scopes(
    code: str,
    tokenizer,
    scopes: List[Tuple[int, int, str]],
) -> torch.Tensor:
    """
    Map mỗi token tới scope ID của nó.

    Returns: [seq_len] long tensor, mỗi giá trị là scope_id.
             -1 nếu token không thuộc scope nào (top-level statement).
    """
    enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.get("offset_mapping", None)

    if offsets is None:
        # Một số tokenizer không support offset_mapping -> tokenize từng phần
        return _approximate_scope_map(code, tokenizer, scopes)

    num_tokens = len(offsets)
    scope_ids = torch.full((num_tokens,), -1, dtype=torch.long)

    for tok_idx, (start, end) in enumerate(offsets):
        # Token trung tâm tại (start + end) / 2
        mid = (start + end) // 2
        # Match scope nhỏ nhất chứa token (nested handling đơn giản)
        best_scope_idx = -1
        best_size = float("inf")
        for s_idx, (s_start, s_end, s_type) in enumerate(scopes):
            if s_start <= mid < s_end:
                size = s_end - s_start
                if size < best_size:
                    best_size = size
                    best_scope_idx = s_idx
        scope_ids[tok_idx] = best_scope_idx

    return scope_ids


def _approximate_scope_map(code, tokenizer, scopes):
    """Fallback khi offset_mapping không có."""
    # Tokenize toàn bộ code, không lấy offset. Phân bổ uniform trong scope.
    ids = tokenizer(code, add_special_tokens=False).input_ids
    return torch.full((len(ids),), -1, dtype=torch.long)


# =====================================================================
# SCOPE-AWARE CENTROID INITIALIZATION
# =====================================================================

def compute_scope_centroids(
    keys: torch.Tensor,
    scope_ids: torch.Tensor,
    num_centroids: int,
) -> torch.Tensor:
    """
    Tạo initial centroids dựa trên scope của code.

    Mỗi scope contribute centroids tỷ lệ với kích thước (số token).
    Centroid = mean của key vectors trong scope.

    Args:
        keys: [num_heads, seq_len, head_dim] tensor cho 1 layer
        scope_ids: [seq_len] long tensor, scope ID cho mỗi token
        num_centroids: tổng số centroids cần

    Returns:
        centroids: [num_heads, num_centroids, head_dim] tensor
    """
    H, S, D = keys.shape

    # Lấy unique scopes (ignore -1)
    unique_scopes = torch.unique(scope_ids[scope_ids >= 0])
    num_scopes = len(unique_scopes)

    if num_scopes == 0:
        # Không có scope -> fallback random
        idx = torch.randperm(S)[:num_centroids]
        return keys[:, idx, :]

    # Phân bổ centroids theo size của scope
    scope_sizes = torch.zeros(num_scopes)
    for i, s in enumerate(unique_scopes):
        scope_sizes[i] = (scope_ids == s).sum().item()

    # Mỗi scope: số centroid = ceil(centroid_total * size / total_size)
    weights = scope_sizes / scope_sizes.sum()
    centroids_per_scope = (weights * num_centroids).long()
    # Đảm bảo mỗi scope có ít nhất 1
    centroids_per_scope = torch.clamp(centroids_per_scope, min=1)

    # Trim/pad để sum = num_centroids
    diff = num_centroids - centroids_per_scope.sum().item()
    if diff > 0:
        top = torch.argsort(scope_sizes, descending=True)[:diff]
        centroids_per_scope[top] += 1
    elif diff < 0:
        bot = torch.argsort(scope_sizes, descending=False)
        i = 0
        while diff < 0 and i < num_scopes:
            if centroids_per_scope[bot[i]] > 1:
                centroids_per_scope[bot[i]] -= 1
                diff += 1
            i += 1

    # Build centroids
    centroids_list = []
    for i, s in enumerate(unique_scopes):
        mask = (scope_ids == s)
        scope_keys = keys[:, mask, :]  # [H, scope_size, D]
        n_c = centroids_per_scope[i].item()

        if scope_keys.shape[1] < n_c:
            # Scope quá nhỏ - dùng tất cả token làm centroid
            centroids_list.append(scope_keys)
        else:
            # Subsample đều trong scope - đây là "init", K-means sẽ refine
            idx = torch.linspace(0, scope_keys.shape[1] - 1, n_c).long()
            centroids_list.append(scope_keys[:, idx, :])

    centroids = torch.cat(centroids_list, dim=1)  # [H, num_centroids, D]

    # Trim nếu vượt do scope quá nhỏ
    if centroids.shape[1] > num_centroids:
        centroids = centroids[:, :num_centroids, :]
    elif centroids.shape[1] < num_centroids:
        # Pad bằng random từ keys
        pad_count = num_centroids - centroids.shape[1]
        rand_idx = torch.randperm(S)[:pad_count]
        pad = keys[:, rand_idx, :]
        centroids = torch.cat([centroids, pad], dim=1)

    return centroids


# =====================================================================
# TOKEN-TYPE WEIGHTED K-MEANS
# =====================================================================

def compute_token_type_weights(
    code: str,
    tokenizer,
    boost_identifier: float = 1.5,
    boost_literal: float = 1.2,
    suppress_punctuation: float = 0.5,
) -> torch.Tensor:
    """
    Tính weight per-token dựa trên loại token (identifier, literal, punctuation).

    Idea: identifier (variable, function name) thường carry nhiều semantic
    information hơn punctuation. Khi K-means, weight cao hơn → token này
    "kéo" centroid mạnh hơn → cluster tốt hơn cho retrieval.

    Args:
        code: source code
        tokenizer: LLM tokenizer
        boost_identifier: hệ số nhân cho identifier-like token
        boost_literal: hệ số nhân cho literal (số, string)
        suppress_punctuation: hệ số nhân cho ký tự đơn (giảm influence)

    Returns:
        weights: [seq_len] float tensor
    """
    import re
    ids = tokenizer(code, add_special_tokens=False).input_ids
    weights = torch.ones(len(ids))

    for i, tok_id in enumerate(ids):
        tok_str = tokenizer.decode([tok_id]).strip()
        if len(tok_str) == 0:
            continue
        # Identifier-like: chữ + có thể underscore/digit
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{1,}$", tok_str):
            weights[i] = boost_identifier
        # Literal số
        elif re.match(r"^[0-9.]+$", tok_str):
            weights[i] = boost_literal
        # Punctuation đơn lẻ
        elif len(tok_str) == 1 and not tok_str.isalnum():
            weights[i] = suppress_punctuation

    return weights


# =====================================================================
# WEIGHTED K-MEANS
# =====================================================================

def weighted_kmeans(
    keys: torch.Tensor,
    weights: torch.Tensor,
    num_centroids: int,
    initial_centroids: Optional[torch.Tensor] = None,
    num_iter: int = 10,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means với weighted samples.

    Args:
        keys: [num_heads, seq_len, head_dim]
        weights: [seq_len] - weight per token
        num_centroids: K
        initial_centroids: [num_heads, num_centroids, head_dim] optional
        num_iter: số iteration

    Returns:
        centroids: [num_heads, num_centroids, head_dim]
        labels: [num_heads, seq_len] long - cluster assignment cho mỗi token
    """
    H, S, D = keys.shape
    keys = keys.to(device)
    weights = weights.to(device)  # [S]

    # Init centroids
    if initial_centroids is None:
        idx = torch.randperm(S, device=device)[:num_centroids]
        centroids = keys[:, idx, :].clone()  # [H, K, D]
    else:
        centroids = initial_centroids.to(device).clone()

    for it in range(num_iter):
        # Compute distance: [H, S, K]
        # ||k - c||^2 = ||k||^2 - 2 k.c + ||c||^2
        # Để đơn giản, dùng direct distance
        dist = torch.cdist(keys, centroids)  # [H, S, K]
        labels = dist.argmin(dim=-1)  # [H, S]

        # Update centroids: weighted mean của các point trong cluster
        new_centroids = torch.zeros_like(centroids)
        for h in range(H):
            for k in range(num_centroids):
                mask = (labels[h] == k)
                if mask.sum() == 0:
                    # Empty cluster - reinit từ farthest point
                    new_centroids[h, k] = keys[h, dist[h].max(dim=0).indices[k], :]
                else:
                    w = weights[mask].unsqueeze(-1)  # [mask_size, 1]
                    k_vecs = keys[h, mask, :]  # [mask_size, D]
                    new_centroids[h, k] = (w * k_vecs).sum(dim=0) / (w.sum() + 1e-9)
        centroids = new_centroids

    return centroids, labels
