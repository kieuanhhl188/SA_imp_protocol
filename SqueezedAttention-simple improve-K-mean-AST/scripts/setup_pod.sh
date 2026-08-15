#!/bin/bash
# ====================================================================
# setup_pod.sh — dựng môi trường trên RunPod A100 80GB PCIe
#
# Chạy MỘT LẦN sau khi tạo pod. Idempotent: chạy lại không hỏng gì.
#
#   bash scripts/setup_pod.sh
#
# Thứ tự cài KHÔNG được đảo. flash-attn cần torch có sẵn lúc build; transformers fork
# phải cài SAU cùng để không bị gói khác kéo bản PyPI đè lên.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
PYVER="${PYVER:-3.10}"

echo "=================================================================="
echo "  Dựng môi trường Squeezed Attention"
echo "  Repo:      $REPO_ROOT"
echo "  Workspace: $WORKSPACE"
echo "=================================================================="

# ---------- 0. Kiểm tra phần cứng và CUDA ----------
echo ""
echo ">>> [0] Phần cứng"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
CUDA_MAJOR="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo '')"
if [ -z "$CUDA_MAJOR" ]; then
  CUDA_MAJOR="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 12)"
fi
echo "    CUDA major = $CUDA_MAJOR"
if [ "$CUDA_MAJOR" != "12" ]; then
  echo "    [!] Script này viết cho CUDA 12. Với CUDA 11 phải đổi cuml-cu12 -> cuml-cu11,"
  echo "        cupy-cuda12x -> cupy-cuda11x, và chọn wheel torch/flash-attn tương ứng."
fi

# ---------- 1. Thư mục trên volume ----------
echo ""
echo ">>> [1] Thư mục trên volume (dữ liệu phải nằm ngoài container disk)"
mkdir -p "$WORKSPACE"/{hf,fixed-prompt-clusters,phase0_results,phase1_data}
cat > "$WORKSPACE/env.sh" <<EOF
# source /workspace/env.sh trước mỗi phiên làm việc
export HF_HOME=$WORKSPACE/hf
export SQA_CLUSTER_DIR=$WORKSPACE/fixed-prompt-clusters
export SQA_RESULT_DIR=$WORKSPACE/phase0_results
export SQA_PHASE1_DIR=$WORKSPACE/phase1_data
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
EOF
# shellcheck disable=SC1090
source "$WORKSPACE/env.sh"
echo "    đã ghi $WORKSPACE/env.sh"
df -h "$WORKSPACE" | tail -1

# ---------- 2. Torch ----------
# GHIM 2.3.1. Lý do là TRITON, không phải torch:
#   squeezedattention/kernels.py dùng `tl.math.exp2` (9 chỗ) — API Triton 2.x.
#   Triton 3.x đã chuyển tl.math.* sang tl.* và bỏ dần hàm cũ.
#   torch 2.3.x -> Triton 2.3 (đúng)     torch 2.8 -> Triton 3.3+ (hỏng kernel)
# Ngoài ra torch 2.3 cùng thời với transformers 4.40.0.dev0 (đều khoảng 3-4/2024).
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
echo ""
echo ">>> [2] PyTorch $TORCH_VERSION (CUDA 12.1)"
CUR="$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
if [[ "$CUR" == "$TORCH_VERSION"* ]]; then
  echo "    đã đúng bản: $CUR"
else
  echo "    đang có '$CUR' -> cài $TORCH_VERSION"
  pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu121
fi

python - <<'PY'
import torch
print(f"    torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"available={torch.cuda.is_available()} | cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}")
try:
    import triton
    v = triton.__version__
    major = int(v.split('.')[0])
    print(f"    triton {v}")
    if major >= 3:
        print("    [!!] Triton 3.x: kernel dùng tl.math.exp2 (API 2.x) rất có thể sẽ lỗi")
        print("         khi biên dịch. Hạ torch: TORCH_VERSION=2.3.1 bash scripts/setup_pod.sh")
except ImportError:
    print("    [!] chưa có triton — lẽ ra torch phải kéo theo")
PY

# ---------- 3. flash-attn ----------
# Build từ nguồn mất 30-60 phút. Ưu tiên wheel dựng sẵn; tên wheel phải khớp ĐỒNG THỜI
# torch minor, CUDA major, phiên bản python, và cờ cxx11abi in ở trên.
echo ""
echo ">>> [3] flash-attn"
if python -c "import flash_attn" 2>/dev/null; then
  echo "    đã có: $(python -c 'import flash_attn;print(flash_attn.__version__)')"
else
  echo "    thử wheel dựng sẵn; nếu trượt sẽ build từ nguồn (chậm)"
  pip install -q flash-attn --no-build-isolation || {
    echo "    [!] cài flash-attn thất bại."
    echo "        Lấy wheel khớp tại https://github.com/Dao-AILab/flash-attention/releases"
    echo "        Tên wheel dạng: flash_attn-<ver>+cu12<torch>cxx11abi<TRUE|FALSE>-cp310-...whl"
    echo "        Chọn cxx11abi khớp giá trị in ở bước [2]."
    exit 1
  }
fi

# ---------- 4. RAPIDS ----------
# squeezedattention/clustering.py import cupy + cuml ở TOP-LEVEL -> thiếu là crash ngay khi
# import, không phải lúc chạy.
echo ""
echo ">>> [4] cuML + CuPy (RAPIDS)"
python -c "import cuml, cupy" 2>/dev/null && echo "    đã có" || \
  pip install -q cuml-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# ---------- 5. Dependency còn lại ----------
echo ""
echo ">>> [5] Dependency còn lại"
# Bỏ qua các dòng đã cài ở trên để pip không đổi phiên bản torch/flash-attn
grep -vE '^(torch|flash-attn|triton)\b' requirements.txt > /tmp/req_rest.txt
pip install -q -r /tmp/req_rest.txt

# ---------- 6. transformers fork — PHẢI cài SAU CÙNG ----------
echo ""
echo ">>> [6] transformers fork (bản đã patch Squeezed Attention)"
pip install -q -e ./transformers
pip install -q -e .

python - <<PY
import sys, transformers, os
root = os.path.abspath("$REPO_ROOT")
p = os.path.abspath(transformers.__file__)
ok = root in p and transformers.__version__.startswith("4.40")
print(f"    transformers {transformers.__version__}")
print(f"    {p}")
if not ok:
    print("    [!!] SAI: không phải bản fork trong repo.")
    print("         Nguyên nhân hay gặp: đã lỡ chạy pip install -r LongBench/requirements.txt")
    print("         (file đó ghim transformers==4.31.0 và ghi đè fork).")
    print("         Sửa: pip uninstall -y transformers && pip install -e ./transformers")
    sys.exit(1)
PY

# ---------- 7. tree-sitter cho Phase 2 ----------
echo ""
echo ">>> [7] tree-sitter (Phase 2)"
# KHÔNG dùng tree_sitter_languages: gói đó không build được trên nhiều môi trường.
# Bản mới dùng API tree_sitter + gói ngôn ngữ riêng.
pip install -q tree-sitter tree-sitter-python tree-sitter-java tree-sitter-javascript
python -c "
from tree_sitter import Language, Parser
import tree_sitter_python as tsp
Parser(Language(tsp.language())).parse(b'def f():\n    pass\n')
print('    tree-sitter OK')
"

# ---------- 8. Kiểm tra tổng thể ----------
echo ""
echo ">>> [8] Kiểm tra môi trường"
python scripts/record_env.py --out "$SQA_RESULT_DIR/env_record.json" --note "setup_pod"

echo ""
echo ">>> [9] Test chạy CPU (không cần GPU, xác nhận code không hỏng)"
python scripts/test_struct_clustering.py > /tmp/t1.log 2>&1 && echo "    struct_clustering: PASS" || { echo "    struct_clustering: FAIL"; tail -20 /tmp/t1.log; }
python scripts/test_gqa_port.py         > /tmp/t2.log 2>&1 && echo "    gqa_port:          PASS" || { echo "    gqa_port:          FAIL"; tail -20 /tmp/t2.log; }
python scripts/prepare_code_data.py --self_test > /tmp/t3.log 2>&1 && echo "    prepare_code_data: PASS" || { echo "    prepare_code_data: FAIL"; tail -20 /tmp/t3.log; }

echo ""
echo "=================================================================="
echo "  Xong. Mỗi phiên mới nhớ:  source $WORKSPACE/env.sh"
echo ""
echo "  Bước tiếp theo, KHÔNG đảo thứ tự:"
echo "    1. bash scripts/phase0_gate.sh                      # gate môi trường"
echo "    2. python scripts/prepare_code_data.py qwen2.5-coder-7b-instruct \\"
echo "         --dataset lcc --limit 3"
echo "    3. python offline_clustering.py qwen2.5-coder-7b-instruct \\"
echo "         --dataset lcc --percent_clusters 5 --output_path /tmp/smoke/"
echo "    4. python offline_clustering_struct.py qwen2.5-coder-7b-instruct \\"
echo "         --dataset lcc --method hard_boundary --limit 3 --output_path /tmp/smoke2/"
echo "=================================================================="
