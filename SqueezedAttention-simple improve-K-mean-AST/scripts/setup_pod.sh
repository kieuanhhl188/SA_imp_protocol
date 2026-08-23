#!/bin/bash
# ====================================================================
# setup_pod.sh — dựng môi trường trên RunPod A100 80GB SXM
#
#   bash scripts/setup_pod.sh
#
# Bản này viết lại sau khi dựng thật ngày 16/8/2026. Toàn bộ phiên bản dưới đây là tổ hợp
# ĐÃ CHẠY ĐƯỢC, không phải phỏng đoán. Đừng đổi lung tung — mỗi con số đều có lý do.
#
# NĂM CÁI BẪY đã gặp, script này chặn sẵn cả năm:
#
#  1. Python của image quá mới (3.12). torch 2.3.1 khai báo `triton==2.3.1 ; python<3.12`
#     nên trên 3.12 nó KHÔNG kéo triton, để nguyên triton 3.4 của image. Mà kernel dùng
#     `tl.math.exp2` (API Triton 2.x). Thêm nữa RAPIDS bản cp312 đòi numpy>=2 còn torch 2.3
#     cần numpy<2 — mâu thuẫn không gỡ được. => Dựng venv Python 3.10 riêng.
#
#  2. `uv venv` KHÔNG cài pip vào venv, nên gõ `pip` sẽ rơi vào pip hệ thống và cài nhầm
#     vào Python 3.12. => Dùng `python -m pip` ở mọi nơi.
#
#  3. pip do `ensurepip` cấp là bản 23.0.1, có bug chuẩn hoá tên package: gặp `Jinja2` thì
#     báo "inconsistent Name: expected 'jinja2'", bỏ wheel, quay sang build sdist rồi chết
#     vì thiếu flit_core. => Nâng pip TRƯỚC khi cài gì khác.
#
#  4. `pip install flash-attn` mặc định BUILD TỪ NGUỒN — mất 2,5 giờ (~$4 tiền GPU) vì
#     không có wheel khớp. Wheel dựng sẵn chỉ có trên GitHub Releases. => Cài thẳng URL wheel.
#
#  5. `datasets` đời mới kéo pyarrow>=21, phá cudf 24.6 (`pyarrow.lib has no attribute
#     PyExtensionType`) và kéo theo cuML chết. => Ghim datasets 2.20 + pyarrow 16.1.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv310}"
PYVER="${PYVER:-3.10}"

# Phiên bản đã kiểm chứng chạy được (16/8/2026)
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

echo "=================================================================="
echo "  Dựng môi trường Squeezed Attention"
echo "  Repo:      $REPO_ROOT"
echo "  Venv:      $VENV  (Python $PYVER)"
echo "=================================================================="

# ---------- 0. Phần cứng ----------
echo ""
echo ">>> [0] Phần cứng"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$WORKSPACE" | tail -1
echo "    LƯU Ý: df hiện dung lượng cả cụm MooseFS, KHÔNG phải hạn mức volume của bạn."
echo "           Xem hạn mức thật trên dashboard RunPod."

# ---------- 1. Venv Python 3.10 ----------
echo ""
echo ">>> [1] Venv Python $PYVER (bẫy #1)"
if [ ! -x "$VENV/bin/python" ]; then
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "    cài uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$HOME/.local/bin/uv"
  [ -x "$UV" ] || UV="$(command -v uv)"
  "$UV" venv --python "$PYVER" "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
echo "    $($PY -V)"

# ---------- 2. pip trong venv ----------
# Bẫy #2 + #3: uv venv không có pip; ensurepip cấp pip 23.0.1 quá cũ.
echo ""
echo ">>> [2] pip trong venv (bẫy #2, #3)"
$PY -m pip --version >/dev/null 2>&1 || $PY -m ensurepip --upgrade
$PY -m pip install -q -U pip setuptools wheel
echo "    $($PY -m pip --version)"

# ---------- 3. Thư mục + biến môi trường ----------
echo ""
echo ">>> [3] Thư mục trên volume"
mkdir -p "$WORKSPACE"/{hf,fixed-prompt-clusters,phase0_results,phase1_data}
cat > "$WORKSPACE/env.sh" <<EOF
# source /workspace/env.sh trước mỗi phiên làm việc
source $VENV/bin/activate
export HF_HOME=$WORKSPACE/hf
export SQA_CLUSTER_DIR=$WORKSPACE/fixed-prompt-clusters
export SQA_RESULT_DIR=$WORKSPACE/phase0_results
export SQA_PHASE1_DIR=$WORKSPACE/phase1_data
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# LongBench có custom loader; không đặt biến này thì job dài sẽ TREO ở prompt [y/N]
export HF_DATASETS_TRUST_REMOTE_CODE=1
EOF
echo "    đã ghi $WORKSPACE/env.sh"

# ---------- 4. Torch (kéo theo triton 2.3.1) ----------
echo ""
echo ">>> [4] PyTorch $TORCH_VERSION + Triton"
$PY -m pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu121
$PY - <<'PY'
import torch, triton
print(f"    torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"available={torch.cuda.is_available()} | cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}")
print(f"    triton {triton.__version__}")
assert triton.__version__.startswith("2."), (
    "Triton phải là 2.x — kernel dùng tl.math.exp2 (API 2.x). "
    "Ra 3.x nghĩa là venv không phải Python 3.10.")
PY

# ---------- 5. numpy + RAPIDS ----------
# RAPIDS 24.6 cùng thời CUDA 12.1 nên KHÔNG kéo đè nvidia-*-cu12 của torch, và dùng numpy 1.x.
echo ""
echo ">>> [5] numpy + cuML + CuPy"
$PY -m pip install -q "numpy<2" "cuml-cu12==24.6.*" "cupy-cuda12x<14" \
    --extra-index-url=https://pypi.nvidia.com

# ---------- 6. flash-attn từ wheel ----------
echo ""
echo ">>> [6] flash-attn (bẫy #4 — build từ nguồn mất 2,5 giờ)"
if $PY -c "import flash_attn" 2>/dev/null; then
  echo "    đã có: $($PY -c 'import flash_attn;print(flash_attn.__version__)')"
else
  CODE="$(curl -sIL -o /dev/null -w '%{http_code}' "$FLASH_ATTN_WHL")"
  if [ "$CODE" = "200" ]; then
    $PY -m pip install -q "$FLASH_ATTN_WHL"
  else
    echo "    [!] wheel không tồn tại (HTTP $CODE)."
    echo "        Tìm wheel khớp tại https://github.com/Dao-AILab/flash-attention/releases"
    echo "        Phải khớp ĐỒNG THỜI: torch minor, cxx11abi (in ở bước [4]), python, cuda."
    echo "        Rồi chạy lại với: FLASH_ATTN_WHL=<url> bash scripts/setup_pod.sh"
    exit 1
  fi
fi

# ---------- 7. Dependency còn lại ----------
# Bẫy #5: datasets mới kéo pyarrow>=21 -> phá cudf 24.6 -> phá cuML.
echo ""
echo ">>> [7] Dependency còn lại (bẫy #5 — ghim datasets + pyarrow)"
# pytest: KHÔNG phải để chạy test — squeezedattention/kernels.py có `import pytest` ở dòng 3
# (sót lại từ repo gốc), mà modeling_llama.py import kernels ở top-level. Thiếu nó là
# `from transformers import LlamaForCausalLM` cũng chết.
$PY -m pip install -q "datasets==2.20.0" "pyarrow>=16.1,<16.2" \
    scikit-learn hf_transfer pytest tqdm rouge jieba fuzzywuzzy python-Levenshtein \
    einops sentencepiece protobuf accelerate matplotlib pandas \
    tree-sitter tree-sitter-python tree-sitter-java tree-sitter-javascript \
    tree-sitter-c-sharp tree-sitter-typescript

# ---------- 8. transformers fork — PHẢI cài SAU CÙNG ----------
echo ""
echo ">>> [8] transformers fork"
$PY -m pip install -q -e ./transformers
$PY -m pip install -q -e .

$PY - <<PY
import sys, transformers, os
# realpath cả hai vế: repo hay được truy cập qua symlink (vd /workspace/sa), mà
# transformers.__file__ trả về đường dẫn ĐÃ giải symlink -> so chuỗi trực tiếp sẽ báo sai.
root = os.path.realpath("$REPO_ROOT")
p = os.path.realpath(transformers.__file__)
print(f"    transformers {transformers.__version__}")
print(f"    {p}")
if not (root in p and transformers.__version__.startswith("4.40")):
    print("    [!!] SAI: không phải bản fork trong repo.")
    print("         KHÔNG chạy pip install -r LongBench/requirements.txt (ghim 4.31.0).")
    sys.exit(1)
PY

# ---------- 9. Kiểm tra tổng thể ----------
echo ""
echo ">>> [9] Kiểm tra môi trường"
$PY - <<'PY'
import numpy, torch, triton, cupy, flash_attn, transformers, datasets, pyarrow, tree_sitter
from cuml.cluster import KMeans
# Gói ngôn ngữ phải import ĐÍCH DANH: `import tree_sitter` chạy được không có nghĩa là
# parser C#/Java có mặt. Thiếu một gói thì Phase 2 chết giữa run (LCC mẫu 0 là C#), lúc đó
# đã trả tiền GPU rồi. Kiểm ở đây để hỏng thì hỏng lúc setup.
import tree_sitter_python, tree_sitter_java, tree_sitter_c_sharp   # noqa: F401
import tree_sitter_typescript, tree_sitter_javascript             # noqa: F401
print("    tree-sitter: python + java + c_sharp + ts + js OK")
print(f"    numpy {numpy.__version__} | pyarrow {pyarrow.__version__} | datasets {datasets.__version__}")
print(f"    torch {torch.__version__} | triton {triton.__version__} | flash_attn {flash_attn.__version__}")
print(f"    cupy {cupy.__version__} | cuml OK | transformers {transformers.__version__}")
PY

echo ""
echo ">>> [10] Kernel Triton chạy thật"
# Phải chạy từ FILE: triton JIT đọc mã nguồn qua inspect, `python -c` không có file nên lỗi
# "could not get source code" — đó là lỗi phương pháp, không phải lỗi triton.
printf 'import torch, triton, triton.language as tl\n@triton.jit\ndef k(x, o, N: tl.constexpr):\n    i = tl.arange(0, N)\n    tl.store(o + i, tl.math.exp2(tl.load(x + i)))\na = torch.zeros(16, device="cuda"); b = torch.empty_like(a)\nk[(1,)](a, b, 16)\nprint("    tl.math.exp2 OK", b[:3].tolist())\n' > /tmp/_tt.py
$PY /tmp/_tt.py

echo ""
echo ">>> [11] Đường clustering thật (cupy + cuML + dlpack + torch)"
$PY -c "import torch; from squeezedattention.clustering import run_clustering; c,l = run_clustering({0: torch.randn(1,4,300,128).cuda()}, 10, observation_window=100, device='cuda:0'); print('    run_clustering OK', tuple(c[0].shape), tuple(l[0].shape))"

echo ""
echo ">>> [12] Test CPU"
$PY scripts/test_struct_clustering.py > /tmp/t1.log 2>&1 && echo "    struct_clustering: PASS" || { echo "    struct_clustering: FAIL"; tail -20 /tmp/t1.log; }
$PY scripts/test_gqa_port.py         > /tmp/t2.log 2>&1 && echo "    gqa_port:          PASS" || { echo "    gqa_port:          FAIL"; tail -20 /tmp/t2.log; }
$PY scripts/prepare_code_data.py --self_test > /tmp/t3.log 2>&1 && echo "    prepare_code_data: PASS" || { echo "    prepare_code_data: FAIL"; tail -20 /tmp/t3.log; }

echo ""
echo ">>> [13] Ghi lại môi trường"
$PY scripts/record_env.py --out "$WORKSPACE/phase0_results/env_record.json" --note "setup_pod"
$PY -m pip freeze > "$WORKSPACE/working_env.txt"
echo "    đã ghi $WORKSPACE/working_env.txt"

echo ""
echo "=================================================================="
echo "  Xong. Mỗi phiên mới:  source $WORKSPACE/env.sh"
echo ""
echo "  Bước tiếp theo, KHÔNG đảo thứ tự:"
echo "    1. python scripts/prepare_code_data.py longchat-v1.5-7b-32k --dataset repobench-p --limit 3"
echo "    2. python offline_clustering.py longchat-v1.5-7b-32k --dataset repobench-p \\"
echo "         --percent_clusters 5 --output_path /workspace/smoke/   # đo s/it rồi Ctrl-C"
echo "    3. du -sh /workspace/smoke/                                  # đo MB/sample"
echo "    4. bash scripts/phase0_gate.sh                               # gate đầy đủ"
echo "=================================================================="
