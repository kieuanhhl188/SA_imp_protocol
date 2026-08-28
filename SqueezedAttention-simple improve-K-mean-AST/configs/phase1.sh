#!/bin/bash
# ====================================================================
# configs/phase1.sh — CAU HINH PHASE 1 (model code chinh)
#
# Source configs/phase0.sh truoc roi CHI ghi de nhung gi thuc su khac,
# de moi tham so clustering/threshold giu nguyen y het gate Phase 0.
# Doi mot tham so o day la doi cho ca Phase 5/6 -> phai co ly do ghi vao
# EXPERIMENT_LOG.md.
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/phase0.sh"

# --- Model chinh ---
#
# CHOT 28/8: quay ve LongChat-7B-v1.5-32K + LCC-only, khop pham vi da thu hep o
# Phase 0 (docs/PHASE0.md §8) va dung model cua chinh bai goc (Table 2).
#
# Vi sao doi lai tu Qwen2.5-Coder:
#   * Qwen la ban PORT sang modeling_qwen2 (GQA) — rui ro nam o dung cho Squeezed
#     Attention da va. LongChat chay thang duong modeling_llama goc, da duoc Phase 0
#     xac nhan (repro_lcc.sh: All-KV 54,83 · Sq-70% 56,08).
#   * LongChat la MHA (num_key_value_heads = num_heads = 32) -> KHONG co GQA. Toan bo
#     xu ly per-head kieu QUEST Appendix G khong ap dung; buoc 1.6 la N/A.
#   * LongChat khong phai ban Instruct va LongBench co y bo chat template cho lcc ->
#     KHONG can --force_chat. (Day la ca dong cot lam Qwen phuc tap.)
#   * "128K" / YaRN / D7 khong lien quan: LongChat native 32K + linear RoPE factor 8,
#     fork da chap nhan san.
#
# HE QUA THUC TE: phan RIENG cua Phase 1 chi con la du lieu 1.4 (offset token) + gate
# du lieu 5 buoc — deu chay CPU. Phan do accuracy (Sq-70% vs All-KV tren LongChat/LCC)
# TRUNG voi Phase 0; khong chay lai o day, xem docs/POD_RUNBOOK.md §4.
#
# Ghi de bang bien moi truong hoac co --model cua phase1_gate.sh:
#   SQA_MODEL_CODE=qwen2.5-coder-7b-instruct SQA_FORCE_CHAT=1 bash scripts/phase1_gate.sh
export SQA_MODEL_CODE="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"

# LongChat + lcc: khong dung chat template. Phai TRUNG voi co dung o pred.py va
# offline_clustering.py, neu khong shared_prefix_length lech.
export SQA_FORCE_CHAT="${SQA_FORCE_CHAT:-0}"

# --- So sample cho smoke test cua gate Phase 1 ---
# Muc dich cua gate la xac nhan du lieu 1.4 + duong nap centroid, KHONG phai do
# accuracy. 20 mau du re. Dat 0 de chay ca dataset.
export SQA_PHASE1_LIMIT="${SQA_PHASE1_LIMIT:-20}"

# --- Thu muc du lieu Phase 1.4 (offset token) ---
export SQA_PHASE1_DIR="${SQA_PHASE1_DIR:-$SQA_REPO_ROOT/phase1_data}"

# --- Task dung cho gate Phase 1 ---
# Chi LCC, cung ly do da chot o Phase 0: RepoBench-P dai gap ~3.7 lan, chi phi
# clustering scale bac hai theo do dai, ma khong tra loi them cau hoi nao.
export SQA_PHASE1_TASK="${SQA_PHASE1_TASK:-lcc}"

# --- Dung sai gate Phase 1 ---
# Tieu chi noi tai: Sq-70% khong duoc thap hon All-KV qua muc nay. Giu ±2.0 nhu
# Phase 0. (Phase 0 da chot khong con gate theo Table 2 — xem docs/PHASE0.md §8.)
export SQA_PHASE1_TOLERANCE="${SQA_PHASE1_TOLERANCE:-2.0}"
