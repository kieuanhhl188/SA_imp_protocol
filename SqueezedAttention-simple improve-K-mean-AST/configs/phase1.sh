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

# --- Model chinh (quyet dinh D1, chot 15/8) ---
export SQA_MODEL_CODE="qwen2.5-coder-7b-instruct"

# --- So sample cho smoke test cua gate Phase 1 ---
# Muc dich cua gate la xac nhan duong GQA nap/tra dung centroid, KHONG phai do
# accuracy. 20 mau du de mot loi tra nham nhom centroid lo ra vai diem, ma van re.
# Dat 0 de chay ca dataset.
export SQA_PHASE1_LIMIT="${SQA_PHASE1_LIMIT:-20}"

# --- Thu muc du lieu Phase 1.4 (offset token) ---
export SQA_PHASE1_DIR="${SQA_PHASE1_DIR:-$SQA_REPO_ROOT/phase1_data}"

# --- Task dung cho gate Phase 1 ---
# Chi LCC, cung ly do da chot o Phase 0: RepoBench-P dai gap ~3.7 lan, chi phi
# clustering scale bac hai theo do dai, ma khong tra loi them cau hoi nao.
export SQA_PHASE1_TASK="${SQA_PHASE1_TASK:-lcc}"

# --- Dung sai gate Phase 1 ---
# Table 2 khong co Qwen -> khong co moc ngoai. Tieu chi la noi tai:
# Sq-70% khong duoc thap hon All-KV qua muc nay. Giu ±2.0 nhu Phase 0.
export SQA_PHASE1_TOLERANCE="${SQA_PHASE1_TOLERANCE:-2.0}"
