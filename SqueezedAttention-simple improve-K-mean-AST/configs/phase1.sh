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

# --- Model chinh (quyet dinh D1, chot 15/8; xem lai 17/8) ---
#
# MAC DINH LA BAN BASE, KHONG PHAI INSTRUCT. Do 17/8 tren 20 mau LCC:
#   qwen2.5-coder-7b-instruct  ->  All-KV 17.60   (prediction gan nhu rong)
# Instruct duoc huan luyen trong khung ChatML; LongBench cho lcc/repobench-p thi
# co y BO chat template (build_chat bo qua hai task nay), nen model roi vao che do
# tro ly, mo mot khoi markdown roi phat token ket thuc som. Khong phai loi pipeline:
# cung duong ong do LongChat ra 54.83, va day la nhanh All-KV khong dung centroid nao.
#
# LCC/RepoBench-P la dien dong code tiep theo trong ngu canh repo -> base model la
# dung cong cu. Doi lai: RepoPreFixQA cua Phase 6 la task QA, cho do can instruct.
# Neu ket cuc dung hai model cho hai loai task thi PHAI ghi ro trong paper.
#
# Ghi de bang bien moi truong hoac co --model cua phase1_gate.sh:
#   SQA_MODEL_CODE=qwen2.5-coder-7b-instruct bash scripts/phase1_gate.sh
export SQA_MODEL_CODE="${SQA_MODEL_CODE:-qwen2.5-coder-7b}"

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
