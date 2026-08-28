#!/bin/bash
# ====================================================================
# run_phase2_phase5_lcc.sh — mot phien pod: Phase 2 (3 nhanh) roi Phase 5 (C2) tren LCC.
#
#   bash scripts/run_phase2_phase5_lcc.sh
#
# ---- CHOT 28/8: model = longchat-v1.5-7b-32k, LCC-only, KHONG --force_chat ----
# Doi tu qwen2.5-coder-7b-instruct tro lai LongChat, khop pham vi da thu hep o Phase 0
# (configs/phase0.sh) va Phase 1 (configs/phase1.sh). Ly do:
#   * LongChat di thang duong modeling_llama goc — dung model cua chinh bai goc (Table 2),
#     khong phai ban PORT sang modeling_qwen2 (GQA).
#   * LongChat la MHA: num_key_value_heads = num_heads = 32 -> KHONG co GQA. Xu ly
#     per-head kieu QUEST Appendix G la N/A.
#   * LongChat khong phai ban Instruct va LCC nam trong NO_CHAT_TEMPLATE -> KHONG
#     --force_chat. Phai TRUNG voi co da dung khi sinh du lieu Phase 1.4 va o pred.py.
#
# ---- CANH BAO DUNG LUONG: LongChat ton gap ~8 lan Qwen ----
# Centroid luu theo so KV head. Qwen co 4, LongChat co 32. Luot Qwen 22/8: sa 5,8 GB ·
# hard_boundary 5,8 GB · struct_hierarchy 7,8 GB (500 mau). Ngoai suy cho LongChat:
# ~46 / ~46 / ~62 GB -> ba nhanh ~150 GB, sat tran volume 200 GB. Kiem `du -sh` truoc,
# va cham chay `check_cluster_integrity.py` sau moi nhanh (MooseFS cat cut im lang).
# Muon nhe hon: LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh
#
# ---- THU TU CO CHU DICH ----
# chay nhanh `hard_boundary` truoc, roi SMOKE Phase 5 tren 3 mau ngay lap tuc. Ly do:
# phase5_recall.py chua tung chay tren LongChat. Neu no hong thi hong som.
#
# ---- Uoc tinh thoi gian ----
# CHUA do tren LongChat. Moc tho: offline_clustering.py o Phase 0 mat ~6h15 cho LCC 500
# mau (MHA, moi seed dung chung threshold). struct clustering lam k-means torch per-unit
# -> uoc moi nhanh ~4-8h, ba nhanh co the tron mot ngay. SMOKE 3 mau truoc de biet
# s/mau thuc te roi hay chay full.
#
# Chay lai duoc: offline_clustering_struct.py bo qua mau da co du file, nen pod restart
# giua chung thi goi lai dung lenh nay.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Cau hinh chot (SQA_MODEL_CODE, SQA_FORCE_CHAT, SQA_PHASE1_DIR) — cung nguon voi Phase 1.
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/phase1.sh"
# shellcheck disable=SC1091
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"
P2="${P2_DIR:-/workspace/p2-longchat}"
PHASE1_DIR="${SQA_PHASE1_DIR:-$REPO_ROOT/phase1_data}"
LIMIT_P5="${LIMIT_P5:-100}"
LIMIT_P2="${LIMIT_P2:--1}"

# --force_chat chi them vao lenh khi SQA_FORCE_CHAT=1. LongChat/LCC = 0.
FORCE_CHAT=()
[ "${SQA_FORCE_CHAT:-0}" = "1" ] && FORCE_CHAT=(--force_chat)

COMMON=("${FORCE_CHAT[@]}" --dataset lcc --level function --level_l1 class
        --percent_clusters 5 --observation_window 100
        --phase1_dir "$PHASE1_DIR")
[ "$LIMIT_P2" != "-1" ] && COMMON+=(--limit "$LIMIT_P2")

echo "=================================================================="
echo "  Phase 2 + Phase 5 — LCC"
echo "  Model     : $MODEL   (force_chat=${SQA_FORCE_CHAT:-0})"
echo "  Phase 1.4 : $PHASE1_DIR/$MODEL"
echo "  Output    : $P2"
echo "  Bat dau   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ ! -f "$PHASE1_DIR/$MODEL/lcc_meta.jsonl" ]; then
  echo "[ERROR] thieu du lieu Phase 1.4 cho $MODEL tai $PHASE1_DIR/$MODEL/"
  echo "        Sinh truoc (CPU, ~1-2 phut): bash scripts/phase1_gate.sh --data-only"
  exit 1
fi

echo ""
echo "########## nhanh 1/3: hard_boundary ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method hard_boundary --output_path "$P2/hard_boundary/lcc/"

echo ""
echo "########## SMOKE Phase 5 (3 mau) — bat loi som ##########"
python phase5_recall.py "$MODEL" "${FORCE_CHAT[@]}" --dataset lcc \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --sparsity 70 --limit 3 --out /workspace/phase5_smoke.json

echo ""
echo "########## nhanh 2/3: struct_hierarchy ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method struct_hierarchy --output_path "$P2/struct_hierarchy/lcc/"

echo ""
echo "########## nhanh 3/3: sa (doi chung) ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method sa --output_path "$P2/sa/lcc/"

echo ""
echo "########## Kiem toan ven file ##########"
for B in sa hard_boundary struct_hierarchy; do
  echo "-- $B"
  python scripts/check_cluster_integrity.py "$P2/$B/lcc" | tail -3
done

echo ""
echo "########## Kiem bat bien Phase 2 ##########"
python scripts/check_phase2_invariants.py \
    --cluster_dir "sa=$P2/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --model "$MODEL" \
    --phase1_dir "$PHASE1_DIR/$MODEL" --dataset lcc \
    2>&1 | tee /workspace/p2_invariants_longchat.log | tail -5 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget ##########"
python phase5_recall.py "$MODEL" "${FORCE_CHAT[@]}" --dataset lcc \
    --cluster_dir "sa=$P2/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --sparsity 70 80 90 --limit "$LIMIT_P5" --out /workspace/phase5_lcc.json

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ket qua: /workspace/phase5_lcc.json"
echo "           /workspace/p2_invariants_longchat.log"
echo "=================================================================="
