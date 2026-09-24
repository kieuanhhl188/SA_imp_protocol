#!/bin/bash
# ====================================================================
# run_phase4_incremental.sh — Phase 4 / C3: t_incr vs t_full khi sua MOT class.
#
# Chay TUNG STAGE, doc ket qua roi moi sang stage sau:
#
#   bash scripts/run_phase4_incremental.sh smoke      # 3 mau, ~10 phut — BAT BUOC chay truoc
#   bash scripts/run_phase4_incremental.sh main       # 100 mau, sua class giua, 1 dong
#   bash scripts/run_phase4_incremental.sh sweep      # vi tri sua first/last x co sua 1/20 dong
#   bash scripts/run_phase4_incremental.sh sa         # 20 mau co SA (cuML) full + SA stale
#   bash scripts/run_phase4_incremental.sh synthetic  # chi clustering, 16K..128K, key ngau nhien
#   bash scripts/run_phase4_incremental.sh summary    # in lai bang tong hop moi file
#
# ---- DOC SMOKE THE NAO (dung lai neu co dong nao sai) ----
#   checks.L*_incr_vs_full_key_relmax  phai nho (~1e-2, sai so bf16). Lon (>0.1) nghia la
#       forward incremental (past_key_values + flash) SAI -> moi so fwd_incr vo nghia.
#   checks.L*_prefix_key_relmax        ~0: key truoc cho sua khong doi (causal).
#   checks.L*_suffix_label_agree       ~1.0: P2 khop hard_boundary full cung k_u.
#   units_local                        = 1 (chi class bi sua).
#
# Truoc khi chay tren pod: python scripts/test_incremental.py (CPU, 29 test).
# Chay lai duoc: moi dong jsonl co key, mau da co thi bo qua.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/phase1.sh"
# shellcheck disable=SC1091
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"
DATASET="${DATASET:-repobench-p}"
PHASE1_DIR="${SQA_PHASE1_DIR:-$REPO_ROOT/phase1_data}"
OUT_DIR="${P4_OUT:-/workspace/phase4_out}"
STAGE="${1:-}"
mkdir -p "$OUT_DIR"

run() {   # run <ten_file> <cac tham so them>
  local name="$1"; shift
  echo ""
  echo "########## $name — $(date '+%H:%M:%S') ##########"
  python phase4_incremental.py "$MODEL" --mode real --dataset "$DATASET" \
      --phase1_dir "$PHASE1_DIR" --percent_clusters 5 --observation_window 100 \
      --out "$OUT_DIR/$name.jsonl" "$@" 2>&1 | tee -a "$OUT_DIR/$name.log"
}

case "$STAGE" in
  smoke)
    python scripts/test_incremental.py | tail -1
    run smoke --limit 3 --edit_pos middle --edit_lines 1 --with_sa
    echo ">>> Doc checks trong $OUT_DIR/smoke.jsonl theo muc 'DOC SMOKE' o dau file nay."
    ;;
  main)
    run rb_mid_l1 --limit 100 --edit_pos middle --edit_lines 1
    ;;
  sweep)
    # vi tri sua quyet dinh fwd_incr (first: forward lai gan het; last: rat it)
    run rb_first_l1 --limit 100 --edit_pos first --edit_lines 1
    run rb_last_l1  --limit 100 --edit_pos last  --edit_lines 1
    # co sua: 20 dong ~ vai tram token
    run rb_mid_l20  --limit 100 --edit_pos middle --edit_lines 20
    ;;
  sa)
    # cuML chay 2 lan/mau (ban cu + ban moi) -> chi 20 mau
    run rb_mid_l1_sa --limit 20 --edit_pos middle --edit_lines 1 --with_sa
    ;;
  synthetic)
    python phase4_incremental.py --mode synthetic --lengths 16384 32768 65536 131072 \
        --percent_clusters 5 --syn_sa_heads 1 --out "$OUT_DIR/synthetic.jsonl" \
        2>&1 | tee -a "$OUT_DIR/synthetic.log"
    ;;
  summary)
    for f in "$OUT_DIR"/rb_*.jsonl; do
      [ -f "$f" ] && python phase4_incremental.py --mode summary --out "$f"
    done
    ;;
  *)
    echo "dung: bash scripts/run_phase4_incremental.sh {smoke|main|sweep|sa|synthetic|summary}"
    exit 1
    ;;
esac
