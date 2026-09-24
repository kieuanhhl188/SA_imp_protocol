#!/usr/bin/env bash
# Full run n=200 RepoBench-P, LongChat-7B-v1.5-32K: All-KV / SA-70% / class-level 70%.
# Chay lai duoc: moi lenh pred.py deu --resume -> bi ngat thi chay lai y nguyen script nay,
# mau da xong duoc bo qua (cau hinh da xong het chi mat ~1 phut nap du lieu).
#   nohup bash /workspace/run_full_200.sh > /workspace/run_full_200.out 2>&1 &
set -eo pipefail

# venv310 + HF_HOME + HF_DATASETS_TRUST_REMOTE_CODE=1 (thieu bien nay job TREO o prompt [y/N])
source /workspace/env.sh
export CUDA_VISIBLE_DEVICES=0

REPO="/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
cd "$REPO/LongBench"   # pred.py/eval.py doc config/ va ghi pred/ theo duong dan tuong doi

LOG_DIR="/workspace/logs_full_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
M="$LOG_DIR/master.log"
log() { echo "$*" | tee -a "$M"; }

MODEL="longchat-v1.5-7b-32k"
TASK="repobench-p"      # ten task LongBench
LIMIT=200               # centroid chi co cho dataidx 0..199 (tap co 500 mau)
PCT=5                   # K = 5% * (shared_prefix_length - obs_window), trung luc cluster
PERCENTILE="0.7"        # Sq-70%
OBS=100
P2="/workspace/p2-longchat-repobench"

COMMON=(--model "$MODEL" --task "$TASK" --limit "$LIMIT" --fixed_context full --resume)
SA_ARGS=(--use_centroids --percent_clusters "$PCT" --percentile "$PERCENTILE" --obs_window "$OBS")
EVAL_SA=(--use_centroids --percent_clusters "$PCT" --percentile "$PERCENTILE")
# run_tag tach thu muc: SA va class co cung ten cau hinh PC5_PERC0.7_lim200, khong co tag
# thi lan class se append vao file prediction cua SA.

log "================================================================="
log "=== BAT DAU FULL RUN (n=$LIMIT $TASK) LUC $(date) ==="
log "=== python: $(command -v python)   log: $LOG_DIR"
log "================================================================="

run() {  # ten, file log, args pred.py..., --, args eval.py...
  local name=$1 logf=$2; shift 2
  local pargs=()
  while [ "$1" != "--" ]; do pargs+=("$1"); shift; done; shift
  log ""
  log "[$name] $(date) - pred.py"
  python pred.py "${COMMON[@]}" "${pargs[@]}" 2>&1 | tee "$LOG_DIR/$logf"
  log "[$name] $(date) - eval.py"
  # --expect: thieu mau thi eval.py tu choi cham, set -e dung script
  python eval.py --model "$MODEL" --limit "$LIMIT" --expect "$LIMIT" "$@" \
      2>&1 | tee -a "$LOG_DIR/$logf" | grep -v Warning | tail -3 | tee -a "$M"
}

# 1/3 All-KV (~15-20 phut)
run "1/3 All-KV" run_all_kv.log \
    --run_tag full200 \
    -- --run_tag full200

# 2/3 SA-70% (K-means thuong, ~4,7 h)
run "2/3 SA-70%" run_sa_70.log \
    "${SA_ARGS[@]}" --path_to_clusters "$P2/sa/" --run_tag full200sa \
    -- "${EVAL_SA[@]}" --run_tag full200sa

# 3/3 class-level 70% (hard_boundary --level class, ~4,7 h)
run "3/3 class-70%" run_class_level_70.log \
    "${SA_ARGS[@]}" --path_to_clusters "$P2/hard_boundary_class/" --run_tag full200class \
    -- "${EVAL_SA[@]}" --run_tag full200class

log ""
log "================================================================="
log "=== XONG CA 3 CAU HINH LUC $(date) ==="
log "=== Ket qua : $REPO/LongBench/pred/${MODEL}_*_lim${LIMIT}_runfull200*/"
log "=== VRAM/thoi gian tung mau: <thu muc ket qua>/_logs/${TASK}.stats.jsonl"
log "=== So sanh GHEP CAP bang scripts/compare_runs.py, khong tru hai trung binh"
log "================================================================="
