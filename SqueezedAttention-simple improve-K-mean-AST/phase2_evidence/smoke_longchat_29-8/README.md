# Phase 2 smoke — LongChat-7B, LCC, 3 mẫu (29/8/2026)

`LIMIT_P2=3 bash scripts/run_phase2_phase5_lcc.sh` trên A100-80GB.
Xác nhận đường ống Phase 2 chạy trên LongChat trước khi chạy full. KHÔNG phải kết quả.

## Kết quả

- `p2_invariants.log` — mọi bất biến QUA:
  - [B] cùng budget K ba nhánh: dataidx 0/1/2 = 897/194/136
  - [A] `hard_boundary` + `struct_hierarchy`: **0,0% cluster vắt biên** cả 3 mẫu
  - [A] `sa` (đối chứng): 22,8% / 33,7% / 36,7% vắt biên
  - [C] shape `[1,32,K,128]` / `[1,32,n_ctx]`, 0% ô rỗng
- `phase5_smoke.json` — `phase5_recall.py` chạy được trên LongChat: `hard_boundary@70`
  recall 0,748 · mass 0,954
- `*/feasibility_*.json` — `"infeasible": []` (3/3 mẫu khả thi ở level=function)
- `struct_hierarchy/k1_stats.json` — K1 thực tế 51/23/20, đều mode=split (K1 chẻ tới số
  function, `--level_l1` vô hiệu — xem EXPERIMENT_LOG Phase 2)

## Số đo full (ngoại suy từ smoke + Phase 0)

- Đĩa ~33 KB/token → ba nhánh full 500 ≈ 232 GB → **phải `LIMIT_P2=200`** (~88 GB)
- Thời gian: full ~18–24h · `LIMIT_P2=200` ~7–10h
