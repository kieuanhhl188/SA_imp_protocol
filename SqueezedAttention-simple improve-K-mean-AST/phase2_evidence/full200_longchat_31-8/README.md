# Phase 2 full — LongChat-7B, LCC, `LIMIT_P2=200` (31/8/2026)

`LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh` trên A100-80GB.
Lượt full đầu tiên trên LongChat (smoke 3 mẫu: `../smoke_longchat_29-8/`).
Centroid ~88 GB ở lại pod (`/workspace/p2-longchat/`), KHÔNG vào git.

## Kiểm bất biến — `p2_invariants_longchat.log`, 200 mẫu × 3 nhánh, TẤT CẢ PASS

- [B] cùng budget K ba nhánh: khớp 200/200
- [A] `hard_boundary` + `struct_hierarchy`: **0,0% cluster vắt qua >1 unit** — 400/400 mẫu
- [A] `sa` (đối chứng): vắt biên trung vị **32,3%** (p25 21,5 · p75 42,7 · min 0,6 · max 85,2)
- [C] shape `[1,32,K,128]` / `[1,32,n_ctx]`, **0,0% ô rỗng** cả 600 file
- `feasibility_lcc_*_function_pc5.json`: `hard_boundary` 200/200 · `struct_hierarchy` 200/200 ·
  **0 infeasible** (chính sách D6 `skip`)

## `k1_stats_summary.json` — tầng L1 vô hiệu trên LCC/function

k1_raw (unit cấp class) trung vị **3** · k1_actual trung vị **16,5** · mode split 199 / merge 1.
`build_l1_groups` chẻ tới số function → `struct_hierarchy` ≡ `hard_boundary`. KHÔNG mô tả
như hierarchy 3 tầng class→function→token trong bài.

## Phase 5 — C2 recall@budget · `phase5_lcc.json` · n=100 · layers first/mid/last

| method | recall@70 | recall@80 | recall@90 | mass@70 |
|---|---:|---:|---:|---:|
| `sa` | **0,7435** | **0,6944** | **0,6238** | 0,9680 |
| `hard_boundary` | 0,7335 | 0,6796 | 0,6005 | 0,9425 |
| `struct_hierarchy` | 0,7335 | 0,6796 | 0,6005 | 0,9425 |

- `struct_hierarchy` ≡ `hard_boundary` từng chữ số (hệ quả của L1 vô hiệu).
- Ranh giới cứng recall **thấp hơn** `sa` ở mọi mức: −0,010 @70 · −0,015 @80 · −0,023 @90.
  Chưa kiểm định (n=100). Ba mức cùng dấu, biên độ tăng theo sparsity → nhiều khả năng thật,
  **ngược chiều giả thuyết Idea 1**.
- `phase5_smoke.json` — smoke 3 mẫu, `hard_boundary@70` recall 0,748.

## Việc còn lại

- C2: chạy `--limit` lớn hơn + bootstrap ghép cặp (như Phase 0) để có kiểm định.
- Chạy trên bộ context dài (RepoBench v1.1) — LCC ~3k token quá ngắn cho structure-aware clustering.
- Phase 6: accuracy@budget end-task.
