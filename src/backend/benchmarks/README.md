# HAgent Benchmark Suite (LeWM vs ReAct Baseline)

Thư mục này chứa kết quả đo đạc và so sánh hiệu năng giữa **HAgent (Latent World Model — LeWM)** và các baseline agents (như **ReAct**).

---

## 1. Mục tiêu và Chỉ số đo lường

Benchmark đánh giá khả năng tối ưu hóa AutoML thông qua 4 tiêu chí cốt lõi:

| Metric | Ý nghĩa | Khen thưởng (Tối ưu) |
|---|---|---|
| `runs_to_target` | Số vòng huấn luyện cần thiết để đạt độ chính xác mục tiêu ($Score \ge 0.90$) | Càng thấp càng tốt (hiệu quả mẫu) |
| `total_compute_time_s` | Tổng thời gian tính toán và tìm kiếm siêu tham số | Càng thấp càng tốt (tiết kiệm chi phí) |
| `surprise_rate` | Tỷ lệ các bước phát sinh surprise mức cao ($|y - \hat{\mu}| / \sigma \ge 3.0$) | Càng thấp càng tốt (World Model chuẩn xác) |
| `replan_frequency` | Số lần phải tái lập kế hoạch do chệch hướng | Càng thấp càng tốt (ổn định) |
| `final_best_score` | Độ chính xác cao nhất đạt được sau campaign | Càng cao càng tốt |

---

## 2. Cách thực thi

### Chạy Benchmark ở chế độ CI / Mock (Không cần API key hoặc live cluster)

```bash
python scripts/hagent_benchmark.py --mock --datasets iris wine breast_cancer --output benchmarks/results/
```

### Chạy Benchmark trên các dataset tùy chỉnh

```bash
python scripts/hagent_benchmark.py --datasets iris wine digits --runs 5 --output benchmarks/results/
```

---

## 3. Cấu trúc kết quả đầu ra

Mỗi lần chạy sẽ sinh một file JSON có timestamp tại `benchmarks/results/benchmark_<timestamp>.json` chứa:
- Metadata môi trường và danh sách dataset.
- Bảng tổng hợp (Summary Table) đối đầu HAgent vs ReAct.
- Chi tiết từng trial (trials logs) cho từng dataset.
