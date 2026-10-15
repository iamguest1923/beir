# TAS-B

Từ `/hdd2/devbe`, kích hoạt môi trường BEIR và chạy:

```bash
python code/beir/runs/TAS-B/evaluate_tasb.py
```

Cấu hình đầu file giống ANCE; mặc định chỉ sáu dataset được yêu cầu.
`trec` được hiểu là `trec-covid`. Kết quả nằm trong `results/<run_id>/<dataset>/`.
Xem [hướng dẫn chung](../README_MODELS.md) để cài dependency, chuẩn bị runtime
và hiểu giới hạn cache/tái lập. DeepCT/ColBERT cần chạy setup_external.py trước.
