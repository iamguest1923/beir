# Kết quả BEIR từ các lượt chạy thực tế

Thời điểm xuất (UTC): 2026-10-15T02:15:28.520724+00:00

- ndcg_at_10.csv: bảng nDCG@10 theo thứ tự dataset/model trong ảnh, làm tròn 3 chữ số.
- ndcg_at_10_full_precision.csv: cùng bảng, giữ nguyên độ chính xác lưu trong metrics.json.
- sources.csv: nguồn artifact, split, run ID và SHA-256 cho từng kết quả.
- all_metrics.csv: tất cả metric/k đã lưu cho các kết quả được chọn.

Chỉ chọn status SUCCESS ở cấp dataset và lấy lượt thành công mới nhất theo thời điểm bắt đầu session.
Kết quả dataset hoàn tất trong session bị dừng vẫn được giữ. Không lấy điểm từ bài báo hoặc smoke test.
Ô chưa có kết quả thành công dùng ký tự _. GenQ chưa có dataset đánh giá hoàn tất.
MS MARCO dùng dev; các dataset còn lại dùng test. CQADupStack là trung bình đều 12 subsets theo runner.
Dòng Avg. Performance vs. BM25 để _ do chưa đủ dữ liệu cho mức trung bình toàn benchmark như bảng mẫu.
Điểm phản ánh code/cấu hình đã chạy; không khẳng định trùng cấu hình hay điểm của bài báo.

Tổng kết quả model/dataset: 56.

| Model | Dataset thành công |
|---|---:|
| BM25 | 7 |
| DeepCT | 6 |
| SPARTA | 6 |
| docT5query | 3 |
| DPR | 6 |
| ANCE | 10 |
| TAS-B | 6 |
| GenQ | 0 |
| ColBERT | 6 |
| BM25+CE | 6 |
