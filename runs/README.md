# Tra cứu các lượt đánh giá

Đây là điểm bắt đầu để tìm code, kết quả và trạng thái trong `beir/runs`.
Các thư mục hiện giữ nguyên đường dẫn vì kết quả, cache và tiến trình có thể tham chiếu đến chúng.

## Chọn đúng nơi cần xem

| Bạn cần xem gì? | Mở ở đâu? |
|---|---|
| Điểm của 10 model cũ trên 9 dataset | [Bảng nDCG@10](completion_queues/20261015T024254Z_completion_a8cd4b36/ndcg_at_10.csv) |
| Điểm của 4 model mới trên 9 dataset | [Bảng nDCG@10](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/ndcg_at_10.csv) |
| Đường dẫn kết quả gốc của từng cặp model–dataset cũ | [Coverage 10 model](completion_queues/20261015T024254Z_completion_a8cd4b36/coverage.csv), cột `directory` |
| Đường dẫn kết quả gốc của từng cặp model–dataset mới | [Coverage 4 model](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/coverage.csv), cột `directory` |
| Ranking và metric chi tiết của một lượt | `<model>/results/<run_id>/<dataset>/` |
| Code đánh giá | `<model>/evaluate_*.py` và `_shared/` |
| Báo cáo Word và biểu đồ | [reports/](reports/) |

## Phân biệt năm thư mục dễ nhầm

| Thư mục | Vai trò | Phạm vi | Dùng để lấy kết quả báo cáo? |
|---|---|---|---|
| `multilingual_environment/` | Môi trường Python: executable và thư viện phụ thuộc | Runtime cho bốn model mới | Không; không chứa lượt đánh giá |
| `multilingual_queues/` | Điều phối bốn model mới, lưu trạng thái, log và CSV tổng hợp | BGE-M3, Multilingual-E5-Large, GreenNode-Embedding, Vietnamese-BiEncoder × 9 dataset | Có, lấy các cặp `SUCCESS` trong `coverage.csv` |
| `completion_queues/` | Kiểm tra kết quả cũ và chạy bổ sung các cặp còn thiếu | BM25, ANCE, TAS-B, DPR, DeepCT, SPARTA, BM25+CE, ColBERT, GenQ, docT5query × 9 dataset | Có, lấy các cặp `SUCCESS` trong `coverage.csv` |
| `model_queues/` | Lịch sử điều phối, chạy song song và khôi phục lỗi trước đợt completion | Chủ yếu sáu dataset BEIR ban đầu | Tra cứu lịch sử; dùng bảng completion để xem kết quả đã được tổng hợp và kiểm tra |
| `model_checks/` | Chạy thử kỹ thuật: tải model, mã hóa, sinh query, huấn luyện và chấm thử | Các mẫu kiểm tra nhỏ của CE, DPR, DeepCT, docT5query và GenQ | Không; có `metrics.json` cũng không có nghĩa là kết quả toàn bộ dataset |

Một **queue** là đợt điều phối nhiều lượt chạy. Một **run** là một lần thực thi của một model, có thể đánh giá nhiều dataset. Một **attempt** là một lần thử, kể cả thử bị từ chối tải model. Số thư mục/log không phải số kết quả hoàn thành.

## Hai đợt đánh giá chính

Trạng thái dưới đây là snapshot ngày 19/10/2026. CSV và `state.json` tại các đường dẫn tương ứng được supervisor cập nhật; nội dung README này không tự cập nhật.

| Đợt | Mã đợt | Kết quả được xác minh | Trạng thái |
|---|---|---:|---|
| 10 model cũ | `20261015T024254Z_completion_a8cd4b36` | 90/90 | COMPLETE |
| 4 model mới | `20261015T154500Z_multilingual_6fff82ea` | 27/36 | Ba model hoàn tất; GreenNode WAITING_ACCESS |

Hai đợt cùng dùng split `test` của chín dataset:

- Dataset của bài báo: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`, `trec-covid`.
- Dataset mở rộng: `uit-viquad`, `vihealthqa`, `zalo-legal`.

Nếu chỉ xét tám model của báo cáo Word trước đây, bỏ hai cột GenQ và docT5query trong bảng 10 model. Bốn model mới thuộc đợt bổ sung riêng.

### Đợt 10 model cũ

- [nDCG@10](completion_queues/20261015T024254Z_completion_a8cd4b36/ndcg_at_10.csv)
- [Recall@100](completion_queues/20261015T024254Z_completion_a8cd4b36/recall_at_100.csv)
- [Coverage và đường dẫn kết quả gốc](completion_queues/20261015T024254Z_completion_a8cd4b36/coverage.csv)
- [Trạng thái](completion_queues/20261015T024254Z_completion_a8cd4b36/state.json)
- [Hướng dẫn chạy và audit](COMPLETION.md)

### Đợt 4 model mới

- [nDCG@10](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/ndcg_at_10.csv)
- [Recall@100](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/recall_at_100.csv)
- [Coverage và đường dẫn kết quả gốc](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/coverage.csv)
- [Trạng thái](multilingual_queues/20261015T154500Z_multilingual_6fff82ea/state.json)
- [Hướng dẫn model, tiền xử lý và chạy](README_MULTILINGUAL.md)

`_` trong CSV là chưa có kết quả hợp lệ, không phải điểm 0. GreenNode đang có nhiều lần thử tải bị từ chối; các lần thử này không được tính là đánh giá thành công.

## Từ bảng tổng hợp đến kết quả gốc

1. Chọn đợt 10 model hoặc 4 model.
2. Mở `coverage.csv`, chọn hàng theo `model` và `dataset`.
3. Kiểm tra `status` là `SUCCESS` rồi mở đường dẫn trong cột `directory`.
4. Đọc `metrics.json` để xem điểm, `ranking.run.trec` để xem thứ hạng, `data.json` và `config.json` để đối chiếu dữ liệu/cấu hình.

```text
<model>/results/<run_id>/
├── config.json                 cấu hình lượt chạy
├── status.json                 trạng thái lượt chạy
├── summary.csv                 các dataset của lượt này
├── run.log                     log thực thi
└── <dataset>/
    ├── status.json              trạng thái dataset
    ├── data.json                nguồn dữ liệu, số lượng và hash
    ├── config.json              cấu hình dataset
    ├── metrics.json             điểm đánh giá
    └── ranking.run.trec         kết quả xếp hạng
```

Một model có thể có nhiều `run_id`; queue completion có thể chọn kết quả hợp lệ từ các lượt khác nhau. Không chọn kết quả chỉ vì thư mục có timestamp mới nhất.

## Các con trỏ và lịch sử

Mỗi nhóm queue có `latest.json` riêng. Nó trỏ đến đợt mới nhất **trong nhóm đó**, không phải đợt mới nhất của toàn bộ dự án. Các liên kết cố định phía trên trỏ đến hai đợt đã xác định trong README này.

Ba phiên trong `model_queues/` hiện là lịch sử:

| Phiên | Trạng thái ghi trong state |
|---|---|
| `20261014T174813Z_92902bec` | REPLACED_BY_PARALLEL |
| `20261014T182619Z_parallel_e4ecf074` | STOPPED_BY_USER |
| `20261015T010034Z_recovery_ea58b155` | STOPPED_BY_USER |

Job cũ có thể vẫn ghi `RUNNING` trong snapshot lịch sử; cần kiểm tra tiến trình thực tế trước khi kết luận còn chạy.

`queue_status.py` hiện chỉ đọc `model_queues/latest.json`; lệnh đó không phản ánh tiến độ của hai nhóm completion và multilingual.

Dữ liệu đầu vào tiếng Việt nằm ngoài `runs`, tại `../new/data/datasets/`. Vị trí corpus thực tế mà từng kết quả sử dụng được ghi trong `data.json` và thông tin đầu vào của queue.
