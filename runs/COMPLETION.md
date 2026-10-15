# Hoàn tất ma trận model × dataset

Phạm vi: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`, `trec-covid` và mọi thư mục BEIR có đủ `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` trong `beir/new/data/datasets`. Snapshot hiện tại có thêm `uit-viquad`, `vihealthqa`, `zalo-legal`: tổng cộng **10 model × 9 dataset = 90 cặp**. CSConDa chưa có dữ liệu local nên chưa nằm trong ma trận.

Model: BM25, ANCE, TAS-B, DPR, DeepCT, SPARTA, BM25+CE, ColBERT, GenQ, docT5query. Giữ các checkpoint và thuật toán của runner hiện có, không thay bằng model khác. Bộ tiếng Việt dùng cùng model đã dùng trên BEIR tiếng Anh để đo khả năng chuyển miền/ngôn ngữ.

## Kiểm tra kết quả

```bash
/hdd2/devbe/.cache/conda_envs/beir/bin/python beir/runs/complete_benchmark.py --audit-only
```

Audit không chỉ dựa vào CSV: đọc status/config/data/metrics/ranking của từng phiên, kiểm tra split test, số document/query, input hashes (nếu runner cũ có lưu), checksum artifact, ID trong ranking và tính lại toàn bộ metric/k bằng `EvaluateRetrieval`. Kết quả không khớp được đưa trở lại danh sách cần chạy. Phiên BM25/ANCE cũ chưa lưu input hashes được đánh dấu `legacy_input_hashes_absent`; audit đối chiếu các file tại đường dẫn nguồn đã ghi và số lượng. Không thể chứng minh file nguồn cũ chưa từng bị thay đổi nếu phiên đó không lưu hash ban đầu.

TREC runfile không ghi được query có zero hit. Audit khôi phục query đó bằng ranking rỗng trước khi tính lại metric và lưu danh sách trong `verified_artifacts.json`; không bỏ query khỏi mẫu số. Query/document ID lạ, duplicate pair, score không hữu hạn và metric tính lại không khớp đều bị từ chối.

Audit đầy đủ đầu tiên xác nhận **48/90 cặp** có kết quả hợp lệ. Không lấy điểm từ bài báo hoặc smoke test.

## Chạy các cặp còn thiếu

```bash
/hdd2/devbe/.cache/conda_envs/beir/bin/python -u beir/runs/complete_benchmark.py
```

Chạy trong môi trường BEIR đã cài dependencies. Để chạy độc lập terminal, dùng subprocess/nohup như các queue cũ. Supervisor dùng lock riêng và lock queue cũ để ngăn nhiều hàng đợi ghi chung cache. Không dừng các tiến trình chạy riêng trước đó.

- BM25 chạy ở một lane CPU; các model còn lại chạy lần lượt ở một lane GPU.
- Chỉ chọn cặp chưa có kết quả hợp lệ. Các phiên cũ và CSV cũ được giữ nguyên.
- Dùng `_shared.common.run` và adapter hiện có cho toàn bộ model; thêm dispatcher cho BM25/ANCE. Mặc định sáu dataset trong các entrypoint cũ không thay đổi.
- BM25 tái sử dụng generator giữ document có contents rỗng, kể cả document chỉ có title.
- BM25+CE dùng batch 8 như recovery runner cũ. ColBERT dùng nhóm 256 candidates và bổ sung partitions cho ba bộ mới.
- GenQ giữ batch train 32, 5 query/document và cấu hình huấn luyện hiện có. docT5query giữ 40 query/document. Không rút ngắn corpus hay thay đổi lượng sinh để tạo kết quả nhanh hơn.
- Tận dụng các generation chunk, sparse matrix, BM25 candidate cache khớp input/config. GenQ không dùng test query/qrels để huấn luyện.
- Một model lỗi vẫn chuyển sang model tiếp theo; lỗi được giữ trong log/status. Nếu còn thiếu cặp ở cuối, trạng thái queue là `INCOMPLETE`, không phải `COMPLETE`. Chạy lại lệnh sau khi sửa nguyên nhân; lần mới chỉ chọn những cặp còn thiếu.

## Đầu ra và theo dõi

`beir/runs/completion_queues/latest.json` trỏ đến phiên audit/chạy gần nhất:

```text
completion_queues/<timestamp>_completion_<id>/
  state.json                 # PID supervisor, lane, job, PID worker, trạng thái
  supervisor.log
  <model>.log
  coverage.csv               # 90 hàng; nguồn kết quả và lỗi nếu còn thiếu
  ndcg_at_10.csv              # 9 dataset × 10 model; thiếu dùng _
  recall_at_100.csv
  inputs.json                # corpus/split/counts/hashes
  verified_artifacts.json    # audit cache và provenance
```

CSV và số cặp hợp lệ được cập nhật khi mỗi worker kết thúc. Trong lúc worker chạy, xem `summary.csv`, `status.json` và log tại `beir/runs/<model>/results/<run_id>/`; dữ liệu từng dataset hoàn tất được ghi ngay. Ranking, metrics và checksum của lượt mới ở các thư mục này, không nằm trong thư mục queue.

GenQ phải sinh query và fine-tune riêng từng dataset; docT5query phải sinh 40 query cho từng document nên có thể chạy rất lâu. Trạng thái `RUNNING`/`PENDING` không phải kết quả đánh giá. Chỉ `COMPLETE` với đủ 90 cặp đã qua audit mới có nghĩa hoàn tất phạm vi hiện tại.

## Kiểm thử

```bash
/hdd2/devbe/.cache/conda_envs/beir/bin/python beir/runs/test_model_runners.py
/hdd2/devbe/.cache/conda_envs/beir/bin/python beir/runs/test_complete_benchmark.py
```
