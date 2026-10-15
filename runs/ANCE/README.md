# ANCE trên các dataset public của BEIR

Chạy từ `/hdd2/devbe` với môi trường BEIR đã kích hoạt:

```bash
python code/beir/runs/ANCE/evaluate_ance.py
```

Cấu hình nằm ở đầu `evaluate_ance.py`, không cần CLI. Mặc định chạy lần lượt
15 dataset public, bắt đầu bằng NFCorpus và SciFact. Không chạy BPR hay reranker.
ANCE dùng checkpoint `sentence-transformers/msmarco-roberta-base-ance-firstp`,
512 token, dot product và exact search. Model chỉ tải một lần mỗi phiên.

Để chỉ chạy thử NFCorpus, sửa:

```python
DATASETS = ["nfcorpus"]
```

Các biến khác: `BATCH_SIZE`, `CORPUS_CHUNK_SIZE`, `DEVICE`, `MODEL_REVISION`,
`SPLITS`, `DATA_DIR`, `EXISTING_DATA_DIRS`. Đặt `MODEL_REVISION` thành commit SHA
Hugging Face nếu muốn pin checkpoint. Mặc định GPU được chọn tự động nếu có.

## Dữ liệu

Script ưu tiên dữ liệu đủ file tại `runs/ANCE/datasets/`, sau đó tìm trong
các thư mục dataset của example dense/lexical/sparse/reranking. Dữ liệu được
tái sử dụng chỉ được kiểm tra cấu trúc; điều này không xác nhận phiên bản hoặc
checksum corpus. ZIP mới được tải qua file `.part`, kiểm tra định dạng, rồi
giải nén sang thư mục tạm. Thiếu dung lượng giải nén sẽ báo lỗi. Không xác thực
checksum theo manifest upstream; kiểm tra CRC diễn ra khi giải nén ZIP.

Mặc định xóa ZIP mới sau khi giải nén thành công; không xóa corpus hay dữ liệu
được tái sử dụng. Vì vậy corpus vẫn tích lũy qua các dataset: cần chủ động chuẩn
bị dung lượng, không mặc định rằng 114 GB đủ cho cả bộ. Mỗi lần chỉ giữ corpus
và kết quả của một dataset trong bộ nhớ, trừ tài nguyên model dùng chung.

MS MARCO dùng `dev` (6.980 query) để tương ứng hàng MS MARCO trong bảng BEIR,
không dùng `test` 43 query. CQADupStack chạy 12 tập con riêng và lấy trung bình
không trọng số của metric giữa các tập con. Không trộn các query thành một tập.
BioASQ, Signal-1M, TREC-NEWS, Robust04 không có trong danh sách tải tự động.

## Đầu ra

```text
results/<UTC_timestamp>_<unique_id>/
├── config.json               # Model, resolved revision, phiên bản, cấu hình
├── evaluate_ance.py           # Snapshot script thực thi
├── run.log                   # Log chung (không gồm thanh tqdm trên stderr)
├── summary.csv               # Một dòng/dataset, NDCG@10 và Recall@100
└── <dataset>/
    ├── config.json
    ├── status.json           # SUCCESS hoặc FAILED và lỗi
    ├── run.log
    ├── data.json             # Đường dẫn, số documents/queries, thời gian
    ├── metrics.json          # NDCG, MAP, Recall, Precision, MRR
    └── ranking.run.trec      # Top 1.000 tài liệu/query
```

Với CQADupStack, ranking và `data.json` nằm trong từng thư mục tập con;
`metrics.json` ở cấp dataset chứa trung bình, `aggregation.json` ghi quy tắc.

Dataset lỗi được ghi lại rồi script tiếp tục; exit code cuối là 1 nếu có lỗi.
Ctrl+C dừng phiên. Mỗi lần chạy tạo phiên mới, không tự bỏ qua kết quả phiên cũ
và không cache embedding. Có thể tiếp tục phần chưa chạy bằng cách sửa DATASETS.
Không chạy hai process cùng tải một dataset vào DATA_DIR.

NDCG@10 là chỉ số so với Table 2 của bài báo. Chạy được pipeline chưa bảo đảm
tái lập chính xác mọi số: cần đối chiếu dữ liệu, checkpoint và cấu hình gốc.
