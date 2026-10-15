# BM25 trực tiếp trên máy, không Docker

Cách cấu hình và kết quả giống runner ANCE. Backend: Anserini qua Pyserini local,
không HTTP, service, Elasticsearch hay GPU. Tận dụng GenericDataLoader,
BaseSearch, EvaluateRetrieval, util.save_results và util.save_runfile của BEIR.

## Cài đặt một lần

Trong môi trường BEIR Python 3.9 hiện có, với Java 11:

```bash
java -version
python -m pip install -r code/beir/runs/BM25/requirements.txt
```

Pyserini 0.12.0 được pin để tương thích Java 11 và stack paper-era. Không tự
đổi sang bản mới nhất vì yêu cầu Python/Java/API có thể khác. Máy hiện tại
đã cài và khởi tạo JVM thành công. Nếu chọn sai Java, đặt trước khi chạy:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
```

## Chạy

Từ `/hdd2/devbe`, kích hoạt môi trường `beir` và chạy:

```bash
python code/beir/runs/BM25/evaluate_bm25.py
```

Cấu hình đầu file: DATASETS (mặc định 15 dataset public), K1=0.9, B=0.4,
FIELDS={contents:1.0,title:1.0}, THREADS=4, BATCH_SIZE=32, JAVA_HEAP="4g".
JAVA_TOOL_OPTIONS nếu đã có sẽ được giữ nguyên. BATCH_SIZE là số query mỗi
batch, không phải batch mã hóa. Muốn chạy thử: DATASETS=["nfcorpus"].

Dữ liệu được tìm trong runs/BM25/datasets, runs/ANCE/datasets rồi các example.
Chỉ tải khi chưa đủ file. MS MARCO dùng dev 6.980 query; các dataset khác dùng
test. CQADupStack chạy 12 tập con rồi lấy trung bình metric không trọng số.

## Quy trình và kết quả

Runner chuyển corpus sang JSONL, chạy `python -m pyserini.index` trực tiếp,
mở SimpleSearcher, đặt BM25 và gọi batch_search. Kiểm tra số documents sau
index và đủ query trong kết quả. Loại self-match trước khi cắt top-k. Không RM3.
Thiếu dependency hoặc JVM lỗi sẽ dừng trước khi tải dataset.

```text
results/<run_id>/
├── config.json                  # Backend, phiên bản, Java, cấu hình
├── evaluate_bm25.py              # Snapshot script
├── run.log
├── summary.csv
└── <dataset>/
    ├── config.json
    ├── data.json
    ├── status.json
    ├── run.log
    ├── index_command.json
    ├── indexing.log
    ├── metrics.json
    └── ranking.run.trec
```

CQADupStack có ranking/log riêng mỗi tập con; metrics cấp dataset là trung bình.
Không ghi đè ANCE hoặc phiên BM25 cũ. Một dataset lỗi sẽ ghi FAILED và tiếp tục;
exit code 1 nếu có lỗi. Không hỗ trợ resume giữa batch hoặc tự skip dataset đã
xong ở phiên cũ. Chọn DATASETS để chạy lại phần còn lại.

KEEP_INDEX=False mặc định xóa index sau khi search thành công, giữ khi lỗi.
Luôn đóng searcher trước khi xóa. JSONL chuyển đổi được xóa khi cleanup;
corpus gốc được giữ. Cần RAM cho corpus/query/results và ổ đĩa cho corpus,
JSONL chuyển đổi và index. Không tự reuse index hoặc cache kết quả giữa phiên.
Dataset folder có sẵn chỉ kiểm tra cấu trúc, không kiểm tra checksum phiên bản.

## Kiểm tra

```bash
python code/beir/runs/BM25/test_runner.py
```

Unit test dùng searcher giả cho luồng lỗi và xuất metric; không thay thế chạy
BM25 thật. NDCG@10 tham chiếu NFCorpus khoảng 0.325, SciFact khoảng 0.665.
