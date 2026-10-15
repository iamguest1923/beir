# Đánh giá tám phương pháp BEIR theo cấu trúc ANCE

Tất cả entrypoint cấu hình ở đầu file và chạy trực tiếp trong môi trường BEIR.
Mặc định chỉ đánh giá `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`,
`trec-covid`; `trec` được chuẩn hóa thành `trec-covid`. Split là `test`.
Không dùng sudo, Docker, hoặc thay thế bằng các model khác tên.

| Phương pháp | Entrypoint | Pipeline |
|---|---|---|
| DeepCT | `DeepCT/evaluate_deepct.py` | Checkpoint gốc chuyển PyTorch → trọng số từ → Anserini BM25 |
| SPARTA | `SPARTA/evaluate_sparta.py` | BEIR SPARTA → sparse matrix → SparseSearch |
| docT5query | `docT5query/evaluate_doct5query.py` | T5 sinh 40 query/document → nối passage → Anserini |
| DPR | `DPR/evaluate_dpr.py` | Multi-DPR question/context encoder riêng → exact dot |
| TAS-B | `TAS-B/evaluate_tasb.py` | TAS-B checkpoint → exact dot |
| GenQ | `GenQ/evaluate_genq.py` | Sinh 5 query/document → fine-tune TAS-B riêng mỗi dataset → exact dot |
| ColBERT | `ColBERT/evaluate_colbert.py` | Checkpoint v1 gốc → token index → IVFPQ ANN → MaxSim |
| BM25+CE | `BM25+CE/evaluate_bm25_ce.py` | Anserini top-100 → MiniLM-L6-v2 reranking |

## Chuẩn bị

Từ `/hdd2/devbe`, kích hoạt môi trường `beir` đang dùng:

```bash
python -m pip install -r code/beir/runs/requirements-models.txt
python code/beir/runs/setup_external.py colbert
```

DeepCT mặc định dùng `BACKEND="pytorch-converted"`, checkpoint `macavaney/deepct`
(model card xác nhận chuyển từ checkpoint tác giả gốc), không cần TensorFlow.
Backend `tensorflow-original` vẫn có sẵn và cần setup riêng. URL BERT Google
cũ đã trả AccessDenied nên config/vocab lấy từ `google-bert/bert-base-uncased`.
Nếu dùng backend này: `python code/beir/runs/setup_external.py deepct`.

DeepCT TensorFlow/ColBERT có environment riêng, không hạ phiên bản PyTorch/Transformers
đang chạy ANCE. Setup pin source Git, tạo venv bằng Python hiện tại (khuyến nghị
3.9) và cài requirements riêng. DeepCT kèm CUDA libraries qua pip; không cần
quyền quản trị. ColBERT dùng torch 1.13.1/Transformers 4.30.2 để tương thích
API v1, không tuyên bố đây là lock môi trường năm 2021. Runtime mới vẫn cần
được chạy end-to-end để kiểm chứng. Checkpoint chỉ tải khi runner cần sử dụng.

## Chạy một model hoặc chạy lần lượt

```bash
python code/beir/runs/DPR/evaluate_dpr.py
python code/beir/runs/run_models.py
```

Chạy đồng thời tất cả phương pháp (theo lựa chọn mới của người dùng):

```bash
python code/beir/runs/run_models_parallel.py
```

Supervisor song song khởi chạy các model cùng lúc và chờ process kết thúc bằng
Linux pidfd, không đọc log liên tục. Mỗi model vẫn chạy lần lượt sáu dataset.
Không chạy chung với supervisor tuần tự; cả hai dùng cùng file lock. Tham số
`--attach-state <state.json>` chỉ dùng sau khi dừng supervisor cũ cùng watcher
retry, để tiếp tục theo dõi model đang chạy mà không dừng model đó.
Phiên song song ghi trạng thái và summary trong `model_queues` như phiên tuần tự.
Nó không tự sửa code hoặc tự chạy lại model lỗi.

Sau khi sửa nguyên nhân lỗi, có thể khôi phục riêng các dataset thất bại:

```bash
python code/beir/runs/recover_failed_models.py --state <session>/state.json
```

Lệnh này giữ các model đang chạy, khởi chạy đồng thời những model đã dừng do
lỗi và chỉ chọn dataset có status FAILED. Nếu GenQ còn chạy, đợi process đó
kết thúc bằng pidfd rồi mới retry để tránh hai tiến trình cùng ghi cache.
Kết quả thành công cũ được giữ và summary/status gộp theo model/dataset.
Mỗi model được retry một lượt; lỗi mới vẫn cần được kiểm tra và sửa.

Anserini 0.12 mặc định bỏ tài liệu có contents rỗng trước khi xét title. Adapter
BM25 dùng generator Java nhỏ cho corpus có trường hợp này; giữ nguyên nội dung
rỗng và title, không thêm từ giả, vẫn xác minh đủ số tài liệu. Cần `javac` Java
11 để biên dịch tại thư mục kết quả, không cần quyền quản trị. ColBERT chấm
MaxSim theo nhóm 256 candidates thay vì cấp bộ đệm 16.384, giữ nguyên thuật
toán và tập candidates. Retry tận dụng index cũ khi input/config/checkpoint
khớp; không xóa index được tái sử dụng.

Hàng đợi chạy đúng thứ tự bảng trên, mỗi lần một subprocess. Model lỗi vẫn
chuyển sang model kế tiếp; mỗi dataset lỗi được ghi riêng và chuyển dataset.
File lock ngăn mở hai hàng đợi đồng thời. Không dừng hay chỉnh process ANCE/BM25
đang chạy trước đó. Mỗi model có config, log và results riêng.

Tác vụ dài nên chạy độc lập terminal:

```bash
nohup python -u code/beir/runs/run_models.py > /tmp/beir-model-queue.log 2>&1 < /dev/null &
```

Không chạy thêm hàng đợi nếu đã có một phiên chạy. `model_queues/latest.json`
trỏ đến phiên mới nhất, `state.json` ghi PID/job/trạng thái/exit code và tối đa
20 dòng log cuối khi lỗi. `summary.csv` ở cấp queue cập nhật khi mỗi model kết
thúc; trong lúc một model đang chạy, xem status/summary của runner đó.

```bash
python code/beir/runs/queue_status.py
```

Lệnh status chỉ đọc artifact trạng thái nhỏ và phần log mới theo byte offset;
không đọc ranking hay toàn bộ log. Để sửa lỗi và chạy lại một model, chạy lại
entrypoint của model đó sau khi process cũ đã kết thúc.

`retry_failed_models.py` chờ supervisor bằng Linux pidfd (không đọc log liên tục),
sau đó chạy lại các model lỗi một lượt, vẫn tuần tự. Các bản sửa code được nạp
bởi subprocess mới. Nó không tự sửa các lỗi chưa được xử lý. Summary giữ lượt
mới nhất của từng model/dataset; artifact cũ vẫn còn để đối chiếu.

## Đầu ra

```text
<model>/results/<run_id>/
├── config.json
├── evaluate_*.py
├── source/                    # Snapshot shared implementation
├── run.log
├── summary.csv
├── status.json
└── <dataset>/
    ├── config.json
    ├── data.json              # Dataset hashes, counts, artifact checksums
    ├── status.json
    ├── run.log
    ├── metrics.json
    ├── ranking.run.trec
    └── ...                    # Model/training/index metadata
```

Tái sử dụng GenericDataLoader, EvaluateRetrieval, SentenceBERT, SPARTA,
SparseSearch, QGenModel, Rerank, CrossEncoder, save_results và save_runfile.
Các phương pháp lexical tái sử dụng adapter BM25 local đã chạy NFCorpus.

## Cache, tiếp tục và tài nguyên

- Tìm dataset trong folder runner, ANCE, BM25 và các example; không sao chép
  corpus nếu đã có đủ file. Dataset input được hash SHA-256 trước khi chấm.
- Kết quả dataset hoàn tất được reuse khi config/source/data khớp, status SUCCESS
  và checksum metric/ranking hợp lệ. Lượt mới lưu `reused_from.json`.
- docT5query/GenQ cache query sinh theo chunk với cấu hình, checkpoint revision,
  corpus fingerprint và seed. Khi ngắt chỉ chunk chưa ghi xong phải sinh lại.
- GenQ lưu model fine-tuned cùng completion marker; training đang dở phải train
  lại từ đầu. Không dùng test queries/qrels để train hoặc chọn checkpoint.
- SPARTA cache sparse matrix hoàn tất; BM25 candidates được cache riêng.
- Dense search chưa lưu embedding theo chunk. ColBERT chưa resume index đang dở.
  Không chạy nhiều process của cùng phương pháp/dataset ghi chung cache.
- Các index tạm lexical/ColBERT mặc định xóa sau khi search thành công, giữ khi
  lỗi. Checkpoint và corpus/cache có thể tốn hàng chục GB. Không tự xóa kết quả
  hoặc cache người dùng. Một model chạy xong được giải phóng GPU qua kết thúc process.

GenQ/docT5query có thể chạy lâu hơn nhiều so với đánh giá dense có sẵn, do phải
sinh hàng trăm nghìn đến hàng triệu query. Không tự giảm số query, số document
hoặc số query test chỉ để chạy nhanh. Các bước inference mặc định FP32;
ColBERT v1 vẫn lưu token embeddings FP16 theo implementation upstream.

## Giới hạn tái lập cần ghi trong báo cáo

- GenQ dùng một epoch, batch 32, lr 2e-5, dot-product loss scale=1 và sample seed 42
  làm mặc định minh bạch; paper không cung cấp trọn mọi training hyperparameter.
- DeepCT dùng max_length=512, scale=100, smoothing=none, keep_all_terms=True;
  tham số này ghi trong config, không ngầm coi là trùng mọi chi tiết paper.
- docT5query nối query sinh vào contents, giữ title field riêng.
- ColBERT dùng checkpoint v1, không v2. IVF partitions cấu hình theo dataset là
  giá trị khởi điểm; không có bằng chứng đây là đúng partitions tác giả dùng.
  V1 xuất qid/pid/rank; runner ghi negative rank làm score để bảo toàn thứ tự.
  ColBERT và CE chỉ trả top-100: Recall@1000 bị giới hạn bởi candidate depth,
  không được diễn giải như retrieval depth 1.000 của dense models.
- Chỉ status SUCCESS với artifact thực mới được xem là đã đánh giá. Unit test,
  preflight, việc cài dependency hoặc số paper tham chiếu không phải kết quả chạy.

Nguồn: bài báo https://arxiv.org/html/2104.08663v4 ; example hiện có trong repo;
https://github.com/Nthakur20/DeepCT ; https://github.com/Nthakur20/beir-ColBERT .
