# Dataset retrieval tiếng Việt

Mã tải và chuyển đổi nằm tại đây; đầu ra dùng trực tiếp với `beir.datasets.data_loader.GenericDataLoader` của project. Không cần tải model để chuẩn bị dữ liệu.

## Tải và chuyển đổi

Chạy từ thư mục gốc checkout `beir`:

```bash
python -m pip install -r new/data/requirements.txt
python new/data/prepare.py --datasets uit-viquad vihealthqa zalo-legal
python new/data/prepare.py --datasets csconda
# Hoặc xử lý tất cả; một dataset lỗi không ngăn các dataset còn lại:
python new/data/prepare.py --datasets all
```

CSConDa yêu cầu [đăng ký quyền truy cập](https://huggingface.co/datasets/ura-hcmut/Vietnamese-Customer-Support-QA). Sau khi tài khoản được cấp quyền, đặt biến môi trường `HF_TOKEN` bằng token có quyền đọc rồi chạy lại. Chương trình không ghi token vào log hoặc metadata. Không có quyền truy cập thì chương trình báo lỗi, không tạo dữ liệu giả.

Các nguồn được cố định bằng commit SHA trong `SOURCES` ở `prepare.py`; `--revision <commit-sha>` cho phép chọn phiên bản khác cho một dataset. Tải dở dùng `.part`; lần chạy sau tận dụng file raw đã tải xong. Thư mục output đã tồn tại sẽ được giữ lại và báo lỗi; dùng `--output-dir` khác để tạo phiên bản mới. Chương trình trả mã thoát khác 0 nếu có dataset thất bại.

```text
new/data/
  prepare.py
  evaluate.py
  raw/<dataset>/<commit>/...         # cache nguồn
  datasets/<dataset>/
    corpus.jsonl                    # {"_id": "...", "title": "...", "text": "..."}
    queries.jsonl                   # {"_id": "...", "text": "..."}
    qrels/train.tsv                 # nếu nguồn có train
    qrels/dev.tsv                   # nếu nguồn có val/validation/dev
    qrels/test.tsv                  # nếu nguồn có test
    metadata.json
```

TSV dùng header `query-id`, `corpus-id`, `score`. `metadata.json` lưu nguồn, revision, hash file nguồn/đầu ra/mã xử lý, số lượng, thống kê gộp trùng và số câu hỏi có văn bản trùng giữa các split. Raw, output và results không được đưa vào Git.

## Nguồn và chính sách chuyển đổi

| Dataset | Nguồn mặc định | Chuyển đổi |
| --- | --- | --- |
| UIT-ViQuAD | [vile99/vi_quad](https://huggingface.co/datasets/vile99/vi_quad) | Mirror có cả câu answerable và unanswerable. Mỗi context là document; giữ câu có đáp án và kiểm tra đáp án nằm trong context. Context của câu unanswerable vẫn được giữ làm distractor. |
| ViHealthQA | [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA) | Nguồn QA chính thức: question → query, answer passage → document. Đây là bản QA chuyển đổi sang retrieval, không phải benchmark retrieval gốc. Không đưa câu hỏi vào title của document. |
| Zalo Legal Text Retrieval 2021 | [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) | Bản retrieval công khai: giữ nguyên ID điều luật, ID câu hỏi, split và các qrels nhiều positive. Gộp các query ID lặp có cùng nội dung; báo lỗi nếu cùng ID khác nội dung. Giữ 357 document chỉ có title ở snapshot này. |
| CSConDa | [ura-hcmut/Vietnamese-Customer-Support-QA](https://huggingface.co/datasets/ura-hcmut/Vietnamese-Customer-Support-QA) | Schema công bố gồm question/answer/type, **không có context**. Mặc định dùng answer passage làm document, question làm query. Đây là phép chuyển QA sang retrieval; không diễn giải là retrieval trên kho tài liệu hỗ trợ gốc. |

Các passage QA được chuẩn hóa Unicode NFC, gộp whitespace và gán ID SHA-256 theo nội dung; context trùng được gộp cả giữa các split. Query QA có prefix split để tránh ID cục bộ va chạm. Các split gốc được giữ, chỉ đổi `val`/`validation` thành `dev`; không tự chia train thành test. Corpus gồm các passage của toàn bộ split để mọi mô hình tìm kiếm trên cùng một corpus.

Không khẳng định mirror ViQuAD mặc định là bản 1.1 gốc: nó chứa câu hỏi không có đáp án. Nếu bài báo yêu cầu chính xác UIT-ViQuAD 1.1, dùng các file SQuAD được cấp từ tác giả qua `--source-dir` và ghi nhận nguồn này trong bài. Mã hỗ trợ cả SQuAD lồng `data/paragraphs/qas`, JSON Lines và bản QA phẳng.

ViHealthQA: file nguồn snapshot này có `train=7009`, `val=993`, `test=2013`, khác phần mô tả split trong dataset card. Mã giữ theo **tên file nguồn**, không tự đổi val/test.

Kết quả chuẩn bị thực tế tại snapshot mặc định:

| Dataset | Documents | Queries train | Queries dev | Queries test |
| --- | ---: | ---: | ---: | ---: |
| UIT-ViQuAD mirror, câu answerable | 4.101 | 15.333 | 1.937 | 1.970 |
| ViHealthQA | 9.894 | 7.009 | 993 | 2.013 |
| Zalo Legal | 61.425 | 2.432 | — | 788 |
| CSConDa | Chưa tải: cần quyền truy cập | — | — | — |

Qrels QA chỉ xác nhận passage đi kèm; các passage khác có thể liên quan nhưng chưa được gán nhãn. Không so sánh trực tiếp điểm retrieval này với điểm QA nguyên bản. Nguồn có query trùng giữa split (xem metadata); cần kiểm tra trước khi fine-tune/chọn checkpoint. Corpus dùng chung không đồng nghĩa được dùng test query/qrels để huấn luyện. CSConDa yêu cầu truy cập và tuân thủ điều kiện nghiên cứu phi thương mại, trích dẫn; ViHealthQA công bố dùng cho nghiên cứu.

## Dùng file có sẵn hoặc bản retrieval khác

```bash
# Thư mục chứa train.json, validation.json, test.json theo SQuAD/QA phẳng:
python new/data/prepare.py --datasets uit-viquad \
  --source-dir /path/to/official-viquad --output-dir /path/to/prepared

# Bản retrieval ViHealthQA đã xác minh đủ corpus, queries và qrels:
python new/data/prepare.py --datasets vihealthqa \
  --source-dir /path/to/vihealthqa-retrieval --output-dir /path/to/prepared

# CSConDa local có context thay vì answer:
python new/data/prepare.py --datasets csconda \
  --source-dir /path/to/local-csconda --context-field context
```

Nguồn local QA nhận `train`, `dev/val/valid/validation`, `test` với đuôi `.json`, `.jsonl`, `.csv`, `.parquet`, kể cả shard `train-00000-of-00001.parquet`. Dùng `--question-field`/`--context-field` để chỉ rõ trường. Với nguồn local đã là retrieval, cần `corpus.jsonl`, `queries.jsonl`, `qrels/{split}.tsv` hoặc `.jsonl`. Chế độ này giữ nguyên IDs và relevance, kiểm tra ID tham chiếu, score và positive coverage. Không suy diễn corpus/qrels từ danh sách negative của dữ liệu embedding.

## Đánh giá mô hình

Cài dependencies của project BEIR theo môi trường hiện có. Script mới tái sử dụng `GenericDataLoader`, model `SentenceBERT`/`DenseRetrievalExactSearch` và `runs/_shared/common.py::score_and_save`; không viết lại metrics.

```bash
python new/data/evaluate.py --data new/data/datasets/vihealthqa --split test \
  --model /path/to/sentence-transformers-checkpoint \
  --output new/data/results/my-model/vihealthqa

# Chấm ranking đã tạo từ bất kỳ mô hình nào:
python new/data/evaluate.py --data new/data/datasets/zalo-legal --split test \
  --results /path/to/results.json --output new/data/results/my-model/zalo-legal
```

`results.json` có dạng `{"query-id": {"document-id": 0.95}}`, bao phủ đúng các query trong split đánh giá. Output gồm `metrics.json` (nDCG, MAP, Recall, Precision, MRR tại k=1,3,5,10,100), `ranking.run.trec`, và `evaluation.json` chứa hash dữ liệu/ranking đầu vào. Thư mục output phải chưa tồn tại. Khi dùng model embedding, hãy áp dụng prompt/query instruction đúng cấu hình mô hình; chế độ mặc định ở đây dùng SentenceBERT với cosine, không tự thêm instruction theo từng mô hình. Các runner cũ đang giới hạn danh sách dataset; dùng entrypoint mới này để đánh giá bốn dataset bổ sung.

Đọc bằng code hiện có:

```python
from beir.datasets.data_loader import GenericDataLoader
corpus, queries, qrels = GenericDataLoader(
    "new/data/datasets/vihealthqa"
).load(split="test")
```

## Kiểm thử

```bash
python -m pytest new/data/test_prepare.py -q
```

Test chạy offline: SQuAD/JSONL, CSV nhiều dòng, CSConDa Parquet, gộp context, bỏ unanswerable, giữ multi-positive, title-only, xung đột IDs, qrels không hợp lệ, checksum và chống ghi đè output.
