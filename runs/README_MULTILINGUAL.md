# Bốn mô hình embedding bổ sung

Phạm vi cố định: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`, `trec-covid`, `uit-viquad`, `vihealthqa`, `zalo-legal`. Mỗi model chạy tuần tự chín dataset; supervisor chạy song song bốn model theo yêu cầu mới nhất của người dùng. Tổng cộng **36 cặp model–dataset**.

Luồng tham khảo `runs/ANCE/evaluate_ance.py`: tìm và tái sử dụng dữ liệu → `GenericDataLoader.load(test)` → encode query/corpus → `DenseRetrievalExactSearch` → `EvaluateRetrieval` → metric, TREC ranking và log. Không dùng nhãn test để huấn luyện.

| Model / thư mục | Entrypoint | Chuẩn bị đầu vào và cách chấm |
|---|---|---|
| BGE-M3 | `BGE-M3/evaluate_bge_m3.py` | `BAAI/bge-m3`; dense CLS embedding, không thêm instruction, chuẩn hóa L2. Lượt này không dùng sparse/ColBERT/hybrid của BGE-M3. |
| Multilingual-E5-Large | `Multilingual-E5-Large/evaluate_multilingual_e5.py` | `intfloat/multilingual-e5-large`; thêm `query: ` cho query, `passage: ` cho document, kể cả tiếng Việt; mean pooling theo config checkpoint; chuẩn hóa L2. |
| GreenNode-Embedding | `GreenNode-Embedding/evaluate_greennode.py` | `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`; CLS/Normalize theo checkpoint, không thêm prefix; có thể cần tài khoản HF được duyệt. |
| Vietnamese-BiEncoder | `Vietnamese-BiEncoder/evaluate_vietnamese_biencoder.py` | `bkai-foundation-models/vietnamese-bi-encoder`; PyVi tách từ trước encode trên ba bộ tiếng Việt; sáu bộ tiếng Anh giữ nguyên văn bản. Mean pooling theo checkpoint, chuẩn hóa L2. |

Mã dùng chung: `_shared/multilingual_dense.py`. Tất cả checkpoint cố định bằng commit SHA trong `SPECS`, không tải mã Python tùy chỉnh từ model (`trust_remote_code=False`). Snapshot chỉ tải một định dạng trọng số phù hợp và các file tokenizer/config cần thiết, tránh tải thêm ONNX/OpenVINO hoặc cả hai bản trọng số.

## Giới hạn độ dài và tính so sánh

- BGE-M3, E5 và GreenNode dùng tối đa **512 token** để có cấu hình độ dài được kiểm soát gần với các runner dense cũ. BGE-M3/GreenNode có khả năng native 8.192 token nhưng lượt này không đo cấu hình dài 8.192.
- BKAI dùng tối đa **256 token** đúng cấu hình native PhoBERT của checkpoint.
- Ghép document theo `title + " " + text`, giữ nguyên ID, corpus và qrels; không tự chia passage mới.
- Exact search dùng dot product trên vector đã chuẩn hóa L2, tương đương cosine. Encoder FP32; điểm không được nhân 100 vì phép nhân đó không thay đổi thứ tự truy hồi.
- Không dùng prefix hai lần: áp dụng prefix E5 tường minh rồi tắt default prompt của Sentence Transformers bằng `prompt=""`.
- Batch khởi điểm cho các model lớn là 2; BKAI là 8. Có thể đổi bằng `--batch-size`; corpus chunk 10.000, CPU threads 2/model.

**Giới hạn zero-shot của BKAI:** model card công bố dùng 80% tập train Zalo Legal 2021 khi huấn luyện. Dataset Zalo local có 24 query ID overlap giữa train/test. Vì vậy kết quả trên Zalo không được tuyên bố là đánh giá zero-shot nghiêm ngặt. Model card GreenNode có bảng đánh giá Zalo/ViQuAD, nhưng điểm đó không được chép vào kết quả local.

## Môi trường chạy

Đã tạo `runs/multilingual_environment` bằng `venv --system-site-packages` từ runtime BEIR Python 3.9 hiện có, sau đó cài `pyvi==0.1.1` vào venv. Cách này tận dụng PyTorch/ST hiện có và không thay dependencies của hàng đợi benchmark đang chạy.

Từ thư mục chứa checkout `beir`:

```bash
/hdd2/devbe/.cache/conda_envs/beir/bin/python -m venv --system-site-packages beir/runs/multilingual_environment
beir/runs/multilingual_environment/bin/python -m pip install -r beir/runs/requirements-multilingual.txt
```

## Chạy

```bash
# Bốn model song song; mỗi model chạy chín dataset lần lượt:
beir/runs/multilingual_environment/bin/python -u beir/runs/run_multilingual_models.py --max-parallel 4

# Chạy từng model riêng, hoặc chọn một vài dataset:
beir/runs/multilingual_environment/bin/python beir/runs/BGE-M3/evaluate_bge_m3.py
beir/runs/multilingual_environment/bin/python beir/runs/Multilingual-E5-Large/evaluate_multilingual_e5.py --datasets nfcorpus uit-viquad

# Tải snapshot; hoặc kiểm tra embedding trên CPU, không ghi điểm benchmark:
beir/runs/multilingual_environment/bin/python beir/runs/Vietnamese-BiEncoder/evaluate_vietnamese_biencoder.py --prefetch
beir/runs/multilingual_environment/bin/python beir/runs/BGE-M3/evaluate_bge_m3.py --check-model
```

Không khởi chạy thêm supervisor khi một phiên đã chạy: `.multilingual-queue.lock` ngăn hai queue mới ghi chung dữ liệu. Queue benchmark cũ được giữ nguyên; do người dùng yêu cầu chạy song song, hai queue có thể dùng chung GPU và thời gian chạy sẽ phụ thuộc tải máy.

## GreenNode và xác thực

Nếu gặp `WAITING_ACCESS`, xin quyền tại [model page](https://huggingface.co/GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1), sau đó đăng nhập bằng tài khoản đã được duyệt:

```bash
/hdd2/devbe/.cache/conda_envs/beir/bin/hf auth login
```

Nhập token trong terminal của bạn; không lưu token trong mã, không gửi vào chat. Supervisor kiểm tra thông tin đăng nhập lưu trên máy mỗi 60 giây và tự thử lại GreenNode. Nếu chỉ export `HF_TOKEN` trong một terminal khác sau khi queue khởi chạy, biến đó không truyền ngược vào process đang chạy; nên dùng đăng nhập được lưu bởi HF CLI. Có token nhưng chưa được cấp quyền thì model vẫn chờ; các model khác tiếp tục chạy.

## Tự khôi phục lỗi

1. CUDA OOM trong encode: giảm batch một nửa, dọn CUDA cache và encode lại; không đổi checkpoint, prefix, token limit, corpus hoặc cách chấm. Ghi `recovery.json` và batch thực tế trong `model.json`.
2. Nếu worker bị lỗi hoặc bị dừng: supervisor chạy lại các dataset chưa hoàn tất tối đa ba lượt khôi phục, có backoff; OOM làm giảm batch ở lượt sau và process cũ đã kết thúc trước retry.
3. Các kết quả hoàn tất được tái sử dụng khi input, cấu hình ngữ nghĩa, mã nguồn, phiên bản thư viện và checksum artifact khớp. Thay batch để khắc phục OOM được ghi nhận; không tuyên bố điểm bitwise-identical giữa các batch/phần cứng.
4. Lỗi quyền truy cập: giữ `WAITING_ACCESS` và thử lại sau xác thực, không thay bằng model khác và không tiêu thụ số lượt retry tài nguyên/mạng.
5. Lỗi không thể tự khôi phục được ghi FAILED; queue cuối là INCOMPLETE nếu chưa đủ 36 kết quả đã qua audit. Không tạo điểm giả hoặc bỏ kiểm tra để đánh dấu thành công.

Mã lỗi khác CUDA OOM không bị wrapper nuốt; full traceback được giữ trong log để kiểm tra và sửa nguyên nhân. Hàng đợi không tự viết lại thuật toán hoặc cài phiên bản thư viện tùy ý khi không biết nguyên nhân lỗi.

## Kết quả và theo dõi

`runs/multilingual_queues/latest.json` trỏ đến phiên mới nhất. Thư mục phiên chứa:

- `state.json`: PID supervisor/worker, các lần thử, trạng thái, batch, tiến độ tổng.
- `coverage.csv`: 36 dòng, trạng thái và nguồn kết quả từng cặp.
- `ndcg_at_10.csv`, `recall_at_100.csv`: bảng 9 dataset × 4 model; `_` là chưa có kết quả hợp lệ.
- `inputs.json`, `verified_artifacts.json`: dữ liệu, hash và kiểm tra tính lại metric từ ranking.
- `<model>.try<n>.log`: log tải model và đánh giá của từng lần chạy.

Artifacts gốc vẫn theo cấu trúc ANCE: `runs/<model>/results/<run_id>/<dataset>/{metrics.json,ranking.run.trec,config.json,status.json,data.json,model.json}`. Đầu ra cũ không bị ghi đè. `--check-model` chỉ ghi `model_checks/`, không được đưa vào bảng benchmark.

Supervisor cập nhật báo cáo định kỳ sau khi một dataset hoàn tất; chỉ công nhận SUCCESS khi ranking/metric qua kiểm tra lại. Chỉ trạng thái COMPLETE với 36/36 kết quả mới nghĩa là hoàn tất toàn bộ phạm vi.

## Kiểm thử và nguồn

```bash
beir/runs/multilingual_environment/bin/python beir/runs/test_multilingual_models.py
```

Các test kiểm tra prefix, tách từ, giữ title/text, tokenizer PhoBERT, recovery OOM, normalized retrieval và metric BEIR thật trên fixture nhỏ; không cần tải trọng số để chạy test.

Hướng dẫn encode được đối chiếu với model card của [BGE-M3](https://huggingface.co/BAAI/bge-m3), [E5](https://huggingface.co/intfloat/multilingual-e5-large), [GreenNode](https://huggingface.co/GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1) và [BKAI](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder). [ViRE](https://github.com/longstnguyen/ViRE) được tham khảo làm bối cảnh nghiên cứu; số liệu trong project được chạy và chấm bằng pipeline BEIR local, không chép điểm từ ViRE/model card.
