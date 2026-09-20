# Changelog

## Giao diện Gradio (`app.py`)

- `app.py` — giao diện 4 tab: chạy nghiên cứu (log trực tiếp, nút dừng), kết quả (bảng giả thuyết +
  chi tiết từng giả thuyết khi bấm vào dòng), báo cáo (`final_report.md` + nút tải), nạp lại
  `state.json` cũ. Log của các agent được thu qua một `logging.Handler` riêng và đẩy dần ra ô nhật ký.
- `sim_backend.py` (mới) — LLM, retriever và encoder giả để chạy thử toàn bộ luồng mà không cần API
  key, không gọi mạng và không cần tải model embedding. Nội dung sinh ra là giả, có gắn nhãn.
- `requirements.txt` — thêm `gradio`.
- Đã kiểm thử: chạy pipeline qua hàm của giao diện ở chế độ mô phỏng (bảng, chi tiết, báo cáo, file tải
  về đều đúng), chặn mục tiêu rỗng, nạp lại `state.json`, dựng giao diện, và khởi động máy chủ thật
  (HTTP 200). Chưa thử ở chế độ thật vì máy chưa có token LLM.

## Gộp `origin/main` (9a122ef) vào nhánh thử `DaoNhatAnh-thu-main`

Main có 5 commit mới (Proximity chuyển sang embedding, cấu trúc báo cáo cuối, sửa tham số
`temperature` trong LLM client, `.env.example`, bổ sung comment). Gộp vào nhánh này gây xung
đột ở 7 file; phần dưới ghi lại cách xử lý.

### Proximity: embedding + LLM xác nhận (`agents/proximity_agent.py`)

- Lấy hướng của main: encode giả thuyết bằng sentence-transformers (chạy local) và tính
  cosine, thay cho chấm từng cặp bằng LLM → bước tính độ tương đồng không tốn lượt gọi API.
- Bổ sung so với bản trên main:
  - Model mặc định đổi `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2`: giả
    thuyết viết tiếng Việt, còn model cũ chỉ huấn luyện cho tiếng Anh (tokenizer uncased bỏ dấu).
  - Cosine ≥ `proximity_duplicate_threshold` (mặc định 0.80) chỉ là **nghi trùng**; LLM xác
    nhận (`_confirm_duplicate`) thì mới đánh dấu DUPLICATE. Lý do: embedding gần như không phân
    biệt được ý ngược nghĩa — trong test, "Tăng cường autophagy…" và "Ức chế autophagy…" có
    cosine 0.85 và bản cũ sẽ loại nhầm một trong hai.
  - Tối đa `proximity_max_llm_checks_per_iteration` (mặc định 10) cặp/lượt, ưu tiên cosine cao;
    cặp chưa kịp hỏi hoặc hỏi lỗi được hỏi lại ở lượt sau, cặp đã kết luận không hỏi lại.
  - Chỉ tính cosine cho cặp mới (vector cache theo id, cạnh đã có trong graph không tính lại).
  - Chỉ đánh dấu trùng khi cả hai giả thuyết còn active.
  - `sentence-transformers` được import lười và encoder thay được
    (`ProximityAgent(..., encoder=...)`): import dự án không kéo theo torch, test chạy không cần torch.
- `orchestrator.py` — nạp model embedding (`ensure_ready()`) **trước** safety check: thiếu
  thư viện thì báo lỗi ngay, chưa tốn lượt gọi LLM nào.
- `config.py` / `main.py` — bỏ `proximity_max_pairs_per_iteration` / `--proximity-max-pairs`;
  thêm `proximity_max_llm_checks_per_iteration`, `--proximity-max-llm-checks`, `--embedding-model`.
- Số lời gọi LLM (đo bằng LLM giả, trung bình 3 lần chạy): cấu hình mặc định 356 → 201
  (riêng Proximity 195 → 32); cấu hình thử nhỏ 45 → 28.

### Xử lý các file xung đột khác

| File | Cách gộp |
|---|---|
| `orchestrator.py` | Giữ luồng của nhánh (safety, pha kết thúc, Reflection có retriever) + `ProximityAgent(..., self.config.embedding)` |
| `agents/meta_review_agent.py` | Gộp cả hai: đọc debate, research overview, cảnh báo an toàn, báo cáo fallback (nhánh) + cấu trúc báo cáo cố định và `_collect_papers` (main); thêm mục "Cảnh báo an toàn" vào cấu trúc báo cáo |
| `agents/ranking_agent.py` | Giữ bản của nhánh (thay đổi trên main là comment cho code cũ) + docstring/comment còn áp dụng được |
| `agents/base_agent.py` | Giữ cả `as_bool` / `truncate` (nhánh) và tham số `embedding` (main) |
| `memory/context_memory.py`, `models/hypothesis.py` | Code của nhánh + comment/docstring của main |

`llm/client.py`, `retrieval/retriever.py`, `agents/generation_agent.py`, `requirements.txt`,
`.env.example` được git tự gộp.

### Bảo mật

- `config.py` trên main ghi cứng một Semantic Scholar API key làm giá trị mặc định, và key
  này đã được push lên GitHub. Nhánh này bỏ giá trị đó, chỉ đọc `SEMANTIC_SCHOLAR_API_KEY` từ
  `.env`. Key cũ cần được thu hồi và tạo key mới.

### Đã kiểm tra (offline, LLM + retriever + encoder giả)

1. Chạy trọn 3 vòng: báo cáo chỉ gồm giả thuyết đã review và đã đấu; mọi cặp active đều có
   cạnh proximity; không hỏi LLM lại cặp đã kết luận; không còn 2 giả thuyết active trùng ý.
2. 25% lời gọi LLM lỗi ngẫu nhiên + báo cáo lỗi → vẫn chạy xong, dùng báo cáo fallback.
3. Mục tiêu không an toàn bị từ chối trước khi sinh giả thuyết.
4. Hai giả thuyết ngược chiều tác động (cosine 0.85) không bị loại; bản trùng thật bị loại.
5. Giới hạn số cặp hỏi LLM mỗi lượt; các lượt sau hỏi tiếp cho đến khi loại hết bản trùng.
6. Chưa cài sentence-transformers → dừng ngay với thông báo rõ ràng, 0 lời gọi LLM.

Chưa chạy với model embedding thật (máy chưa cài torch) và chưa hiệu chỉnh ngưỡng 0.80 trên
dữ liệu thật.

## Đối chiếu với tài liệu Co-Scientist (co_sci_docs.pdf)

Các thay đổi dưới đây sửa theo thứ tự ưu tiên rút ra khi đối chiếu code với bài báo
"Accelerating scientific discovery with Co-Scientist" (Methods tr. 29–38, pseudocode
Supp. Note 8, prompt Supp. Note 9, an toàn Supp. Note 7).

### Bước 1 — Sửa lỗi luồng

**1a. Báo cáo cuối không còn chứa giả thuyết chưa được đánh giá**

- Trước: giả thuyết do Evolution tạo ở vòng cuối chưa được review hay đấu trận nào
  nhưng vẫn giữ Elo mặc định 1200, nên có thể vượt các giả thuyết đã thua (Elo < 1200)
  và lọt vào top của báo cáo với điểm 0.0.
- Sau:
  - `orchestrator.py` — thêm `run_closing_phase()`: sau vòng lặp cuối, chạy proximity →
    review → tournament cho các giả thuyết chưa review, rồi mới viết báo cáo.
  - `memory/context_memory.py` — `get_top_k(k, evaluated_only=False)`: khi
    `evaluated_only=True` chỉ lấy giả thuyết có full review và đã đấu ≥ 1 trận. Dùng cho
    báo cáo cuối (`MetaReviewAgent.run_final_report`) và Evolution.
  - `agents/ranking_agent.py` — `_select_pairs()` viết lại: chỉ giả thuyết có full review
    mới vào tournament; mỗi giả thuyết mới (chưa đấu trận nào) chắc chắn có ≥ 1 trận
    (đối thủ: láng giềng gần nhất trên proximity graph, không có thì Elo gần nhất); sau đó
    top-rank đấu láng giềng, rồi bù bằng cặp Elo liền kề, cuối cùng mới ngẫu nhiên. Không
    ghép trùng cặp (a,b)/(b,a). Số trận có thể vượt `matches_per_iteration` khi số giả
    thuyết mới nhiều hơn.

**1b. Một lỗi LLM/JSON không còn làm dừng cả lần chạy**

- `llm/client.py` — `complete_json()` gọi lại tối đa `max_retries` lần khi parse JSON lỗi;
  hàm mới `_parse_json()` cắt phần lời dẫn trước/sau JSON trước khi parse.
- `agents/{reflection,proximity,ranking,evolution}_agent.py` — `asyncio.gather(...,
  return_exceptions=True)`: phần tử lỗi được log và bỏ qua. Giả thuyết review lỗi và
  cặp proximity lỗi sẽ được thử lại ở lượt sau.
- `agents/meta_review_agent.py` — feedback lỗi thì bỏ qua vòng đó; báo cáo lỗi thì dùng
  bản tổng hợp tự động từ top giả thuyết.
- `orchestrator.py` — lưu `state.json` sau **mỗi pha** (trước đây chỉ cuối mỗi vòng).

### Bước 2 — Reflection có tra cứu tài liệu + initial review để lọc

- `agents/reflection_agent.py` viết lại theo 2 bước:
  1. **Initial review** (không tool): chấm nhanh 3 tiêu chí, quyết định `pass`, đề xuất
     1–2 query tìm bài báo. Không qua → status `ARCHIVED`.
  2. **Full review** (có tra cứu qua `Retriever`): tóm tắt phần đã biết trong tài liệu
     trước khi chấm novelty, mô phỏng cơ chế từng bước, chấm lại 3 tiêu chí. Tài liệu
     đã dùng lưu ở `Review.references`.
- Mỗi giả thuyết chỉ được review một lần (trước đây review lại toàn bộ giả thuyết active
  mỗi vòng). Prompt Reflection giờ có cả ràng buộc (`constraints`).
- `models/hypothesis.py` — thêm `Review.references`, `Hypothesis.reviews_of()`,
  `Hypothesis.is_reviewed`; `average_score()` ưu tiên điểm full review (initial review
  không tra cứu hay chấm novelty quá cao — ablation trong bài: 6.14 vs 2.38).
- `retrieval/retriever.py` — `search()` nhận `k_per_source`, `pool_size` tuỳ chọn; thêm
  semaphore giới hạn request đồng thời; retry cả khi gặp HTTP 429.
- `config.py` — `RetrieverConfig`: `max_concurrent_requests`, `review_max_queries`,
  `review_k_per_source`, `review_papers_per_review`.

### Bước 3 — Nối feedback, debate và research overview

- `ProximityAgent`, `RankingAgent`, `EvolutionAgent` giờ chèn `feedback_block()` vào prompt
  (trước đây Meta-review có sinh feedback cho các agent này nhưng không ai đọc).
- `RankingAgent` — prompt đưa **nội dung** full review của mỗi giả thuyết và dặn không
  dựa vào điểm số (theo prompt mẫu Supp. Note 9.3), thay vì đưa điểm trung bình.
- `EvolutionAgent` — prompt giờ có mục tiêu nghiên cứu và ràng buộc (trước đây không có).
- `MetaReviewAgent` — đọc thêm lý do phân xử các trận (`match_history`); JSON feedback
  thêm `research_overview` (hướng đã/chưa khám phá), lưu vào `memory.research_overview`.
- `GenerationAgent` — thêm `_expansion_block()` (research expansion): prompt liệt kê các
  giả thuyết đã có + research overview, yêu cầu không lặp ý tưởng cũ.

### Bước 4 — Proximity không tính lại, có giới hạn chi phí

- `memory/context_memory.py` — `proximity_graph` đổi sang `{id: {other_id: sim}}`: ghi
  cạnh không còn tạo bản ghi trùng; thêm `has_proximity()`. `load()` vẫn đọc được
  `state.json` định dạng cũ (list).
- `agents/proximity_agent.py` — chỉ chấm cặp chưa có trong graph; tối đa
  `proximity_max_pairs_per_iteration` cặp/lượt, ưu tiên cặp trùng từ vựng cao (Jaccard);
  cặp chưa chấm vẫn là ứng viên ở lượt sau. Chỉ đánh dấu DUPLICATE khi cả 2 còn active.
- `config.py` / `main.py` — `proximity_max_pairs_per_iteration` (mặc định 60),
  CLI `--proximity-max-pairs`.

### Bước 5 — Kiểm tra an toàn

- `agents/safety_agent.py` (mới) — `SafetyAgent` kiểm tra mục tiêu nghiên cứu trước khi
  chạy; không an toàn → `UnsafeResearchGoalError`, `main.py` in lý do và thoát mã 1.
- Initial review của Reflection có `safe` / `safety_concerns`: giả thuyết không an toàn →
  status mới `UNSAFE` (không vào tournament, không tiến hoá, không vào báo cáo). Thiếu
  trường `safe` được coi là không an toàn (fail-closed).
- Meta-review trả `safety_concerns` cho hướng nghiên cứu hiện tại → log cảnh báo.
- Mọi cảnh báo lưu ở `memory.safety_alerts` (có trong `state.json`) và đưa vào báo cáo cuối.

### Đã kiểm tra

Chạy offline toàn bộ pipeline với LLM + retriever giả (3 vòng, cấu hình mặc định):
báo cáo chỉ gồm giả thuyết đã review và đã đấu; không còn giả thuyết active chưa review
sau pha kết thúc; mỗi cặp proximity chỉ được chấm 1 lần (195 lời gọi so với ~855 trước
đây); chạy xong khi 25% lời gọi LLM lỗi ngẫu nhiên và báo cáo lỗi (dùng bản fallback);
mục tiêu không an toàn bị từ chối trước khi sinh giả thuyết; `state.json` lưu/tải lại
được, kể cả định dạng cũ. Chưa chạy với LLM thật.

### Chưa làm (so với tài liệu)

- **Supervisor agent + hàng đợi task bất đồng bộ** (bước 6) — vẫn là vòng 3 pha tuần tự.
- Tranh luận nhiều lượt: self-debate trong Generation và debate cho các trận top-rank.
- Các loại review khác: deep verification, observation, tournament review.
- Chiến lược Evolution: enhancement through grounding, feasibility improvement, inspiration.
- Parse mục tiêu thành research plan (preferences / attributes / constraints).
- Chuyên gia tham gia vòng lặp (đưa giả thuyết, review thủ công, chỉnh mục tiêu).
- Resume từ `state.json` khi chạy lại.
