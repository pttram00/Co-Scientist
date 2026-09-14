
# Research Co-Scientist

Research Co-Scientist là một framework đa tác tử (multi-agent) dùng để sinh, phản biện và cải tiến giả thuyết nghiên cứu khoa học theo chu trình có cấu trúc. Dự án được xây dựng theo mô hình điều phối một Orchestrator và nhiều agent chuyên biệt, tất cả chia sẻ cùng một bộ nhớ ngữ cảnh để duy trì trạng thái của cả hệ thống.

Lịch sử thay đổi và đối chiếu với tài liệu gốc: xem [CHANGELOG.md](CHANGELOG.md).

## Mục tiêu của dự án

Dự án hiện tại tập trung vào các nhiệm vụ sau:

- Sinh ra các giả thuyết nghiên cứu mới từ một mục tiêu cụ thể, có grounding từ bài báo khoa học.
- Phản biện (có tra cứu tài liệu) và lọc các giả thuyết trùng lặp, yếu hoặc không an toàn.
- So sánh và xếp hạng giả thuyết bằng một cơ chế tương tự Elo.
- Cải tiến các giả thuyết tốt nhất theo nhiều chiến lược khác nhau.
- Xuất báo cáo tổng quan cuối cùng và lưu trạng thái toàn bộ hệ thống.

## Kiến trúc hiện tại

Trước khi chạy, hệ thống nạp model embedding cho ProximityAgent, rồi **SafetyAgent** kiểm tra mục tiêu nghiên cứu; mục tiêu không an toàn bị từ chối.

Sau đó Orchestrator điều phối ba pha liên tiếp trong mỗi vòng lặp:

1. Pha 1 — Generation & Proximity
   - GenerationAgent: sinh giả thuyết mới bằng ba chiến lược: literature-grounded, self-debate và assumption analysis. Từ vòng 2, prompt có thêm danh sách giả thuyết đã có và research overview của Meta-review để tránh lặp ý tưởng (research expansion).
   - ProximityAgent: encode mỗi giả thuyết bằng model embedding đa ngôn ngữ (sentence-transformers, chạy trên máy) và tính cosine cho mọi cặp mới — **không tốn lượt gọi API**. Cặp có cosine ≥ `proximity_duplicate_threshold` chỉ là *nghi trùng*: LLM phải xác nhận (tối đa `proximity_max_llm_checks_per_iteration` cặp mỗi vòng) thì mới đánh dấu trùng lặp. Nhờ vậy hai giả thuyết ngược chiều tác động (tăng cường / ức chế cùng một cơ chế), vốn có vector gần như giống nhau, không bị loại nhầm.

2. Pha 2 — Reflection & Ranking
   - ReflectionAgent: review mỗi giả thuyết **một lần**, theo hai bước:
     - Initial review (không tra cứu): chấm nhanh correctness, novelty, feasibility và kiểm tra an toàn. Không qua thì chuyển sang `archived`, không an toàn thì `unsafe`.
     - Full review (có tra cứu bài báo qua Retriever): tóm tắt phần đã biết trong tài liệu, mô phỏng cơ chế từng bước, chấm lại 3 tiêu chí.
   - RankingAgent: tổ chức các trận đấu cặp đôi giữa các giả thuyết đã có full review. Giả thuyết mới luôn được đấu ít nhất một trận; sau đó ưu tiên cặp gần nhau trên đồ thị proximity và cặp có Elo liền kề. Cập nhật Elo cho từng giả thuyết.

3. Pha 3 — Evolution & Meta Review
   - EvolutionAgent: tạo các biến thể mới từ các giả thuyết top-rank (đã review và đã đấu) bằng các chiến lược simplify, analogy và combine.
   - MetaReviewAgent: tổng hợp nhận xét phản biện và lý do phân xử các trận đấu, tạo feedback cho mọi agent ở vòng tiếp theo, cập nhật research overview và cảnh báo nếu hướng nghiên cứu có vấn đề an toàn.

Sau vòng lặp cuối, **pha kết thúc** review và xếp hạng các giả thuyết do Evolution vừa tạo, rồi MetaReviewAgent mới viết báo cáo tổng quan (gồm tổng quan, tổng hợp bài báo nền tảng, giả thuyết nổi bật, kết luận). Báo cáo chỉ gồm giả thuyết đã có full review và đã đấu ít nhất một trận.

Toàn bộ hệ thống dùng ContextMemory làm lớp trung tâm lưu trữ:

- pool giả thuyết
- review của từng giả thuyết
- đồ thị proximity
- lịch sử các trận đấu
- ghi chú meta-review và feedback cho từng agent
- pool bài báo dùng làm grounding
- research overview (hướng nghiên cứu đã/chưa khám phá)
- cảnh báo an toàn
- số vòng lặp hiện tại

## Cấu trúc thư mục

```text
research_coscientist/
├── agents/                  # các agent xử lý từng phần của quy trình
│   ├── base_agent.py
│   ├── safety_agent.py
│   ├── generation_agent.py
│   ├── proximity_agent.py
│   ├── reflection_agent.py
│   ├── ranking_agent.py
│   ├── evolution_agent.py
│   └── meta_review_agent.py
├── llm/                     # wrapper gọi mô hình LLM qua giao diện Anthropic-compatible
├── memory/                  # ContextMemory và trạng thái dùng chung
├── models/                  # model dữ liệu: Hypothesis, Review, MatchResult, Paper
├── retrieval/               # tra cứu bài báo từ arXiv, Semantic Scholar, OpenAlex
├── output/                  # thư mục lưu báo cáo và state JSON
├── .env.example             # mẫu file .env
├── config.py                # cấu hình chung cho LLM, retriever, embedding và orchestrator
├── main.py                  # CLI entrypoint
├── orchestrator.py          # điều phối vòng lặp 3 pha
├── requirements.txt         # phụ thuộc Python
├── CHANGELOG.md             # lịch sử thay đổi
└── README.md                # tài liệu dự án
```

## Mô hình dữ liệu chính

Các class quan trọng trong code hiện tại:

- Hypothesis: đại diện cho một giả thuyết khoa học, gồm nội dung, cơ chế, mục tiêu nghiên cứu, nguồn agent sinh ra, chiến lược, review, Elo và trạng thái (`active`, `duplicate`, `archived`, `unsafe`).
- Review: lưu kết quả phản biện của ReflectionAgent (`initial` hoặc `full`) với các điểm correctness, novelty, feasibility, nhận xét tổng hợp và danh sách tài liệu đã tra cứu (full review).
- MatchResult: lưu kết quả một trận đấu ranking giữa hai giả thuyết.
- Paper: một bài báo tra cứu được, dùng làm grounding.
- ContextMemory: lớp trung tâm dùng để đọc/ghi trạng thái cho tất cả agent.

## Cấu hình

Các tham số cấu hình được định nghĩa trong [config.py](config.py) và bao gồm:

- LLMConfig
  - model: mặc định là `GLM-5.2`
  - max_tokens: mặc định 2000
  - temperature: mặc định 0.7
  - max_retries: mặc định 3 (áp dụng cả khi model trả JSON lỗi)
  - max_concurrency: mặc định 5

- RetrieverConfig
  - k_per_source, pool_size, min_papers_for_grounding, papers_per_strategy: tra cứu grounding cho GenerationAgent
  - review_max_queries, review_k_per_source, review_papers_per_review: tra cứu trong full review của ReflectionAgent
  - max_concurrent_requests: số request đồng thời tối đa tới các nguồn ngoài (mặc định 4)

- EmbeddingConfig (ProximityAgent)
  - model_name: mặc định `paraphrase-multilingual-MiniLM-L12-v2` (đa ngôn ngữ, hợp với giả thuyết tiếng Việt)
  - device: `cpu` hoặc `cuda`
  - batch_size: số câu encode cùng lúc

- OrchestratorConfig
  - n_iterations: số vòng lặp chạy
  - hypotheses_per_iteration: số giả thuyết GenerationAgent sinh mỗi vòng
  - matches_per_iteration: số cặp đấu RankingAgent chạy mỗi vòng (có thể nhiều hơn để mọi giả thuyết mới đều được đấu)
  - top_k_for_evolution: số giả thuyết tốt nhất được EvolutionAgent cải tiến
  - proximity_duplicate_threshold: ngưỡng cosine để coi một cặp là nghi trùng (mặc định 0.80; nên hiệu chỉnh lại khi đổi model embedding)
  - proximity_max_llm_checks_per_iteration: số cặp nghi trùng tối đa hỏi LLM xác nhận mỗi vòng (mặc định 10)
  - output_dir: thư mục đầu ra

## Biến môi trường

Sao chép `.env.example` thành `.env` ở thư mục gốc (cùng thư mục với `config.py`) rồi điền giá trị:

```env
ANTHROPIC_AUTH_TOKEN=your_token_here
ANTHROPIC_BASE_URL=https://ai.boltz.one/ai/anthropic
# Tuỳ chọn:
SEMANTIC_SCHOLAR_API_KEY=
OPENALEX_MAILTO=
```

Lưu ý:

- Trong code hiện tại, client LLM dùng giao diện Anthropic-compatible và tự thêm header `Authorization: Bearer <token>`.
- Nếu endpoint hoặc token không hợp lệ, quá trình chạy sẽ báo lỗi tại bước gọi model.
- **Không ghi key thật vào code.** Mọi key chỉ đặt trong `.env` (đã nằm trong `.gitignore`).

## Cài đặt

Khuyến nghị tạo môi trường ảo trước khi cài đặt:

```bash
python -m venv .venv
source .venv/bin/activate
# Trên Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# torch bản CPU từ index chính thức của PyTorch (bản trên PyPI từng lỗi DLL trên Windows)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Lần chạy đầu tiên, sentence-transformers sẽ tự tải model embedding (vài trăm MB) về cache HuggingFace (`~/.cache/huggingface`). Nếu chưa cài `sentence-transformers` / `torch`, chương trình dừng ngay ở bước nạp model, trước khi gọi LLM lượt nào.

## Chạy chương trình

Chạy từ thư mục gốc của dự án:

```bash
python main.py --goal "Tìm cơ chế phân tử mới để ức chế sự già hóa tế bào thần kinh" \
               --iterations 3 \
               --hypotheses-per-iteration 6 \
               --matches-per-iteration 10
```

### Các tham số CLI hiện có

- `--goal`: mục tiêu nghiên cứu chính
- `--constraints`: ràng buộc hoặc bối cảnh bổ sung
- `--iterations`: số vòng lặp chạy
- `--hypotheses-per-iteration`: số giả thuyết sinh mỗi vòng
- `--matches-per-iteration`: số trận đấu ranking mỗi vòng
- `--top-k-for-evolution`: số giả thuyết top được cải tiến
- `--proximity-max-llm-checks`: số cặp nghi trùng tối đa ProximityAgent hỏi LLM xác nhận mỗi vòng
- `--embedding-model`: tên model sentence-transformers dùng cho ProximityAgent
- `--model`: tên model dùng để gọi LLM
- `--output-dir`: thư mục lưu đầu ra

Nếu mục tiêu nghiên cứu bị SafetyAgent từ chối, chương trình in lý do và thoát với mã 1.

## Quy trình chạy thực tế

0. Nạp model embedding, rồi SafetyAgent kiểm tra mục tiêu nghiên cứu.

Mỗi iteration, orchestrator thực hiện theo thứ tự sau (lưu `state.json` sau mỗi pha):

1. GenerationAgent tạo mới các giả thuyết.
2. ProximityAgent tính cosine embedding cho các cặp mới, hỏi LLM xác nhận các cặp nghi trùng và cập nhật đồ thị proximity.
3. ReflectionAgent review các giả thuyết chưa được review (initial → full).
4. RankingAgent chọn cặp đấu và cập nhật Elo.
5. EvolutionAgent tạo các biến thể mới từ các giả thuyết top-rank.
6. MetaReviewAgent tạo feedback cho các agent và cập nhật research overview.

Sau iteration cuối: pha kết thúc (proximity → review → ranking cho giả thuyết còn tồn đọng), rồi MetaReviewAgent sinh báo cáo tổng quan.

Một lời gọi LLM lỗi (kể cả JSON lỗi sau khi đã thử lại) chỉ làm bỏ qua phần việc đó; giả thuyết review lỗi hoặc cặp nghi trùng chưa xác nhận được sẽ được thử lại ở lượt sau.

## Đầu ra

Sau khi chạy, hệ thống sẽ tạo:

- `output/final_report.md`: báo cáo tổng quan cuối cùng được viết bằng Markdown. Nếu lời gọi LLM viết báo cáo lỗi, hệ thống ghi bản tổng hợp tự động từ top giả thuyết.
- `output/state.json`: lưu toàn bộ trạng thái hệ thống, bao gồm mục tiêu, giả thuyết, review, đồ thị proximity, lịch sử đấu, feedback, meta-notes, research overview và cảnh báo an toàn.

> File `state.json` hiện đang được lưu lại để tiện kiểm tra và mở rộng tính năng resume/continue trong các phiên bản sau.

## Mở rộng và tùy chỉnh

Một số điểm có thể mở rộng tiếp:

- Thay đổi model hoặc endpoint tại `.env` và [config.py](config.py).
- Điều chỉnh các tham số như số vòng lặp, số giả thuyết mỗi vòng, ngưỡng trùng lặp hoặc số trận đấu.
- Đổi model embedding qua `--embedding-model`, hoặc truyền backend embedding khác qua tham số `encoder` của `ProximityAgent`.
- Thêm agent mới hoặc thay thế logic hiện tại trong các module trong thư mục [agents](agents).
- Mở rộng [memory/context_memory.py](memory/context_memory.py) để hỗ trợ load lại trạng thái từ file một cách tự động hơn.

Các phần của tài liệu gốc chưa được triển khai được liệt kê ở cuối [CHANGELOG.md](CHANGELOG.md).

## Lưu ý quan trọng

- Dự án này phụ thuộc vào một endpoint LLM có thể gọi được và token hợp lệ.
- Nếu endpoint không phản hồi hoặc token không đúng, chương trình sẽ dừng lại tại bước gọi mô hình.
- ReflectionAgent tra cứu bài báo cho từng giả thuyết, nên số request tới arXiv / Semantic Scholar / OpenAlex tăng theo số giả thuyết. Nên đặt `SEMANTIC_SCHOLAR_API_KEY` trong `.env` để tránh bị giới hạn tốc độ.
- Mặc dù có tính năng lưu state, khung hiện tại chưa tự động resume từ `state.json` khi chạy lại; việc này có thể được mở rộng trong tương lai.
