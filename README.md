# Research Co-Scientist

Research Co-Scientist là một framework đa tác tử (multi-agent) dùng để sinh, phản biện và cải tiến giả thuyết nghiên cứu khoa học theo chu trình có cấu trúc. Dự án được xây dựng theo mô hình điều phối một Orchestrator và nhiều agent chuyên biệt, tất cả chia sẻ cùng một bộ nhớ ngữ cảnh để duy trì trạng thái của cả hệ thống.

## Mục tiêu của dự án

Dự án hiện tại tập trung vào các nhiệm vụ sau:

- Sinh ra các giả thuyết nghiên cứu mới từ một mục tiêu cụ thể.
- Phản biện và lọc các giả thuyết trùng lặp hoặc yếu về tính khả thi.
- So sánh và xếp hạng giả thuyết bằng một cơ chế tương tự Elo.
- Cải tiến các giả thuyết tốt nhất theo nhiều chiến lược khác nhau.
- Xuất báo cáo tổng quan cuối cùng và lưu trạng thái toàn bộ hệ thống.

## Kiến trúc hiện tại

Hệ thống gồm một Orchestrator điều phối ba pha liên tiếp trong mỗi vòng lặp:

1. Pha 1 — Generation & Proximity
   - GenerationAgent: sinh giả thuyết mới bằng ba chiến lược: literature-grounded, self-debate và assumption analysis.
   - ProximityAgent: tính độ tương đồng giữa các giả thuyết và đánh dấu các cặp gần như trùng lặp dựa trên ngưỡng `duplicate_threshold`.

2. Pha 2 — Reflection & Ranking
   - ReflectionAgent: phản biện từng giả thuyết theo ba tiêu chí: correctness, novelty và feasibility.
   - RankingAgent: tổ chức các trận đấu cặp đôi giữa giả thuyết, chọn cặp ưu tiên theo đồ thị proximity và cập nhật Elo cho từng giả thuyết.

3. Pha 3 — Evolution & Meta Review
   - EvolutionAgent: tạo các biến thể mới từ các giả thuyết top-rank bằng các chiến lược simplify, analogy và combine.
   - MetaReviewAgent: tổng hợp các nhận xét phản biện và tạo phản hồi cho các agent ở vòng tiếp theo; ở cuối chu trình thì sinh báo cáo tổng quan.

Toàn bộ hệ thống dùng ContextMemory làm lớp trung tâm lưu trữ:

- pool giả thuyết
- review của từng giả thuyết
- đồ thị proximity
- lịch sử các trận đấu
- ghi chú meta-review và feedback cho từng agent
- số vòng lặp hiện tại

## Cấu trúc thư mục

```text
research_coscientist/
├── agents/                  # các agent xử lý từng phần của quy trình
│   ├── base_agent.py
│   ├── generation_agent.py
│   ├── proximity_agent.py
│   ├── reflection_agent.py
│   ├── ranking_agent.py
│   ├── evolution_agent.py
│   └── meta_review_agent.py
├── llm/                     # wrapper gọi mô hình LLM qua giao diện Anthropic-compatible
├── memory/                  # ContextMemory và trạng thái dùng chung
├── models/                  # model dữ liệu: Hypothesis, Review, MatchResult
├── output/                  # thư mục lưu báo cáo và state JSON
├── config.py                # cấu hình chung cho LLM và orchestrator
├── main.py                  # CLI entrypoint
├── orchestrator.py          # điều phối vòng lặp 3 pha
├── requirements.txt         # phụ thuộc Python
└── README.md                # tài liệu dự án
```

## Mô hình dữ liệu chính

Các class quan trọng trong code hiện tại:

- Hypothesis: đại diện cho một giả thuyết khoa học, gồm nội dung, cơ chế, mục tiêu nghiên cứu, nguồn agent sinh ra, chiến lược, review, Elo và trạng thái.
- Review: lưu kết quả phản biện của ReflectionAgent với các điểm correctness, novelty, feasibility và nhận xét tổng hợp.
- MatchResult: lưu kết quả một trận đấu ranking giữa hai giả thuyết.
- ContextMemory: lớp trung tâm dùng để đọc/ghi trạng thái cho tất cả agent.

## Cấu hình

Các tham số cấu hình được định nghĩa trong [config.py](config.py) và bao gồm:

- LLMConfig
  - model: mặc định là `glm-5.2`
  - max_tokens: mặc định 2000
  - temperature: mặc định 0.7
  - max_retries: mặc định 3
  - max_concurrency: mặc định 5

- OrchestratorConfig
  - n_iterations: số vòng lặp chạy
  - hypotheses_per_iteration: số giả thuyết GenerationAgent sinh mỗi vòng
  - matches_per_iteration: số cặp đấu RankingAgent chạy mỗi vòng
  - top_k_for_evolution: số giả thuyết tốt nhất được EvolutionAgent cải tiến
  - proximity_duplicate_threshold: ngưỡng đánh dấu trùng lặp
  - output_dir: thư mục đầu ra

## Biến môi trường

Dự án đọc biến môi trường từ file `.env` ở thư mục gốc:

```env
ANTHROPIC_AUTH_TOKEN=your_token_here
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

Lưu ý:

- Trong code hiện tại, client LLM dùng giao diện Anthropic-compatible và tự thêm header `Authorization: Bearer <token>`.
- Nếu endpoint hoặc token không hợp lệ, quá trình chạy sẽ báo lỗi tại bước gọi model.

## Cài đặt

Khuyến nghị tạo môi trường ảo trước khi cài đặt:

```bash
python -m venv .venv
source .venv/bin/activate
# Trên Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

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
- `--model`: tên model dùng để gọi LLM
- `--output-dir`: thư mục lưu đầu ra

## Quy trình chạy thực tế

Mỗi lần chạy một iteration, orchestrator sẽ thực hiện theo thứ tự sau:

1. GenerationAgent tạo mới các giả thuyết.
2. ProximityAgent so sánh từng cặp giả thuyết và lưu vào đồ thị proximity.
3. ReflectionAgent đánh giá các giả thuyết đang active.
4. RankingAgent chọn cặp đấu và cập nhật Elo.
5. EvolutionAgent tạo các biến thể mới từ các giả thuyết top-rank.
6. MetaReviewAgent tạo feedback cho các agent và cuối cùng sinh báo cáo tổng quan.

## Đầu ra

Sau khi chạy, hệ thống sẽ tạo:

- `output/final_report.md`: báo cáo tổng quan cuối cùng được viết bằng Markdown.
- `output/state.json`: lưu toàn bộ trạng thái hệ thống, bao gồm mục tiêu, giả thuyết, review, lịch sử đấu, feedback và meta-notes.

> File `state.json` hiện đang được lưu lại để tiện kiểm tra và mở rộng tính năng resume/continue trong các phiên bản sau.

## Mở rộng và tùy chỉnh

Một số điểm có thể mở rộng tiếp:

- Thay đổi model hoặc endpoint tại `.env` và [config.py](config.py).
- Điều chỉnh các tham số như số vòng lặp, số giả thuyết mỗi vòng, ngưỡng trùng lặp hoặc số trận đấu.
- Thêm agent mới hoặc thay thế logic hiện tại trong các module trong thư mục [agents](agents).
- Mở rộng [memory/context_memory.py](memory/context_memory.py) để hỗ trợ load lại trạng thái từ file một cách tự động hơn.

## Lưu ý quan trọng

- Dự án này phụ thuộc vào một endpoint LLM có thể gọi được và token hợp lệ.
- Nếu endpoint không phản hồi hoặc token không đúng, chương trình sẽ dừng lại tại bước gọi mô hình.
- Mặc dù có tính năng lưu state, khung hiện tại chưa tự động resume từ `state.json` khi chạy lại; việc này có thể được mở rộng trong tương lai.

