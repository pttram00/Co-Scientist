"""Central registry các tool schema (theo định nghĩa `tools` của Anthropic
/v1/messages) cho các agent gọi `complete_json_tool`.

Mỗi TOOL_* là một dict định nghĩa:
  - name: tên tool (model gọi tool này qua tool_choice)
  - description: mô tả ngắn gọn
  - input_schema: JSON Schema mô tả các field + type + required.

Schema ở đây ĐỒNG BỘ với `JSON_SCHEMA_HINT` trong từng agent file — hint vẫn
được giữ trong prompt làm mô tả thêm giúp model sinh rationale/comments chất
hơn, còn schema thật (ép presence/type) do tool định nghĩa ở đây.
"""
from __future__ import annotations

# ----------------------------------------------------------- RankingAgent
# Mỗi trận so sánh trả về một verdict (A/B thắng) kèm lý do.
TOOL_MATCH_VERDICT = {
    "name": "record_match_verdict",
    "description": "Ghi nhận kết quả phân xử giữa hai giả thuyết A và B.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["A", "B"],
                "description": "Giả thuyết thắng trong trận so sánh.",
            },
            "rationale": {
                "type": "string",
                "description": "Lý do phân xử, 2-4 câu, dựa trên tính đúng đắn, "
                               "tính mới, khả năng kiểm chứng và mức độ tác động.",
            },
        },
        "required": ["winner", "rationale"],
    },
}

# -------------------------------------------------------- ReflectionAgent
# Đánh giá 1 giả thuyết theo 3 tiêu chí số (0-10) kèm mô phỏng/nhận xét.
# simulation_notes/comments là optional (tương ứng `.get()` trong agent).
TOOL_REVIEW = {
    "name": "record_hypothesis_review",
    "description": "Ghi nhận đánh giá phản biện một giả thuyết khoa học.",
    "input_schema": {
        "type": "object",
        "properties": {
            "simulation_notes": {
                "type": "string",
                "description": "Mô phỏng cơ chế từng bước, chỉ ra điểm yếu nếu có.",
            },
            "correctness": {
                "type": "number",
                "description": "Điểm tính đúng đắn, 0-10.",
            },
            "novelty": {
                "type": "number",
                "description": "Điểm tính mới, 0-10.",
            },
            "feasibility": {
                "type": "number",
                "description": "Điểm khả năng kiểm chứng, 0-10.",
            },
            "comments": {
                "type": "string",
                "description": "Nhận xét tổng hợp, gợi ý cải thiện nếu có.",
            },
        },
        "required": ["correctness", "novelty", "feasibility"],
    },
}

# --------------------------------------------- GenerationAgent / EvolutionAgent
# Sinh/Mội/cải tiến 1 giả thuyết mới. content+rationale là bắt buộc; suggested_experiment
# optional (tương ứng `.get()` trong agent — nhưng khi nêu required sẽ ép model điền,
# giúp tránh crash hiện tại). Vì code dùng `.get()` cho suggested_experiment, giữ
# nó ngoài required để khớp behaviour cũ.
TOOL_HYPOTHESIS = {
    "name": "record_hypothesis",
    "description": "Ghi nhận một giả thuyết khoa học mới (content + rationale + thí nghiệm).",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Phát biểu giả thuyết, 1-3 câu, cụ thể và kiểm chứng được.",
            },
            "rationale": {
                "type": "string",
                "description": "Lý do / cơ chế đề xuất, 3-6 câu.",
            },
            "suggested_experiment": {
                "type": "string",
                "description": "Thí nghiệm/phân tích đề xuất để kiểm chứng, 1-3 câu.",
            },
        },
        "required": ["content", "rationale"],
    },
}

# ----------------------------------- GenerationAgent — query expansion sub-task
TOOL_QUERIES = {
    "name": "record_search_queries",
    "description": "Ghi nhận danh sách câu truy vấn tìm kiếm bài báo khoa học.",
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 câu truy vấn, mỗi câu 3-7 từ, đa góc nhìn.",
            },
        },
        "required": ["queries"],
    },
}

# ----------------------------------- GenerationAgent — paper selection sub-task
TOOL_PAPER_IDS = {
    "name": "record_selected_paper_ids",
    "description": "Ghi nhận danh sách id các paper được chọn làm grounding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "5-8 id paper, đúng id thuộc pool đã cho.",
            },
        },
        "required": ["ids"],
    },
}

# ------------------------------------------------------- MetaReviewAgent
# Tổng hợp mẫu hình + phản hồi cho từng agent. patterns và feedback đều optional
# trong code (`.get(..., default)`), nên `required` rỗng để khớp behaviour.
TOOL_FEEDBACK = {
    "name": "record_review_feedback",
    "description": "Ghi nhận các mẫu hình phản biện lặp lại và phản hồi cải thiện "
                   "từng agent ở vòng lặp tiếp theo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Các mẫu hình phản biện lặp lại.",
            },
            "feedback": {
                "type": "object",
                # Mỗi key là tên agent, value là string phản hồi (có thể rỗng).
                "additionalProperties": {"type": "string"},
                "description": "Phản hồi cụ thể cho từng agent "
                               "(generation_agent, proximity_agent, reflection_agent, "
                               "ranking_agent, evolution_agent).",
            },
        },
        "required": [],
    },
}
