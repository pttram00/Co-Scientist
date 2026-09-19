"""User comment: LLM chấm điểm 0-10 → Review → memory.

Comment của user được LLM phân tích thành 3 score (correctness/novelty/
feasibility) rồi lưu như một Review (review_type="user_comment"). Từ đó tự chảy
vào các agent ở vòng sau (Evolution đọc h.reviews[-2:], Reflection/ MetaReview
đọc tất cả reviews) — không cần sửa agent nào.

Hai điểm vào:
- add_review_comment: gắn nhận xét vào 1 giả thuyết (đã biết id + text, dùng khi
  intent classifier đã xác định target_hypothesis_id + comment_text).
- add_agent_feedback_comment: gắn nhận xét vào agent (memory.agent_feedback).
"""
from __future__ import annotations

from llm.client import LLMClient
from memory.context_memory import ContextMemory

SCORE_SYSTEM = """Bạn dựa vào nhận xét của người dùng về 1 giả thuyết khoa
học để chấm điểm theo 3 tiêu chí, thang 0-10 (số thực):

- correctness: mức độ nhận xét cho rằng giả thuyết ĐÚNG ĐẮN về logic/cơ chế.
  Nhận xét phản biện (ý kiến ngược) → điểm thấp; đồng tình → điểm cao.
- novelty: mức độ nhận xét đánh giá tính MỚI của giả thuyết. Nhận xét cho rằng ý
  tưởng mới → điểm cao; cho rằng nhàm chán/trùng lặp → điểm thấp.
- feasibility: mức độ nhận xét đánh giá KHẢ NĂNG KIỂM CHỨNG. Nhận xét cho rằng khả
  thi → điểm cao; cho rằng khó kiểm chứng → điểm thấp.

Trả JSON đúng schema, không kèm giải thích."""

SCORE_SCHEMA = """Schema JSON trả về:
{
  "correctness": <0.0-10.0>,
  "novelty": <0.0-10.0>,
  "feasibility": <0.0-10.0>
}"""


async def _score_comment(llm: LLMClient, comment_text: str) -> dict:
    """Gọi LLM chấm 3 score 0-10 từ text comment. Fail → mặc định 5.0 (trung tính)."""
    user = f'Nhận xét: "{comment_text}"\n\n{SCORE_SCHEMA}'
    try:
        data = await llm.complete_json(SCORE_SYSTEM, user)
        return {
            "correctness": float(data.get("correctness", 5.0)),
            "novelty": float(data.get("novelty", 5.0)),
            "feasibility": float(data.get("feasibility", 5.0)),
        }
    except Exception:
        # LLM fail (vd: lỗi kết nối) → dùng điểm trung tính, vẫn giữ comment text.
        return {"correctness": 5.0, "novelty": 5.0, "feasibility": 5.0}


async def add_review_comment(
    memory: ContextMemory,
    llm: LLMClient,
    hypothesis_id: str,
    comment_text: str,
    state_path: str,
) -> str:
    """Gắn nhận xét vào 1 giả thuyết (đã biết id + text, không cần parse). Dùng khi
    intent classifier đã xác định target_hypothesis_id + comment_text."""
    if hypothesis_id not in memory.hypotheses:
        return f"Không tìm thấy giả thuyết id '{hypothesis_id}'."
    if not comment_text:
        return "Nội dung nhận xét trống."

    scores = await _score_comment(llm, comment_text)
    memory.add_user_review(hypothesis_id, comment_text, scores)
    memory.save(state_path)
    return (
        f"✓ Đã ghi nhận xét vào [{hypothesis_id}] — điểm chấm: "
        f"correctness {scores['correctness']:.1f}/novelty {scores['novelty']:.1f}"
        f"/feasibility {scores['feasibility']:.1f}. "
        f"Sẽ ảnh hưởng Reflection/Ranking/Evolution ở vòng sau."
    )


async def add_agent_feedback_comment(
    memory: ContextMemory,
    agent_name: str,
    comment_text: str,
    state_path: str,
) -> str:
    """Gắn nhận xét của user vào agent (memory.agent_feedback[agent_name]). Không
    cần chấm điểm — chỉ text, để agent đó đọc ở lần chạy sau qua feedback_block().
    agent_name chuẩn hoá: khớp tên bước hợp lệ (generation_agent/proximity_agent/reflection_agent/
    ranking_agent/evolution_agent/meta_review_agent)."""
    valid = ["generation_agent", "proximity_agent", "reflection_agent", "ranking_agent", "evolution_agent", "meta_review_agent"]
    head = (agent_name or "").lower()
    head = next((s for s in valid if s == head or s in head), None)
    if not head:
        return f"Agent '{agent_name}' không hợp lệ. Hợp lệ: {', '.join(valid)}."
    if not comment_text:
        return "Nội dung nhận xét trống."

    memory.add_agent_feedback(head, f"[user] {comment_text}")
    memory.save(state_path)
    return (
        f"✓ Đã ghi nhận xét vào agent '{head}'. Agent này sẽ đọc nhận xét của bạn "
        f"ở lần chạy lại kế tiếp (qua feedback_block)."
    )
