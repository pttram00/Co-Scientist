"""User comment: parse "<id> <text>" → LLM chấm điểm 0-10 → Review → memory.

Comment của user được LLM phân tích thành 3 score (correctness/novelty/
feasibility) rồi lưu như một Review (review_type="user_comment"). Từ đó tự chảy
vào các agent ở vòng sau (Evolution đọc h.reviews[-2:], Reflection/ MetaReview
đọc tất cả reviews) — không cần sửa agent nào.
"""
from __future__ import annotations

import re
from typing import Tuple

from llm.client import LLMClient
from memory.context_memory import ContextMemory

# Parse "<A1> nội dung nhận xét..." → (hypothesis_id, comment_text).
_COMMENT_RE = re.compile(r"^<\s*([A-Za-z0-9_-]+)\s*>\s*(.+)$", re.DOTALL)

SCORE_SYSTEM = """Bạn chấm điểm MỘT nhận xét của người dùng về 1 giả thuyết khoa
học theo 3 tiêu chí, thang 0-10 (số thực):

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


def parse_comment(raw: str) -> Tuple[str, str] | None:
    """Trả (hypothesis_id, comment_text) hoặc None nếu không khép định dạng."""
    m = _COMMENT_RE.match(raw.strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip()


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


async def add_user_comment(
    memory: ContextMemory,
    llm: LLMClient,
    raw_input: str,
    state_path: str,
) -> str:
    """Xử lý 1 dòng comment của user (cú pháp <id> <text>). Trả thông báo tiếng Việt."""
    parsed = parse_comment(raw_input)
    if not parsed:
        return ("Cú pháp comment: <id> <nội dung>. Ví dụ: <a1> cơ chế chưa thuyết phục.")

    hid, comment_text = parsed
    if not comment_text:
        return "Nội dung nhận xét trống."

    if hid not in memory.hypotheses:
        # Gợi ý id gần đúng (trùng tiền tố) để user khắc phục.
        near = [h_id for h_id in memory.hypotheses if hid.lower() in h_id.lower()]
        hint = f" (có ý gần: {', '.join(near[:3])})" if near else ""
        return f"Không tìm thấy giả thuyết id '{hid}'.{hint}"

    return await add_review_comment(memory, llm, hid, comment_text, state_path)


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
    agent_name chuẩn hoá: khớp tên bước hợp lệ (generation/proximity/reflection/
    ranking/evolution/meta_review)."""
    valid = ["generation", "proximity", "reflection", "ranking", "evolution", "meta_review"]
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
