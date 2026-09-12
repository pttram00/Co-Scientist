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
    """Xử lý 1 dòng comment của user. Trả thông báo kết quả (chuỗi tiếng Việt)."""
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

    scores = await _score_comment(llm, comment_text)
    memory.add_user_review(hid, comment_text, scores)
    memory.save(state_path)
    return (
        f"✓ Đã ghi nhận xét vào [{hid}] — điểm chấm: "
        f"correctness {scores['correctness']:.1f}/novelty {scores['novelty']:.1f}"
        f"/feasibility {scores['feasibility']:.1f}. "
        f"Sẽ ảnh hưởng Reflection/Ranking/Evolution ở vòng sau."
    )
