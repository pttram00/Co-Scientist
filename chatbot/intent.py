"""Intent classification: phân loại câu tự nhiên của user thành 1 trong 5 kind.

kind:
- "new":      yêu cầu tạo hướng nghiên cứu mới.
- "review":   nhận xét về hướng có sẵn (kèm/sú kèm rerun_step).
- "rerun":    yêu cầu chạy lại 1 bước (agent) cụ thể, không kèm nhận xét.
- "question": câu hỏi RAG về kết quả hiện có.
- "unknown":  không phân loại được.

Comment trong "review" được LLM tự phân bổ: vào hypothesis (target_hypothesis_id)
hay vào agent (target_agent) tuỳ ngữ nghĩa — chatbot sẽ dùng trường đó để gọi
add_user_comment hoặc add_agent_feedback_comment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from llm.client import LLMClient
from memory.context_memory import ContextMemory


@dataclass
class IntentResult:
    kind: str = "unknown"
    target_hypothesis_id: Optional[str] = None
    target_agent: Optional[str] = None
    comment_text: Optional[str] = None
    rerun_step: Optional[str] = None
    research_goal: Optional[str] = None
    constraints: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# 6 bước hợp lệ của hệ thống — dùng để validate target_agent / rerun_step.
VALID_STEPS = ["generation", "proximity", "reflection", "ranking", "evolution", "meta_review"]

SYSTEM_PROMPT = """Bạn phân loại ý định của 1 câu tiếng Việt để điều phối hệ thống
multi-agent nghiên cứu khoa học. Đọc câu của user + context (danh sách id giả
thuyết và tên agent đang có). Trả JSON đúng schema, không kèm giải thích.

Các kind:
- "new":      user muốn BẮT ĐẦU nghiên cứu mới ( tạo hướng + chạy hệ thống).
- "review":   user NHẬN XÉT về 1 giả thuyết hoặc 1 agent đã chạy. Có thể kèm yêu
              cầu chạy lại bước nào đó → đặt rerun_step.
- "rerun":    user YÊU CẦU CHẠY LẠI 1 bước cụ thể, không kèm nhận xét nội dung.
- "question": user HỎI thông tin (muốn được trả lời, không yêu cầu thay đổi).
- "unknown":  không đủ thông tin để phân biệt.

Quy tắc phân bổ:
- Nếu user chê/nhắc 1 GIẢ THUYẾT cụ thể (vd có id như a1, be4c9fa2...) → đặt
  target_hypothesis_id = id đó, comment_text = nội dung nhận xét.
- Nếu user chê 1 AGENT cụ thể (reflection/ranking/evolution/generation/
  proximity/meta_review) → đặt target_agent = tên agent, comment_text = nhận xét.
- Cả hai đều có thể kèm rerun_step (tên 1 agent) nếu user yêu cầu chạy lại.
- Cú pháp ngắn "<id> <text>" hoặc "<a1> abc" → kind="review",
  target_hypothesis_id = id, comment_text = text.

Chỉ đủa các bước hợp lệ: generation, proximity, reflection, ranking, evolution,
meta_review. Nếu user nói "chạy lại X" mà X không hợp lệ → xem như unknown
hoặc đặt rerun_step=null + comment_text=nguyên câu.
"""

SCHEMA = """Schema JSON:
{
  "kind": "new|review|rerun|question|unknown",
  "target_hypothesis_id": "<id hoặc null>",
  "target_agent": "<agent_name hoặc null>",
  "comment_text": "<text hoặc null>",
  "rerun_step": "<agent_name hoặc null>",
  "research_goal": "<text hoặc null>"
}"""


def _build_context(memory: ContextMemory) -> str:
    """Tóm tắt state hiện tại cho LLM: id giả thuyết + agent đang có."""
    ids = ", ".join(memory.hypotheses.keys()) or "(chưa có giả thuyết)"
    return (
        f"Context hiện tại:\n"
        f"- Các agent trong hệ thống: {', '.join(VALID_STEPS)}\n"
        f"- Các id giả thuyết đang có: {ids}\n"
        f"- Vòng lặp hiện tại: {memory.iteration}"
    )


def _normalize(data: dict, memory: ContextMemory) -> IntentResult:
    """Chuẩn hoá + validate kết quả từ LLM. Chuẩn hoá id giả thuyết nếu cần."""
    kind = data.get("kind", "unknown")
    if kind not in ("new", "review", "rerun", "question", "unknown"):
        kind = "unknown"

    hid = data.get("target_hypothesis_id")
    if hid and memory.hypotheses and hid not in memory.hypotheses:
        # Thử khớp không phân biệt hoa thường / tiền tố.
        low = hid.lower()
        match = next((h for h in memory.hypotheses if h.lower() == low), None)
        if match:
            hid = match
        else:
            match = next((h for h in memory.hypotheses if low in h.lower()), None)
            if match:
                hid = match

    agent = data.get("target_agent")
    if agent:
        agent_low = agent.lower()
        agent = next((s for s in VALID_STEPS if s == agent_low or s in agent_low), None)

    rerun = data.get("rerun_step")
    if rerun:
        rerun_low = rerun.lower()
        rerun = next((s for s in VALID_STEPS if s == rerun_low or s in rerun_low), None)

    return IntentResult(
        kind=kind,
        target_hypothesis_id=hid,
        target_agent=agent,
        comment_text=data.get("comment_text"),
        rerun_step=rerun,
        research_goal=data.get("research_goal"),
    )


async def classify_intent(user_text: str, llm: LLMClient,
                            memory: ContextMemory) -> IntentResult:
    """1 LLM call: đọc câu user + state → IntentResult. Fail → kind="unknown"."""
    if not user_text.strip():
        return IntentResult(kind="unknown")
    user = f"{_build_context(memory)}\n\nCâu của user: {user_text}\n\n{SCHEMA}"
    try:
        data = await llm.complete_json(SYSTEM_PROMPT, user)
        if not isinstance(data, dict):
            return IntentResult(kind="unknown")
        return _normalize(data, memory)
    except Exception as e:
        # LLM fail (vd lỗi kết nối) → unknown, chatbot sẽ tự hỏi lại user.
        return IntentResult(kind="unknown", comment_text=f"LLM lỗi: {e}")
