"""Intent classification: phân loại câu tự nhiên của user thành 1 trong 6 kind.

kind:
- "new_topic":       đổi chủ đề nghiên cứu → reset memory + chạy full từ đầu.
- "resume_newfinal":  tiếp tục trên state hiện có, chạy thêm iteration để ra
                      final_report mới (tận dụng dữ liệu phiên trước đó, KHÔNG
                      reset). Dùng khi user nói "tạo lại", "sinh lại", "tạo mới
                      dựa trên nhận xét/data cũ", nhưng không đổi chủ đề.
- "review":          nhận xét về 1 giả thuyết hoặc 1 agent đã chạy. Có thể kèm
                      yêu cầu chạy lại bước nào đó → đặt rerun_step.
- "rerun":           yêu cầu chạy lại 1 bước (agent) cụ thể, không kèm nhận xét.
- "question":        câu hỏi RAG về kết quả hiện có.
- "unknown":         không phân loại được.

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

# Các kind hợp lệ.
VALID_KINDS = ["new_topic", "resume_newfinal", "review", "rerun", "question", "unknown"]

SYSTEM_PROMPT = """Bạn phân loại ý định của 1 câu tiếng Việt để điều phối hệ thống
multi-agent nghiên cứu khoa học. Đọc câu của user + context (danh sách id giả
thuyết và tên agent đang có). Trả JSON đúng schema, không kèm giải thích.

Các kind:
- "new_topic":       user muốn BẮT ĐẦU nghiên cứu CHỦ ĐỀ MỚI hoặc ĐỔI CHỦ ĐỀ, bỏ
                     dữ liệu cũ. Dấu hiệu: "tôi muốn nghiên cứu X", "đổi chủ đề
                     sang Y", "nghiên cứu về Z". → đặt research_goal.
- "resume_newfinal": user muốn TẠO FINAL/REPORT MỚI dựa trên dữ liệu CẢ PHIÊN
                     TRƯỚC lẫn mới, KHÔNG đổi chủ đề, KHÔNG reset. Dấu hiệu:
                     "tạo lại", "sinh lại", "tạo mới dựa trên nhận xét của tôi",
                     "chạy thêm để ra báo cáo mới", "dựa trên dữ liệu cũ".
                     Điểm nhận biết: KHÔNG nhắc chủ đề mới; nói về "dựa trên
                     nhận xét/data" hoặc đơn giản "tạo lại". → KHÔNG đặt
                     research_goal.
- "review":         user NHẬN XÉT về 1 giả thuyết hoặc 1 agent đã chạy. Có thể
                     kèm yêu cầu chạy lại bước nào đó → đặt rerun_step.
- "rerun":           user YÊU CẦU CHẠY LẠI 1 bước cụ thể, không kèm nhận xét nội
                     dung.
- "question":       user HỎI thông tin (muốn được trả lời, không yêu cầu thay đổi).
- "unknown":         không đủ thông tin để phân biệt.

Quy tắc phân bổ comment:
- Nếu user chê/nhắc 1 GIẢ THUYẾT cụ thể (vd có id như a1, be4c9fa2...) → đặt
  target_hypothesis_id = id đó, comment_text = nội dung nhận xét.
- Nếu user chê 1 AGENT cụ thể (reflection/ranking/evolution/generation/
  proximity/meta_review) → đặt target_agent = tên agent, comment_text = nhận xét.
- NẾu user nói "có 2 ý tưởng/giả thuyết GIỐNG NHAU/TRÙNG LẠP" mà không chỉ id cụ
  thể → đây là nhận xét về proximity (agent phát hiện trùng lặp): đặt
  target_agent="proximity", comment_text=nguyên câu, kind="review".
  → QUAN TRỌNG: KHÔNG tự đặt rerun_step (để null) — chỉ ghi nhận xét, user tự
  quyết rerun sau.
- Cả review/agent và review/hypothesis đều có thể kèm rerun_step (tên 1 agent)
  nếu user EXPLICIT yêu cầu chạy lại ("chạy lại X", "rerun X"). Nếu user chỉ
  nhận xét mà không yêu cầu chạy lại → rerun_step=null.
- Cú pháp ngắn "<id> <text>" hoặc "<a1> abc" → kind="review",
  target_hypothesis_id = id, comment_text = text.

Chỉ chọn các bước hợp lệ: generation, proximity, reflection, ranking, evolution,
meta_review. Nếu user nói "chạy lại X" mà X không hợp lệ → rerun_step=null +
comment_text=nguyên câu, kind="review".
"""

SCHEMA = """Schema JSON:
{
  "kind": "new_topic|resume_newfinal|review|rerun|question|unknown",
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
    if kind not in VALID_KINDS:
        # Hỗ trợ LLM trả giá cũ "new" → "new_topic" để không phá vỡ.
        if kind == "new":
            kind = "new_topic"
        else:
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
