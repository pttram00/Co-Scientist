"""SafetyAgent: kiểm tra an toàn cho mục tiêu nghiên cứu trước khi chạy
(Supp. Note 7 — "Initial research goal safety review"): mục tiêu bị đánh giá là
không an toàn sẽ bị từ chối, hệ thống không sinh giả thuyết nào.

Kiểm tra an toàn cho TỪNG giả thuyết nằm trong initial review của ReflectionAgent;
theo dõi an toàn của hướng nghiên cứu nằm trong MetaReviewAgent.
"""
from __future__ import annotations

from typing import Tuple

from agents.base_agent import BaseAgent, as_bool

SYSTEM_PROMPT = """Bạn là SafetyAgent trong hệ thống multi-agent hỗ trợ nghiên
cứu khoa học. Nhiệm vụ: đánh giá xem mục tiêu nghiên cứu có thể dẫn tới nghiên
cứu nguy hiểm, phi đạo đức hoặc gây hại hay không, ví dụ: tạo hoặc tăng độc
lực/khả năng lây của mầm bệnh, vũ khí sinh học/hoá học/phóng xạ, né tránh cơ chế
kiểm soát an toàn sinh học, thí nghiệm trên người không có đồng thuận, gây hại
có chủ đích cho con người hoặc môi trường. Nghiên cứu khoa học thông thường (cơ
chế bệnh, tái định vị thuốc, tìm đích điều trị, khoa học cơ bản...) là AN TOÀN.
Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "safe": true | false,
  "reason": "<lý do ngắn gọn, 1-3 câu>"
}"""


class UnsafeResearchGoalError(RuntimeError):
    """Mục tiêu nghiên cứu bị SafetyAgent từ chối."""


class SafetyAgent(BaseAgent):
    name = "safety_agent"

    async def run(self) -> Tuple[bool, str]:
        """Trả (safe, reason). Thiếu trường "safe" -> coi là không an toàn (fail-closed)."""
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}\n\n"
            f"{JSON_SCHEMA_HINT}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        return as_bool(data.get("safe"), default=False), str(data.get("reason", "")).strip()
