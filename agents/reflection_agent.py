"""Pha 2 — ReflectionAgent: phản biện từng giả thuyết còn hoạt động."""
from __future__ import annotations

import asyncio
from typing import List

from agents.base_agent import BaseAgent
from models.hypothesis import Hypothesis, Review

SYSTEM_PROMPT = """Bạn là ReflectionAgent — một nhà phản biện khoa học
nghiêm khắc nhưng công tâm trong hệ thống multi-agent hỗ trợ nghiên cứu.
Nhiệm vụ: đánh giá một giả thuyết theo 3 tiêu chí (0-10):
- correctness: tính đúng đắn / hợp lý về mặt khoa học
- novelty: tính mới so với hiểu biết hiện tại của lĩnh vực
- feasibility: khả năng kiểm chứng bằng thực nghiệm/phân tích trong thực tế
Hãy MÔ PHỎNG cơ chế được đề xuất từng bước để phát hiện lỗ hổng logic trước
khi chấm điểm. Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "simulation_notes": "<mô phỏng cơ chế từng bước, chỉ ra điểm yếu nếu có>",
  "correctness": <0-10>,
  "novelty": <0-10>,
  "feasibility": <0-10>,
  "comments": "<nhận xét tổng hợp, gợi ý cải thiện nếu có>"
}"""


class ReflectionAgent(BaseAgent):
    name = "reflection_agent"

    async def _review_one(self, h: Hypothesis) -> Review:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Giả thuyết: {h.content}\nCơ chế đề xuất: {h.rationale}\n"
            f"Thí nghiệm đề xuất: {h.suggested_experiment or '(chưa có)'}\n\n"
            f"{JSON_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        return Review(
            reviewer=self.name,
            review_type="full",
            correctness=float(data["correctness"]),
            novelty=float(data["novelty"]),
            feasibility=float(data["feasibility"]),
            comments=f"{data.get('simulation_notes', '')}\n{data.get('comments', '')}".strip(),
        )

    async def run(self) -> List[Hypothesis]:
        active = self.memory.get_active_hypotheses()
        reviews = await asyncio.gather(*[self._review_one(h) for h in active])
        for h, r in zip(active, reviews):
            h.reviews.append(r)
        return active
