"""Pha 3 — EvolutionAgent: cải tiến các giả thuyết top-rank."""
from __future__ import annotations

import asyncio
from typing import List

from agents.base_agent import BaseAgent
from llm.tool_schemas import TOOL_HYPOTHESIS
from models.hypothesis import GenerationStrategy, Hypothesis

SYSTEM_PROMPT = """Bạn là EvolutionAgent trong hệ thống multi-agent hỗ trợ
nghiên cứu khoa học. Nhiệm vụ: tạo ra một giả thuyết MỚI, cải tiến hơn, dựa
trên (các) giả thuyết đầu vào và phản hồi phản biện đã có. Giả thuyết mới
phải giữ được ý tưởng cốt lõi có giá trị nhưng khắc phục điểm yếu đã nêu.
Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "content": "<phát biểu giả thuyết mới>",
  "rationale": "<lý do / cơ chế, đã khắc phục điểm yếu của bản gốc>",
  "suggested_experiment": "<thí nghiệm đề xuất>"
}"""


class EvolutionAgent(BaseAgent):
    name = "evolution_agent"

    def _feedback_summary(self, h: Hypothesis) -> str:
        """ Tóm tắt phản biện gần nhất cho một giả thuyết để đưa vào prompt."""
        if not h.reviews:
            return "(chưa có phản biện)"
        return "\n".join(f"- [{r.review_type}] {r.comments}" for r in h.reviews[-2:])

    async def _simplify(self, h: Hypothesis) -> Hypothesis:
        user = (
            f"Chiến lược: SIMPLIFY — đơn giản hoá giả thuyết sau để dễ kiểm chứng "
            f"hơn, giữ nguyên ý tưởng cốt lõi.\n\n"
            f"Giả thuyết gốc: {h.content}\nCơ chế gốc: {h.rationale}\n"
            f"Phản biện gần nhất:\n{self._feedback_summary(h)}\n\n{JSON_SCHEMA_HINT}"
        )
        data = await self.llm.complete_json_tool(SYSTEM_PROMPT, user, tool=TOOL_HYPOTHESIS)
        return Hypothesis(
            content=data["content"], rationale=data["rationale"],
            research_goal=self.memory.research_goal, source_agent=self.name,
            strategy=GenerationStrategy.EVOLUTION_SIMPLIFY, parent_ids=[h.id],
            suggested_experiment=data.get("suggested_experiment"),
        )

    async def _analogy(self, h: Hypothesis) -> Hypothesis:
        user = (
            f"Chiến lược: OUT-OF-BOX ANALOGY — đề xuất một biến thể táo bạo của "
            f"giả thuyết sau, lấy cảm hứng từ một cơ chế/hiện tượng ở lĩnh vực "
            f"khác (analogical reasoning), nhưng vẫn phải bám mục tiêu nghiên cứu.\n\n"
            f"Giả thuyết gốc: {h.content}\nCơ chế gốc: {h.rationale}\n"
            f"Phản biện gần nhất:\n{self._feedback_summary(h)}\n\n{JSON_SCHEMA_HINT}"
        )
        data = await self.llm.complete_json_tool(SYSTEM_PROMPT, user, tool=TOOL_HYPOTHESIS)
        return Hypothesis(
            content=data["content"], rationale=data["rationale"],
            research_goal=self.memory.research_goal, source_agent=self.name,
            strategy=GenerationStrategy.EVOLUTION_ANALOGY, parent_ids=[h.id],
            suggested_experiment=data.get("suggested_experiment"),
        )

    async def _combine(self, h1: Hypothesis, h2: Hypothesis) -> Hypothesis:
        user = (
            f"Chiến lược: COMBINE — tổng hợp 2 giả thuyết sau thành MỘT giả "
            f"thuyết mới mạnh hơn, kết hợp điểm mạnh của cả hai.\n\n"
            f"Giả thuyết 1: {h1.content}\nCơ chế 1: {h1.rationale}\n\n"
            f"Giả thuyết 2: {h2.content}\nCơ chế 2: {h2.rationale}\n\n{JSON_SCHEMA_HINT}"
        )
        data = await self.llm.complete_json_tool(SYSTEM_PROMPT, user, tool=TOOL_HYPOTHESIS)
        return Hypothesis(
            content=data["content"], rationale=data["rationale"],
            research_goal=self.memory.research_goal, source_agent=self.name,
            strategy=GenerationStrategy.EVOLUTION_COMBINE, parent_ids=[h1.id, h2.id],
            suggested_experiment=data.get("suggested_experiment"),
        )

    async def run(self, top_k: int = 4) -> List[Hypothesis]:
        top = self.memory.get_top_k(top_k)
        if not top:
            return []

        # Tạo ra hai biến thể mới từ mỗi giả thuyết top-k
        tasks = []
        for h in top:
            tasks.append(self._simplify(h))
            tasks.append(self._analogy(h))
        # Nếu có ít nhất 2 giả thuyết top-k thì kết hợp 2 giả thuyết tốt nhất để tạo ra một giả thuyết mới
        if len(top) >= 2:
            tasks.append(self._combine(top[0], top[1]))

        # return_exceptions=True: 1 biến thể lỗi không rớt cả pha Evolution;
        # bỏ biến thể lỗi, vẫn giữ phần thành công (partial, giống GenerationAgent).
        evolved = await asyncio.gather(*tasks, return_exceptions=True)
        for h in evolved:
            if isinstance(h, Exception):
                print(f"[EvolutionAgent] 1 biến thể lỗi, bỏ qua: {h}")
                continue
            self.memory.add_hypothesis(h)
        return [h for h in evolved if not isinstance(h, Exception)]
