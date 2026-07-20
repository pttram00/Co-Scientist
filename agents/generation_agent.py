"""Pha 1 — GenerationAgent: sinh giả thuyết khoa học mới."""
from __future__ import annotations

import asyncio
from typing import List

from agents.base_agent import BaseAgent
from models.hypothesis import GenerationStrategy, Hypothesis

SYSTEM_PROMPT = """Bạn là một nhà khoa học giàu kinh nghiệm, đóng vai
GenerationAgent trong một hệ thống multi-agent hỗ trợ nghiên cứu khoa học.
Nhiệm vụ của bạn là đề xuất giả thuyết khoa học MỚI, cụ thể, có thể kiểm
chứng được, bám sát mục tiêu nghiên cứu và các ràng buộc được cung cấp.
Luôn trả lời bằng JSON theo đúng schema được yêu cầu."""

STRATEGY_INSTRUCTIONS = {
    GenerationStrategy.LITERATURE_GROUNDED: (
        "Chiến lược: literature-grounded. Hãy dựa trên các cơ chế/lý thuyết "
        "đã được biết đến trong lĩnh vực liên quan, đề xuất một hướng mở rộng "
        "hợp lý nhưng chưa được kiểm chứng."
    ),
    GenerationStrategy.SELF_DEBATE: (
        "Chiến lược: self-debate. Hãy mô phỏng một cuộc tranh luận ngắn giữa "
        "2 chuyên gia có quan điểm khác nhau về mục tiêu nghiên cứu, sau đó "
        "tổng hợp thành MỘT giả thuyết dung hoà được các phản biện mạnh nhất."
    ),
    GenerationStrategy.ASSUMPTION_ANALYSIS: (
        "Chiến lược: assumption analysis. Hãy liệt kê các giả định ngầm mà "
        "lĩnh vực này thường mặc nhiên chấp nhận, chọn MỘT giả định đáng ngờ "
        "nhất, và đề xuất giả thuyết dựa trên việc đảo ngược/nới lỏng giả định đó."
    ),
}

JSON_SCHEMA_HINT = """Schema JSON trả về (1 object):
{
  "content": "<phát biểu giả thuyết, 1-3 câu, cụ thể và kiểm chứng được>",
  "rationale": "<lý do / cơ chế đề xuất, 3-6 câu>",
  "suggested_experiment": "<thí nghiệm/phân tích đề xuất để kiểm chứng, 1-3 câu>"
}"""


class GenerationAgent(BaseAgent):
    name = "generation_agent"

    async def _generate_one(self, strategy: GenerationStrategy) -> Hypothesis:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}\n\n"
            f"{STRATEGY_INSTRUCTIONS[strategy]}\n\n{JSON_SCHEMA_HINT}"
            f"{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        return Hypothesis(
            content=data["content"],
            rationale=data["rationale"],
            research_goal=self.memory.research_goal,
            source_agent=self.name,
            strategy=strategy,
            suggested_experiment=data.get("suggested_experiment"),
        )

    async def run(self, n_hypotheses: int = 6) -> List[Hypothesis]:
        strategies = list(STRATEGY_INSTRUCTIONS.keys())
        tasks = [
            self._generate_one(strategies[i % len(strategies)])
            for i in range(n_hypotheses)
        ]
        new_hypotheses = await asyncio.gather(*tasks)
        for h in new_hypotheses:
            self.memory.add_hypothesis(h)
        return list(new_hypotheses)
