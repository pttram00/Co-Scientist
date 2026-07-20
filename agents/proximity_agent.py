"""Pha 1 — ProximityAgent: đo độ tương đồng giữa các giả thuyết, dựng đồ thị
proximity, gắn cờ trùng lặp để các pha sau (Ranking, Evolution) không lãng
phí tài nguyên so sánh/cải tiến các giả thuyết gần như giống hệt nhau.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import List, Tuple

from agents.base_agent import BaseAgent
from models.hypothesis import Hypothesis, HypothesisStatus

SYSTEM_PROMPT = """Bạn là ProximityAgent trong hệ thống multi-agent hỗ trợ
nghiên cứu khoa học. Nhiệm vụ: chấm điểm độ tương đồng NGỮ NGHĨA (không chỉ
từ ngữ) giữa hai giả thuyết khoa học, dựa trên: cơ chế đề xuất, đối tượng
nghiên cứu, và cách kiểm chứng. Trả lời bằng JSON. Ví dụ định dạng:
{
    "score" : 0.55,
    "reason" : "Cả hai đều được ..."
}
"""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "similarity": <số thực 0.0-1.0, 1.0 = trùng lặp hoàn toàn về ý tưởng>,
  "reason": "<giải thích ngắn 1 câu>"
}"""


class ProximityAgent(BaseAgent):
    name = "proximity_agent"

    async def _score_pair(self, a: Hypothesis, b: Hypothesis) -> Tuple[str, str, float]:
        user = (
            f"Giả thuyết A: {a.content}\nCơ chế A: {a.rationale}\n\n"
            f"Giả thuyết B: {b.content}\nCơ chế B: {b.rationale}\n\n"
            f"{JSON_SCHEMA_HINT}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        return a.id, b.id, float(data["similarity"])

    async def run(self, duplicate_threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        active = self.memory.get_active_hypotheses()
        pairs = list(itertools.combinations(active, 2))
        if not pairs:
            return []

        results = await asyncio.gather(*[self._score_pair(a, b) for a, b in pairs])

        for id_a, id_b, sim in results:
            self.memory.set_proximity(id_a, id_b, sim)
            if sim >= duplicate_threshold:
                # giữ lại giả thuyết có elo cao hơn (hoặc mới hơn nếu bằng nhau),
                # đánh dấu bản còn lại là trùng lặp để loại khỏi tournament/evolution
                ha, hb = self.memory.hypotheses[id_a], self.memory.hypotheses[id_b]
                loser = hb if ha.elo_rating >= hb.elo_rating else ha
                if loser.status == HypothesisStatus.ACTIVE:
                    self.memory.mark_status(loser.id, HypothesisStatus.DUPLICATE)

        return results
