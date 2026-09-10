"""Pha 1 — ProximityAgent: đo độ tương đồng giữa các giả thuyết, dựng đồ thị
proximity, gắn cờ trùng lặp để các pha sau (Ranking, Evolution) không lãng
phí tài nguyên so sánh/cải tiến các giả thuyết gần như giống hệt nhau.

Tối ưu chi phí:
    - Chỉ chấm các cặp CHƯA có trong proximity graph (kết quả được cache giữa các
      vòng) thay vì chấm lại toàn bộ cặp mỗi vòng.
    - Tối đa max_pairs cặp / lượt. Nếu nhiều hơn, ưu tiên cặp có độ trùng từ vựng
      cao (Jaccard trên token) vì đó là cặp có khả năng trùng lặp cao nhất. Cặp
      chưa được chấm vẫn là ứng viên ở lượt sau.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
from typing import List, Optional, Set, Tuple

from agents.base_agent import BaseAgent
from models.hypothesis import Hypothesis, HypothesisStatus

logger = logging.getLogger("proximity_agent")

SYSTEM_PROMPT = """Bạn là ProximityAgent trong hệ thống multi-agent hỗ trợ
nghiên cứu khoa học. Nhiệm vụ: chấm điểm độ tương đồng NGỮ NGHĨA (không chỉ
từ ngữ) giữa hai giả thuyết khoa học, dựa trên: cơ chế đề xuất, đối tượng
nghiên cứu, và cách kiểm chứng. Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "similarity": <số thực 0.0-1.0, 1.0 = trùng lặp hoàn toàn về ý tưởng>,
  "reason": "<giải thích ngắn 1 câu>"
}"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(h: Hypothesis) -> Set[str]:
    # Bỏ token 1 ký tự; giữ token 2 ký tự vì nhiều âm tiết tiếng Việt chỉ dài 2 ("tế", "bào").
    return {t for t in _TOKEN_RE.findall(f"{h.content} {h.rationale}".lower()) if len(t) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ProximityAgent(BaseAgent):
    name = "proximity_agent"

    async def _score_pair(self, a: Hypothesis, b: Hypothesis) -> Tuple[str, str, float]:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Giả thuyết A: {a.content}\nCơ chế A: {a.rationale}\n\n"
            f"Giả thuyết B: {b.content}\nCơ chế B: {b.rationale}\n\n"
            f"{JSON_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        sim = max(0.0, min(1.0, float(data["similarity"])))
        return a.id, b.id, sim

    async def run(
        self, duplicate_threshold: float = 0.85, max_pairs: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        active = self.memory.get_active_hypotheses()
        # Chỉ các cặp chưa chấm (cache trong proximity graph).
        candidates = [
            (a, b) for a, b in itertools.combinations(active, 2)
            if not self.memory.has_proximity(a.id, b.id)
        ]
        if not candidates:
            return []

        if max_pairs is not None and len(candidates) > max_pairs:
            toks = {h.id: _tokens(h) for h in active}
            candidates.sort(key=lambda p: _jaccard(toks[p[0].id], toks[p[1].id]), reverse=True)
            logger.info("Proximity: %d cặp mới, chỉ chấm %d cặp có độ trùng từ vựng cao nhất",
                        len(candidates), max_pairs)
            candidates = candidates[:max_pairs]

        results = await asyncio.gather(
            *[self._score_pair(a, b) for a, b in candidates], return_exceptions=True
        )

        scored: List[Tuple[str, str, float]] = []
        for (a, b), res in zip(candidates, results):
            if isinstance(res, Exception):
                # Không ghi vào graph -> cặp này sẽ được chấm lại ở lượt sau.
                logger.warning("Proximity: chấm cặp %s-%s lỗi, bỏ qua: %s", a.id, b.id, res)
                continue
            id_a, id_b, sim = res
            self.memory.set_proximity(id_a, id_b, sim)
            scored.append(res)
            # Chỉ xử lý khi cả 2 còn active: nếu 1 bên đã bị đánh dấu trùng lặp ở cặp
            # trước thì không kéo thêm bên kia xuống theo.
            if (sim >= duplicate_threshold
                    and a.status == HypothesisStatus.ACTIVE
                    and b.status == HypothesisStatus.ACTIVE):
                # giữ lại giả thuyết có elo cao hơn, đánh dấu bản còn lại là trùng lặp
                # để loại khỏi tournament/evolution
                loser = b if a.elo_rating >= b.elo_rating else a
                self.memory.mark_status(loser.id, HypothesisStatus.DUPLICATE)

        return scored
