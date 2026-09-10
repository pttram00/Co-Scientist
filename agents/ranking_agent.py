"""Pha 2 — RankingAgent: tổ chức tournament so sánh cặp (Elo) giữa các giả
thuyết đã có full review.

Thứ tự ưu tiên ghép cặp (Methods, tr. 35):
    1. Giả thuyết mới (chưa đấu trận nào): mỗi cái chắc chắn có ít nhất 1 trận,
       đối thủ là láng giềng gần nhất trên proximity graph, nếu không có thì là
       giả thuyết có Elo gần nhất. Nhờ vậy không còn giả thuyết nào giữ Elo mặc
       định 1200 mà chưa từng được so sánh.
    2. Giả thuyết top-rank đấu với láng giềng gần trên proximity graph (so sánh 2
       ý tưởng tương tự cho nhiều thông tin hơn là 2 ý tưởng bất kỳ).
    3. Bù bằng cặp có Elo liền kề (trận cạnh tranh hơn), cuối cùng mới ghép ngẫu nhiên.
Không ghép trùng cặp (a, b) / (b, a) trong cùng một lượt.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, List, Optional, Set, Tuple

from agents.base_agent import BaseAgent, truncate
from models.hypothesis import Hypothesis, MatchResult

logger = logging.getLogger("ranking_agent")

SYSTEM_PROMPT = """Bạn là RankingAgent, đóng vai một hội đồng phản biện khoa
học phân xử giữa 2 giả thuyết. Hãy so sánh dựa trên: tính đúng đắn, tính
mới, khả năng kiểm chứng, và mức độ tác động nếu đúng. Mỗi giả thuyết kèm một
bản review độc lập; review có thể chứa điểm số — KHÔNG dựa vào điểm số (không so
sánh được giữa các review), hãy dựa vào nội dung lập luận. Trả lời bằng JSON."""

JSON_SCHEMA_HINT = """Schema JSON trả về:
{
  "winner": "A" | "B",
  "rationale": "<lý do phân xử, 2-4 câu>"
}"""

K_FACTOR = 32  # hệ số cập nhật Elo


def _elo_update(rating_a: float, rating_b: float, a_wins: bool) -> Tuple[float, float]:
    """
    Đoạn này là nơi tính toán Elo rating mới cho 2 giả thuyết sau khi đấu xong
    vì Elo rating ban đầu là giá trị mặc định được truyền vào rating_a và rating_b
    được quy ước trước(1200) nên có thể bị lệch so với thực tế .
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    score_a = 1.0 if a_wins else 0.0
    new_a = rating_a + K_FACTOR * (score_a - expected_a)
    new_b = rating_b + K_FACTOR * ((1 - score_a) - (1 - expected_a))
    return new_a, new_b


class RankingAgent(BaseAgent):
    name = "ranking_agent"

    def _select_pairs(self, n_matches: int) -> List[Tuple[Hypothesis, Hypothesis]]:
        # Chỉ giả thuyết đã có full review mới được vào tournament.
        pool = [h for h in self.memory.get_active_hypotheses() if h.is_reviewed]
        if len(pool) < 2:
            return []

        by_id = {h.id: h for h in pool}
        pairs: List[Tuple[Hypothesis, Hypothesis]] = []
        seen: Set[frozenset] = set()

        def add(a: Hypothesis, b: Hypothesis) -> bool:
            key = frozenset((a.id, b.id))
            if a.id == b.id or key in seen:
                return False
            seen.add(key)
            pairs.append((a, b))
            return True

        # (1) Mỗi giả thuyết mới có ít nhất 1 trận. Một trận có thể "phủ" 2 giả thuyết
        # mới cùng lúc nếu chúng là đối thủ của nhau.
        covered: Set[str] = set()
        for h in pool:
            if h.matches_played > 0 or h.id in covered:
                continue
            opponent = self._pick_opponent(h, pool, by_id, seen)
            if opponent is not None and add(h, opponent):
                covered.update((h.id, opponent.id))
        # Số trận có thể vượt n_matches khi có nhiều giả thuyết mới hơn n_matches.
        budget = max(n_matches, len(pairs))

        # (2) Top-rank đấu với láng giềng gần trên proximity graph.
        ranked = sorted(pool, key=lambda x: x.elo_rating, reverse=True)
        for h in ranked:
            for neighbor_id, _ in self.memory.neighbors(h.id, min_similarity=0.3)[:2]:
                if len(pairs) >= budget:
                    break
                if neighbor_id in by_id:
                    add(h, by_id[neighbor_id])

        # (3) Bù bằng cặp Elo liền kề, rồi ghép ngẫu nhiên nếu vẫn chưa đủ số trận.
        for a, b in zip(ranked, ranked[1:]):
            if len(pairs) >= budget:
                break
            add(a, b)
        attempts = 0
        while len(pairs) < budget and attempts < budget * 5:
            a, b = random.sample(pool, 2)
            add(a, b)
            attempts += 1

        return pairs[:budget]

    def _pick_opponent(
        self,
        h: Hypothesis,
        pool: List[Hypothesis],
        by_id: Dict[str, Hypothesis],
        seen: Set[frozenset],
    ) -> Optional[Hypothesis]:
        """Đối thủ cho giả thuyết mới: láng giềng gần nhất trên proximity graph,
        nếu không có thì là giả thuyết có Elo gần nhất."""
        for neighbor_id, _ in self.memory.neighbors(h.id):
            if neighbor_id in by_id and frozenset((h.id, neighbor_id)) not in seen:
                return by_id[neighbor_id]
        others = [o for o in pool if o.id != h.id and frozenset((h.id, o.id)) not in seen]
        if not others:
            return None
        return min(others, key=lambda o: abs(o.elo_rating - h.elo_rating))

    @staticmethod
    def _review_text(h: Hypothesis) -> str:
        full = h.reviews_of("full")
        return truncate(full[-1].comments, 1200) if full else "(chưa có review)"

    async def _run_match(self, a: Hypothesis, b: Hypothesis) -> MatchResult:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n"
            f"Ràng buộc/bối cảnh: {self.memory.constraints or '(không có)'}\n\n"
            f"Giả thuyết A: {a.content}\nCơ chế A: {a.rationale}\n"
            f"Review độc lập của A:\n{self._review_text(a)}\n\n"
            f"Giả thuyết B: {b.content}\nCơ chế B: {b.rationale}\n"
            f"Review độc lập của B:\n{self._review_text(b)}\n\n"
            f"{JSON_SCHEMA_HINT}{self.feedback_block()}"
        )
        data = await self.llm.complete_json(SYSTEM_PROMPT, user)
        verdict = str(data.get("winner", "")).strip().upper()
        if verdict.startswith(("A", "1")):
            winner = a
        elif verdict.startswith(("B", "2")):
            winner = b
        else:
            raise ValueError(f"winner không hợp lệ: {data.get('winner')!r}")
        return MatchResult(
            hypothesis_a_id=a.id,
            hypothesis_b_id=b.id,
            winner_id=winner.id,
            rationale=data.get("rationale", ""),
        )

    async def run(self, n_matches: int = 10) -> List[MatchResult]:
        pairs = self._select_pairs(n_matches)
        if not pairs:
            return []

        results = await asyncio.gather(
            *[self._run_match(a, b) for a, b in pairs], return_exceptions=True
        )

        done: List[MatchResult] = []
        for (a, b), result in zip(pairs, results):
            if isinstance(result, Exception):
                # 1 trận lỗi -> bỏ trận đó, không làm dừng cả tournament.
                logger.warning("Ranking: trận %s vs %s lỗi, bỏ qua: %s", a.id, b.id, result)
                continue
            a_wins = result.winner_id == a.id
            new_a, new_b = _elo_update(a.elo_rating, b.elo_rating, a_wins)
            a.elo_rating, b.elo_rating = new_a, new_b
            a.matches_played += 1
            b.matches_played += 1
            self.memory.record_match(result)
            done.append(result)

        return done
