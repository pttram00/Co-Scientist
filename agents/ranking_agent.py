"""Pha 2 — RankingAgent: tổ chức tournament so sánh cặp (Elo) giữa các giả
thuyết. Ưu tiên ghép cặp các giả thuyết "gần nhau" theo đồ thị proximity
(so sánh 2 ý tưởng tương tự cho nhiều thông tin hơn là 2 ý tưởng bất kỳ),
và ghép cặp có elo gần nhau để trận đấu cạnh tranh hơn.
"""
from __future__ import annotations

import asyncio
import random
from typing import List, Tuple

from agents.base_agent import BaseAgent
from llm.tool_schemas import TOOL_MATCH_VERDICT
from models.hypothesis import Hypothesis, MatchResult

SYSTEM_PROMPT = """Bạn là RankingAgent, đóng vai một hội đồng phản biện khoa
học phân xử giữa 2 giả thuyết. Hãy so sánh dựa trên: tính đúng đắn, tính
mới, khả năng kiểm chứng, và mức độ tác động nếu đúng. Trả lời bằng JSON."""

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
    được quy ước trước(1200) nên có thể bị lệch so với thực tế.
    Điểm elo này có thể nói là một giá trị đánh giá tổng hợp về một giả thuyết dựa trên các trận đấu.
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    if a_wins is True:
        score_a = 1.0
    elif a_wins is False:
        score_a = 0.0
    else:
        score_a = 0.5
    score_b = 1.0 - score_a
    new_a = rating_a + K_FACTOR * (score_a - expected_a)
    new_b = rating_b + K_FACTOR * (score_b - expected_b)
    return new_a, new_b


class RankingAgent(BaseAgent):
    name = "ranking_agent"

    def _select_pairs(self, n_matches: int) -> List[Tuple[Hypothesis, Hypothesis]]:
        """ Chọn các cặp giả thuyết phù hợp để so sánh với đầu vào là số trận đấu mong muốn. """
        active = self.memory.get_active_hypotheses()
        if len(active) < 2:
            return []

        pairs: List[Tuple[Hypothesis, Hypothesis]] = []              # Tạo danh sách các cặp giả thuyết để so sánh
        by_id = {h.id: h for h in active}                            # Tạo từ điển ánh xạ id của giả thuyết đến đối tượng giả thuyết 

        # Ưu tiên ghép theo proximity graph (là những cặp gần nhau trong graph-những cặp có độ tương đồng cao)
        for h in active:
            # [:2] là cách để lấy được hai giá trị đầu tiên của danh sách - hai giả thuyết tương đồng cao
            for neighbor_id, sim in self.memory.neighbors(h.id, min_similarity=0.3)[:2]:
                if neighbor_id in by_id and len(pairs) < n_matches:
                    pairs.append((h, by_id[neighbor_id]))

        # Bù thêm bằng ghép ngẫu nhiên nếu chưa đủ số trận 
        # Tuy nhiên nó sẽ không cần thiết nếu như số lượng giả thuyết lớn và tránh tốn token khi gọi LLM thì khồng cần thiết 
        attempts = 0
        while len(pairs) < n_matches and attempts < n_matches * 5:
            a, b = random.sample(active, 2)
            if a.id != b.id:
                pairs.append((a, b))
            attempts += 1

        return pairs[:n_matches]

    async def _run_match(self, a: Hypothesis, b: Hypothesis) -> MatchResult:
        user = (
            f"Mục tiêu nghiên cứu: {self.memory.research_goal}\n\n"
            f"Giả thuyết A: {a.content}\nCơ chế A: {a.rationale}\n"
            f"Điểm phản biện A (correctness/novelty/feasibility trung bình): "
            f"{a.average_score('correctness'):.1f}/{a.average_score('novelty'):.1f}/"
            f"{a.average_score('feasibility'):.1f}\n\n"
            f"Giả thuyết B: {b.content}\nCơ chế B: {b.rationale}\n"
            f"Điểm phản biện B (correctness/novelty/feasibility trung bình): "
            f"{b.average_score('correctness'):.1f}/{b.average_score('novelty'):.1f}/"
            f"{b.average_score('feasibility'):.1f}\n\n"
            f"{JSON_SCHEMA_HINT}"
        )
        # Cách 2 — tool calling ép schema: winner chỉ có thể là "A"/"B",
        # rationale luôn là string. Không còn kẹt `data["winner"]` trần.
        try:
            data = await self.llm.complete_json_tool(SYSTEM_PROMPT, user, tool=TOOL_MATCH_VERDICT)
            winner = a if str(data.get("winner", "")).strip().upper().startswith("A") else b
            return MatchResult(
                hypothesis_a_id=a.id,
                hypothesis_b_id=b.id,
                winner_id=winner.id,
                rationale=str(data.get("rationale", "")),
            )
        except Exception as e:
            # Fallback an toàn: 1 trận lỗi không crash cả tournament.
            # winner_id=None -> _elo_update nhánh score_a=0.5 (hoà).
            print(f"[RankingAgent] _run_match lỗi, dùng kết quả hoà: {e}")
            return MatchResult(
                hypothesis_a_id=a.id,
                hypothesis_b_id=b.id,
                winner_id=None,
                rationale=f"(LLM lỗi, kết quả dự phòng hoà: {e})",
            )

    async def run(self, n_matches: int = 10) -> List[MatchResult]:
        pairs = self._select_pairs(n_matches)
        if not pairs:
            return []

        results = await asyncio.gather(*[self._run_match(a, b) for a, b in pairs])

        for (a, b), result in zip(pairs, results):
            # winner_id=None (fallback hoà) -> a_wins=None -> _elo_update nhánh 0.5 (hoà).
            a_wins = None if result.winner_id is None else result.winner_id == a.id
            new_a, new_b = _elo_update(a.elo_rating, b.elo_rating, a_wins)
            a.elo_rating, b.elo_rating = new_a, new_b
            # Sau khi cập nhật elo rating thì tăng số trận đấu đã chơi của hai giả thuyết
            a.matches_played += 1
            b.matches_played += 1
            self.memory.record_match(result)

        return list(results)
